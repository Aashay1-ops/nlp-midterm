"""The three next-character models.

Every model maps a (batch, time) tensor of character ids to (batch, time,
vocab_size) logits, so training, evaluation and sampling in train.py treat
all three identically.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Model 1: Bigram ---------------------------------------------------------

class BigramLM(nn.Module):
    """Next-character logits looked up from the current character alone.

    The whole model is one vocab_size x vocab_size table: row i holds the
    logits for whatever follows character i. It is trained by gradient
    descent through the same loop as the other two models, so its loss and
    training time are measured the same way.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)
        nn.init.zeros_(self.table.weight)  # start at exactly the uniform guess

    def forward(self, idx):
        return self.table(idx)


def bigram_count_loss(train_data, val_data, vocab_size, alpha=1.0):
    """Closed-form bigram: add-alpha smoothed pair counts, scored on val.

    This is the best any bigram can do on the training counts, so it is a
    check that BigramLM actually converged rather than a fourth model.

    Returns:
        dict: val loss in nats/char, plus how many of the possible pairs never
        occur in training and how many val positions hit one of them.
    """
    pair_ids = train_data[:-1] * vocab_size + train_data[1:]
    counts = torch.bincount(pair_ids, minlength=vocab_size ** 2).float()
    counts = counts.view(vocab_size, vocab_size)
    log_p = torch.log((counts + alpha) / (counts + alpha).sum(dim=1, keepdim=True))
    val_pairs = (val_data[:-1], val_data[1:])
    return {
        "val_loss": -log_p[val_pairs].mean().item(),
        "unseen_pairs": int((counts == 0).sum()),
        "val_positions_with_unseen_pair": int((counts[val_pairs] == 0).sum()),
    }


# --- Model 2: LSTM -----------------------------------------------------------

class LSTMLM(nn.Module):
    """Embedding -> stacked LSTM -> linear head.

    Dropout is applied to the embeddings, between LSTM layers, and to the
    LSTM output, so the model is regularised at roughly as many points as the
    transformer rather than only between layers.
    """

    def __init__(self, vocab_size, n_embd, hidden, n_layer, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.lstm = nn.LSTM(n_embd, hidden, num_layers=n_layer,
                            dropout=dropout if n_layer > 1 else 0.0,
                            batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, idx):
        x = self.drop(self.tok_emb(idx))
        out, _ = self.lstm(x)  # fresh zero state for every window
        return self.head(self.drop(out))


# --- Model 3: Small transformer ----------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position t can only see positions <= t.

    Written out by hand rather than with F.scaled_dot_product_attention so the
    attention weights can be kept for the heatmap in the notebook.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must divide evenly into heads"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))
        self.keep_weights = False
        self.last_weights = None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                   for t in (q, k, v))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        if self.keep_weights:
            self.last_weights = weights.detach()

        out = self.attn_drop(weights) @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # re-join heads
        return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise MLP: expand 4x, GELU, project back."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln(x)), then x + ffwd(ln(x))."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only transformer.

    Required components:
        token embeddings      -> self.tok_emb
        positional embeddings -> self.pos_emb
        self-attention        -> Block.attn (CausalSelfAttention)
        feedforward layer     -> Block.ffwd (FeedForward)
        output layer          -> self.ln_f + self.head
    """

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        T = idx.shape[1]
        assert T <= self.block_size, f"sequence length {T} > block_size {self.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
