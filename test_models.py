"""Checks that would otherwise fail silently. Run: python -m pytest -q"""
import torch

from data import get_batch, vocab_size
from models import BigramLM, LSTMLM, TransformerLM, count_parameters
from train import CONFIG, build_model


def small_transformer():
    torch.manual_seed(0)
    return TransformerLM(vocab_size, n_embd=32, n_head=4, n_layer=2,
                         block_size=16, dropout=0.0).eval()


def test_every_model_has_the_same_interface():
    x = torch.randint(0, vocab_size, (2, 16))
    models = [BigramLM(vocab_size), LSTMLM(vocab_size, 16, 16, 2, 0.0), small_transformer()]
    for model in models:
        assert model(x).shape == (2, 16, vocab_size)


def test_causal_mask_hides_the_future():
    """Changing a later character must not move any earlier prediction.

    A broken mask lets the model read the character it is predicting: the
    loss looks excellent and the model has learned nothing.
    """
    model = small_transformer()
    x = torch.randint(0, vocab_size, (1, 16))
    changed = x.clone()
    changed[0, 10] = (changed[0, 10] + 1) % vocab_size
    with torch.no_grad():
        before, after = model(x), model(changed)
    assert torch.equal(before[0, :10], after[0, :10])
    assert not torch.equal(before[0, 10:], after[0, 10:])


def test_attention_weights_are_zero_above_the_diagonal():
    model = small_transformer()
    attn = model.blocks[0].attn
    attn.keep_weights = True
    with torch.no_grad():
        model(torch.randint(0, vocab_size, (1, 16)))
    upper = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)
    assert attn.last_weights[0][:, upper].abs().max() == 0


def test_lstm_and_transformer_are_matched_in_size():
    lstm, tf = build_model("lstm"), build_model("transformer")
    ratio = count_parameters(lstm) / count_parameters(tf)
    assert abs(ratio - 1) < 0.01, ratio


def test_training_batches_ignore_other_randomness():
    """The batch generator must give the same windows whatever else draws random numbers."""
    def first_batches(noise):
        g = torch.Generator().manual_seed(123)
        out = []
        for _ in range(3):
            torch.rand(noise)  # stands in for dropout using the global RNG
            out.append(get_batch("train", CONFIG["block_size"], 4, g)[0].cpu())
        return out

    for a, b in zip(first_batches(1), first_batches(1000)):
        assert torch.equal(a, b)
