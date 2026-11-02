# Bigram vs LSTM vs Small Transformer — Next-Character Prediction

Applied NLP midterm · Aashay Gokhale

> ✏️ **Before submitting:** every block marked ✏️ needs your input. Sections
> marked *draft* were written with AI help and should be rewritten in your own
> words. Results marked `TODO` come from `results/` after your Colab run. Delete
> this note when done.

## What NLP task did I solve?

Character-level language modelling. Given the preceding characters, each
model outputs a probability distribution over the next character. It is trained by
minimising cross-entropy and, at generation time, sampled one character at a
time, with each sampled character fed back in as context. Losses are reported in
nats per character. Perplexity is `e^loss`, which is roughly "how many
characters the model is choosing between".

## What dataset did I use?

**Tiny Shakespeare** (1,115,394 characters, 65 distinct). The split is the first
90% for training and the last 10% for validation. It is contiguous, because
overlapping windows in a random split would leak validation text into training.
See [`data/README.md`](data/README.md).

Two baselines are computed before any model is trained, so a loss number has
something to be compared with. A model that doesn't beat the unigram baseline is
not using context at all:

| Baseline | Val loss (nats/char) |
| --- | ---: |
| Uniform guess, ln 65 | 4.1744 |
| Unigram: training-set character frequencies, no context | 3.3473 |

## What three models did I compare?

| | Context it can use | How that context reaches the prediction |
| --- | --- | --- |
| Bigram | 1 character | one row of a 65×65 table |
| LSTM | 256-character window | read one step at a time into a fixed-size memory |
| Transformer | 256-character window | every position reads earlier positions directly via attention |

## How does each model work?

### Bigram — *draft, rewrite in your own words*

A 65×65 table of logits. Row *i* is the distribution over the character that
follows character *i*. It is trained by gradient descent in the same loop as the
other models. As a check, the closed-form count-based bigram (add-1 smoothing)
reaches a val loss of 2.4819, and the trained table gets within ~0.01 of it.

- **Uses:** only the current character.
- **Ignores:** everything before it, including which word it is in, who is speaking and
  whether a line just started.
- **Why it's limited:** it can learn that `u` follows `q`, but not that `t-h-e`
  is a word, because it never sees two characters at once. 2,845 of the 4,225
  possible pairs never occur in training, so a count-based version also needs
  smoothing, or the 187 validation positions that use an unseen pair get
  infinite loss.

### LSTM — *draft, rewrite in your own words*

Embedding → 2-layer LSTM (hidden size 358) → linear head. At each step the
LSTM updates a hidden state `h_t` and a cell state `c_t` through forget,
input and output gates:

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

- **More context than the bigram:** the prediction at position *t* depends on
  everything read so far in the window, not just the last character.
- **Improvement:** the cell state is updated by addition. When the forget gate is
  close to 1, information and gradients pass through many steps without
  shrinking, which a plain RNN cannot do. That is how it learns spelling, words and
  the `NAME:` format.
- **Why it can still struggle:** everything it knows about the past must fit into
  one fixed-size vector, and older information gets overwritten. Information from
  far back must survive every intermediate step. It is also sequential, so step *t*
  cannot start until step *t−1* finishes, which limits GPU parallelism.

### Small transformer — *draft, rewrite in your own words*

4 pre-norm decoder blocks, 6 heads, embedding size 192, context 256.

| Required component | In [`models.py`](models.py) |
| --- | --- |
| Token embeddings | `TransformerLM.tok_emb` |
| Positional information (learned) | `TransformerLM.pos_emb` |
| Self-attention (causal, multi-head) | `CausalSelfAttention` |
| Feedforward layer | `FeedForward` (192 → 768 → 192, GELU) |
| Output layer | `TransformerLM.ln_f` + `head` |

Each position produces a query, a key and a value, and the attention weights are
`softmax(QKᵀ / √d_k)`. A position can read any earlier position in one step
instead of waiting for it to pass through a recurrent state. Positional
embeddings are needed because attention is otherwise order-blind ("dog" = "god").
The causal mask stops a position from seeing the character it is predicting.
This is tested in [`test_models.py`](test_models.py): changing a later character
must leave every earlier prediction bit-for-bit identical.

## How the comparison was kept fair

- **One interface and one training loop.** Every model maps `(batch, time)` →
  `(batch, time, 65)` logits and goes through the same `train_model`,
  `full_loss` and `generate` in [`train.py`](train.py).
- **Identical data order.** Training batches come from a dedicated generator,
  re-seeded for each model, so all three see exactly the same windows in the same
  order. This holds even though dropout uses the global random number generator,
  and it is tested.
- **Identical budget and schedule.** Context 256, batch 64, 5,000 steps,
  200-step warmup, cosine decay to 10%, AdamW, gradient clipping at 1.0.
- **Matched size.** The LSTM's hidden size of 358 gives it 1,854,455 parameters against the
  transformer's 1,854,017 (within 0.03%), so a win can't be explained by size.
- **Matched regularisation.** Dropout is 0.2 for both. The LSTM applies it to the
  embeddings, between layers and on the output, so it isn't regularised only
  between layers while the transformer has dropout everywhere.
- **What differs, and why.** Peak learning rate is 2e-3 for both trained
  models. The bigram uses 0.1 with no weight decay, because it is a single lookup
  table that would otherwise not converge within the budget. ✏️ *If you run an
  LR pilot, record it here.*
- **Honest numbers.** Final losses cover every character of the split, not a
  random sample. Timing is wall-clock with GPU synchronisation, and excludes
  evaluation.
- **Identical sampling.** Same prompts (`"\n"` and `"ROMEO:\n"`), seed (2026),
  temperature (0.8) and length (600 characters) for every model.

## Results

✏️ `TODO` — paste in [`results/results_table.md`](results/results_table.md)
from your Colab run and fill in the quality column.

| Model | Parameters | Train Loss | Val Loss | Best Val (step) | Training Time | Generated Text Quality | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Bigram | 4,225 | TODO | TODO | TODO | TODO | Low / Medium / High | |
| LSTM | 1,854,455 | TODO | TODO | TODO | TODO | Low / Medium / High | |
| Transformer | 1,854,017 | TODO | TODO | TODO | TODO | Low / Medium / High | |

Learning curves: [`results/learning_curves.png`](results/learning_curves.png) ·
attention maps: [`results/attention.png`](results/attention.png) ·
samples: [`results/samples/`](results/samples/) ·
everything, including the full config and per-checkpoint losses:
[`results/metrics.json`](results/metrics.json).

## Which model performed best?

✏️ Answer from your own numbers. Consider the final val loss and the best
checkpoint, and say whether they agree.

## Which model trained fastest?

✏️ Compare total seconds and ms/step from notebook §5.3.

## Which model generated the most realistic text?

✏️ Quote a line or two from each sample in `results/samples/`.

## What did attention improve?

✏️ Compare it with the bigram and, separately, with the LSTM. Then answer: was it worth the
added complexity?

## What did I learn from this project?

✏️ Your reflection: what you learned about attention, what surprised you, and what
you'd do differently.

## How did I use AI responsibly?

See [`ai_use_summary.md`](ai_use_summary.md). ✏️ Add a two-sentence summary here.

---

## Repository layout

```
.
├── README.md
├── ai_use_summary.md      how AI was used, and which parts are mine
├── nlp_midterm.ipynb      runs everything and shows results (Colab)
├── data.py                loading, vocabulary, split, batching, baselines
├── models.py              BigramLM, LSTMLM, TransformerLM
├── train.py               shared training / evaluation / sampling, writes results/
├── test_models.py         causal-mask, interface, size-matching and data-order tests
├── requirements.txt
├── data/
│   ├── input.txt          Tiny Shakespeare
│   └── README.md
└── results/               generated by the run — not edited by hand
    ├── results_table.md
    ├── metrics.json
    ├── learning_curves.png
    ├── attention.png
    └── samples/{bigram,lstm,transformer}.txt
```

## How to run

**Colab (used for the reported results):** open `nlp_midterm.ipynb` in Colab,
set Runtime → Change runtime type → T4 GPU, then Run all. The first cell clones
this repo.

**Locally:**

```bash
pip install -r requirements.txt
python -m pytest -q                            # sanity tests, a few seconds
python train.py --steps 60 --eval-interval 30  # quick end-to-end check
python train.py                                # full run, writes results/
```

The device is chosen automatically: CUDA first, then Apple MPS, then CPU.
