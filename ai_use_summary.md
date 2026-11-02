# AI-Use Summary

**Assistant:** Claude (Claude Code, Opus 5.5)
**Project:** next-character prediction on Tiny Shakespeare — bigram vs LSTM vs small transformer

> ✏️ This file must be honest and in your own voice. The log entry below records
> what actually happened in the AI session. Edit its wording if you like, but
> don't remove facts. Fill in every ✏️ section yourself.

## Division of work

| Area | What the AI did | What I did |
| --- | --- | --- |
| Project structure | Reviewed a classmate's public repo with me for the general structure of a submission | Chose to build my own implementation rather than reuse theirs |
| Choice of model 2, dataset, hardware | Laid out the options | **My decisions:** LSTM, Tiny Shakespeare, Colab T4 |
| `data.py` | Filled in the TODOs in the skeleton I had written | Wrote the skeleton and its docstrings ✏️ *confirm* |
| `models.py`, `train.py`, `test_models.py`, notebook | Drafted the code and smoke-tested it locally on a short run | ✏️ Read every function, ran it, and … (what you checked or changed) |
| Running the experiments | — | ✏️ Ran the full training in Colab |
| Model explanations in the README | Drafted | ✏️ Rewrote in my own words |
| Sample quality judgements, analysis, reflection | — | ✏️ Entirely mine |

## Log

### 2026-11-02 — Building the project

**Starting point.** I shared a classmate's public midterm repository
(Chirag-Mokashi/nlp-midterm) and asked Claude to make the same thing for me.
Claude declined to copy it, because the brief requires the code, analysis and
explanations to be my own, and offered to build an original implementation
instead. Claude read that repo; I used it only as an example of what a complete
submission looks like. No code or text was copied from it.

**My decisions.** When asked, I chose an LSTM for model 2, Tiny Shakespeare as
the dataset, and Colab's T4 GPU for training. I gave Claude the assignment
brief so the build would match the actual requirements.

**What Claude built.**

- Filled in the TODOs of my `data.py` skeleton: device selection, vocabulary,
  encode/decode, the contiguous 90/10 split and `get_batch`. It also added a
  generator argument so every model gets identical batches, a full-split
  evaluation iterator, and the uniform and unigram baselines.
- `models.py`: a bigram trained by gradient descent (plus a closed-form
  count-based bigram as a convergence check), an LSTM, and a decoder-only
  transformer with hand-written causal attention so the weights can be plotted.
- `train.py`: one shared training loop (AdamW, warmup + cosine, gradient
  clipping), evaluation on every character of each split, seeded sampling,
  GPU-synchronised timing, and output files in `results/`.
- `test_models.py`: tests that the causal mask hides the future, that the
  LSTM and transformer are within 1% in size, and that the training batch
  sequence doesn't depend on other random draws.
- The notebook, the README skeleton and draft explanations, and this file.

**Design choices Claude proposed** (✏️ *say whether you agreed, and why*):
context length 256; the LSTM hidden size of 358, chosen to match the
transformer's parameter count to within 0.03%; dropout on the LSTM's embeddings
and outputs as well as between layers, so the two models are regularised
comparably; and an equal step budget, not equal wall-clock time.

**Checks Claude ran locally** (CPU/MPS, short runs only; none of the reported
numbers come from these): all tests pass; the 60-step pipeline runs end to end;
the notebook executes top to bottom; the trained bigram reaches val loss 2.4877
against the closed-form optimum of 2.4819. The smoke-test outputs were deleted.

### ✏️ YYYY-MM-DD — Running it in Colab

What you ran, anything that broke and how you fixed it, and how long it took.

### ✏️ YYYY-MM-DD — Analysis and reflection

What you wrote yourself. If AI checked your claims against your numbers or
edited your grammar, say so, and list anything it caught.

## What I understand without assistance

✏️ Write this at the end, without notes: the task, each model, why attention
helps, and the trade-offs.
