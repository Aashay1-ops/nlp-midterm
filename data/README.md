# Dataset — Tiny Shakespeare

About 40,000 lines from Shakespeare's plays, each speech written as `SPEAKER:`
on its own line followed by the speaker's text. It is public domain.

`input.txt` is committed, so nothing needs downloading after a clone. If it is
missing, `data.py` downloads it automatically on first import, or you can fetch it by hand:

```bash
curl -o data/input.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
shasum -a 256 data/input.txt
# 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed
```

## Measured properties (`python data.py`)

| Property | Value |
| --- | --- |
| Characters | 1,115,394 |
| Vocabulary (distinct characters) | 65 |
| Train (first 90%) | 1,003,854 |
| Validation (last 10%) | 111,540 |
| Validation characters never seen in training | 0 |
| Uniform-guess loss, ln 65 | 4.1744 nats/char |
| Unigram loss on validation (training frequencies, no context) | 3.3473 nats/char |
| Character pairs never seen in training (of 4,225) | 2,845 |
| Validation positions whose pair never occurs in training | 187 |
