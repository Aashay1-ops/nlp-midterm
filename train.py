"""Shared training, evaluation and sampling for all three models.

Fairness rules enforced here:
  * one training loop for every model: same context length, batch size, step
    budget, LR schedule, optimizer and gradient clipping;
  * the training batches come from a dedicated generator re-seeded for each
    model, so every model sees exactly the same windows in the same order;
  * mid-training evals use fixed batches drawn once, and the final losses
    cover every character of each split;
  * training time is wall-clock with the GPU synchronised, and excludes the
    periodic evals.

Run everything:   python train.py
Quick CPU check:  python train.py --steps 50 --eval-interval 25
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import data
from data import decode, device, encode, get_batch, iter_windows, vocab_size
from models import BigramLM, LSTMLM, TransformerLM, bigram_count_loss, count_parameters

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CONFIG = {
    # shared by every model
    "block_size": 256,
    "batch_size": 64,
    "max_steps": 5000,
    "warmup_steps": 200,
    "min_lr_ratio": 0.1,      # cosine decays to 10% of the peak LR
    "grad_clip": 1.0,
    "eval_interval": 250,
    "eval_batches": 20,
    "dropout": 0.2,
    # transformer
    "n_embd": 192,
    "n_head": 6,
    "n_layer": 4,
    # LSTM: hidden size chosen so its parameter count matches the transformer's
    "lstm_embd": 192,
    "lstm_hidden": 358,
    "lstm_layers": 2,
    # per-model optimizer settings (the only things that differ, see README)
    "lr": {"bigram": 1e-1, "lstm": 2e-3, "transformer": 2e-3},
    "weight_decay": {"bigram": 0.0, "lstm": 0.1, "transformer": 0.1},
    # sampling, identical for every model
    "sample_prompts": ["\n", "ROMEO:\n"],
    "sample_chars": 600,
    "temperature": 0.8,
    "sample_seed": 2026,
}

MODEL_NAMES = ["bigram", "lstm", "transformer"]
DISPLAY = {"bigram": "Bigram", "lstm": "LSTM", "transformer": "Transformer"}


def build_model(name, cfg=CONFIG):
    if name == "bigram":
        model = BigramLM(vocab_size)
    elif name == "lstm":
        model = LSTMLM(vocab_size, cfg["lstm_embd"], cfg["lstm_hidden"],
                       cfg["lstm_layers"], cfg["dropout"])
    elif name == "transformer":
        model = TransformerLM(vocab_size, cfg["n_embd"], cfg["n_head"],
                              cfg["n_layer"], cfg["block_size"], cfg["dropout"])
    else:
        raise ValueError(f"unknown model {name!r}")
    return model.to(device)


def sync():
    """Wait for queued GPU work, so wall-clock timings measure real compute."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def lr_at(step, peak_lr, cfg=CONFIG):
    """Linear warmup, then cosine decay to min_lr_ratio * peak_lr."""
    if step < cfg["warmup_steps"]:
        return peak_lr * (step + 1) / cfg["warmup_steps"]
    progress = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return peak_lr * (cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * cosine)


def loss_fn(logits, y):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


_EVAL_BATCHES = {}


def fixed_eval_batches(cfg=CONFIG):
    """The same eval batches for every model and every checkpoint."""
    key = (cfg["block_size"], cfg["batch_size"], cfg["eval_batches"])
    if key not in _EVAL_BATCHES:
        g = torch.Generator().manual_seed(data.SEED + 1)
        _EVAL_BATCHES[key] = {
            split: [get_batch(split, cfg["block_size"], cfg["batch_size"], g)
                    for _ in range(cfg["eval_batches"])]
            for split in ("train", "val")
        }
    return _EVAL_BATCHES[key]


@torch.no_grad()
def estimate_loss(model, cfg=CONFIG):
    """Mean loss over the fixed eval batches. Used for the learning curves."""
    model.eval()
    out = {split: sum(loss_fn(model(x), y).item() for x, y in batches) / len(batches)
           for split, batches in fixed_eval_batches(cfg).items()}
    model.train()
    return out


@torch.no_grad()
def full_loss(model, split, cfg=CONFIG):
    """Loss over every character of a split, in non-overlapping windows."""
    model.eval()
    total, count = 0.0, 0
    for x, y in iter_windows(split, cfg["block_size"], cfg["batch_size"]):
        logits = model(x)
        total += F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1),
                                 reduction="sum").item()
        count += y.numel()
    model.train()
    return total / count


def train_model(name, model, cfg=CONFIG, log=True):
    """Train one model. Returns the per-checkpoint history and the timings."""
    peak_lr = cfg["lr"][name]
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                                  weight_decay=cfg["weight_decay"][name])
    batch_gen = torch.Generator().manual_seed(data.SEED)  # same batches for every model
    history = {"step": [], "train": [], "val": [], "elapsed_s": []}
    best = {"val": float("inf"), "step": None, "state": None}
    train_time = 0.0

    def checkpoint(step):
        losses = estimate_loss(model, cfg)
        for k in ("train", "val"):
            history[k].append(losses[k])
        history["step"].append(step)
        history["elapsed_s"].append(train_time)
        if losses["val"] < best["val"]:
            best.update(val=losses["val"], step=step,
                        state={k: v.detach().clone() for k, v in model.state_dict().items()})
        if log:
            print(f"  step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}"
                  f" | {train_time:6.1f}s")

    model.train()
    checkpoint(0)
    for step in range(cfg["max_steps"]):
        x, y = get_batch("train", cfg["block_size"], cfg["batch_size"], batch_gen)
        sync()
        t0 = time.perf_counter()

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, peak_lr, cfg)
        loss = loss_fn(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        sync()
        train_time += time.perf_counter() - t0
        if (step + 1) % cfg["eval_interval"] == 0 or step + 1 == cfg["max_steps"]:
            checkpoint(step + 1)

    return {"history": history, "train_time_s": train_time,
            "best_val_est": best["val"], "best_step": best["step"],
            "best_state": best["state"]}


@torch.no_grad()
def generate(model, prompt, max_new_chars, cfg=CONFIG):
    """Sample text, feeding each sampled character back in as context.

    Context is cropped to block_size for every model, and sampling uses its
    own seeded generator, so the only thing that differs between models'
    samples is the model.
    """
    model.eval()
    g = torch.Generator().manual_seed(cfg["sample_seed"])
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new_chars):
        logits = model(idx[:, -cfg["block_size"]:])[0, -1] / cfg["temperature"]
        probs = F.softmax(logits, dim=-1).cpu()
        nxt = torch.multinomial(probs, 1, generator=g).to(device)
        idx = torch.cat([idx, nxt.view(1, 1)], dim=1)
    model.train()
    return decode(idx[0].tolist())


def run(name, cfg=CONFIG, log=True):
    """Build, train, evaluate and sample one model. Returns (model, result)."""
    torch.manual_seed(data.SEED)  # same init RNG state for every model
    model = build_model(name, cfg)
    if log:
        print(f"\n== {DISPLAY[name]} ({count_parameters(model):,} parameters) on {device} ==")
    run_out = train_model(name, model, cfg, log)

    result = {
        "parameters": count_parameters(model),
        "train_time_s": run_out["train_time_s"],
        "final": {"train_loss": full_loss(model, "train", cfg),
                  "val_loss": full_loss(model, "val", cfg)},
        "best_step": run_out["best_step"],
        "history": run_out["history"],
    }
    # Score the best checkpoint on the full val set too, then restore the final
    # weights: samples and the headline numbers come from the final model.
    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(run_out["best_state"])
    result["best_val_loss"] = full_loss(model, "val", cfg)
    model.load_state_dict(final_state)

    result["samples"] = {p: generate(model, p, cfg["sample_chars"], cfg)
                         for p in cfg["sample_prompts"]}
    if log:
        print(f"  final train {result['final']['train_loss']:.4f} | "
              f"final val {result['final']['val_loss']:.4f} | "
              f"best val {result['best_val_loss']:.4f} @ step {result['best_step']} | "
              f"{result['train_time_s']:.1f}s")
    return model, result


def results_table(results):
    """Markdown results table. The quality column is filled in by hand."""
    lines = [
        "| Model | Parameters | Train Loss | Val Loss | Best Val (step) | Val Perplexity "
        "| Training Time (s) | Generated Text Quality | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for name, r in results.items():
        lines.append(
            f"| {DISPLAY[name]} | {r['parameters']:,} | {r['final']['train_loss']:.4f} "
            f"| {r['final']['val_loss']:.4f} | {r['best_val_loss']:.4f} ({r['best_step']}) "
            f"| {math.exp(r['final']['val_loss']):.2f} | {r['train_time_s']:.1f} "
            f"| _Low / Medium / High_ | |")
    return "\n".join(lines)


def plot_curves(results, path, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colours = {"bigram": "tab:gray", "lstm": "tab:blue", "transformer": "tab:orange"}
    for ax, zoom in zip(axes, (False, True)):
        for name, r in results.items():
            h = r["history"]
            ax.plot(h["step"], h["train"], "--", color=colours[name], alpha=0.6,
                    label=f"{DISPLAY[name]} train")
            ax.plot(h["step"], h["val"], color=colours[name], label=f"{DISPLAY[name]} val")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("loss (nats/char)")
        ax.grid(alpha=0.3)
        if zoom:
            ax.set_ylim(0.9, 2.0)
            ax.set_title("zoomed: LSTM vs transformer")
        else:
            ax.axhline(base["uniform"], color="k", lw=0.8, ls=":", label="uniform guess")
            ax.axhline(base["unigram"], color="r", lw=0.8, ls=":", label="unigram baseline")
            ax.set_title("all models")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return fig


def save_results(results, cfg=CONFIG):
    """Write metrics.json, results_table.md, samples/ and the learning curves."""
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "samples").mkdir(exist_ok=True)
    base = data.baselines()
    counts = bigram_count_loss(data.train_data, data.val_data, vocab_size)

    metrics = {
        "device": device,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "config": cfg,
        "baselines": base,
        "bigram_closed_form": counts,
        "results": {k: {kk: vv for kk, vv in r.items() if kk != "samples"}
                    for k, r in results.items()},
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    table = results_table(results)
    (RESULTS_DIR / "results_table.md").write_text(
        f"# Results\n\nGenerated by `train.py` on `{device}`"
        f"{' (' + metrics['gpu'] + ')' if metrics['gpu'] else ''}. "
        f"All losses are nats/char over every character of the split.\n\n{table}\n\n"
        f"Baselines: uniform guess {base['uniform']:.4f}, "
        f"unigram (character frequencies only) {base['unigram']:.4f}, "
        f"closed-form smoothed bigram {counts['val_loss']:.4f}.\n")

    for name, r in results.items():
        body = "\n\n".join(f"--- prompt: {p!r} ---\n{s}" for p, s in r["samples"].items())
        (RESULTS_DIR / "samples" / f"{name}.txt").write_text(body + "\n")

    plot_curves(results, RESULTS_DIR / "learning_curves.png", base)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--steps", type=int, help="override max_steps (for quick checks)")
    parser.add_argument("--eval-interval", type=int)
    args = parser.parse_args()

    cfg = dict(CONFIG)
    if args.steps:
        cfg["max_steps"] = args.steps
        cfg["warmup_steps"] = min(cfg["warmup_steps"], args.steps // 10)
    if args.eval_interval:
        cfg["eval_interval"] = args.eval_interval

    results = {name: run(name, cfg)[1] for name in args.models}
    print("\n" + save_results(results, cfg))


if __name__ == "__main__":
    main()
