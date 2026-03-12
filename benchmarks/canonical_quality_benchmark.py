#!/usr/bin/env python3
"""
Canonical CGGR quality benchmark.

This script is intentionally narrow:
- dataset-backed training/evaluation
- fixed protocol and metadata capture
- held-out full-loss evaluation
- compares standard training, random sequence selection, random token selection, and CGGR

It is designed to support exploratory quality claims inside this repo,
not broad production claims.
"""

import argparse
import datetime as dt
import gc
import json
import math
import os
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cggr import CGGRModel, create_truncated_router
from canonical_common import (
    build_artifact_metadata,
    collect_platform_metadata,
    get_git_commit,
    resolve_device,
)


@dataclass
class QualityBenchmarkConfig:
    model_name: str = "HuggingFaceTB/SmolLM-135M"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    train_split: str = "train"
    eval_split: str = "validation"
    seq_len: int = 256
    batch_size: int = 4
    train_steps: int = 20
    eval_batches: int = 16
    cggr_ratio: float = 0.25
    learning_rate: float = 5e-5
    seed: int = 42
    max_train_texts: int = 2000
    max_eval_texts: int = 1000
    device: str = "auto"
    require_cuda: bool = False
    artifact_tag: str = "default"
    benchmark_tier: str = "canonical"
    output: str = "benchmark_results/canonical_quality_benchmark.json"


@dataclass
class QualityBenchmarkRow:
    name: str
    train_steps: int
    mean_train_loss: float
    final_train_loss: float
    eval_loss: float
    eval_ppl: float
    loss_history: List[float] = field(default_factory=list)


def reset_runtime_state():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def autocast_context(device: str):
    return torch.amp.autocast(device_type="cuda") if device == "cuda" else nullcontext()


class RandomSequenceBaseline(torch.nn.Module):
    """Budget-matched baseline for sequence-level selective training."""

    def __init__(self, model: torch.nn.Module, ratio: float):
        super().__init__()
        self.model = model
        self.ratio = ratio

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        if labels is None:
            return self.model(input_ids, **kwargs)

        batch_size = input_ids.shape[0]
        k = max(1, int(batch_size * self.ratio))
        selected = torch.randperm(batch_size, device=input_ids.device)[:k]

        outputs = self.model(input_ids[selected], **kwargs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
        shift_labels = labels[selected][:, 1:].contiguous().view(-1)
        return F.cross_entropy(shift_logits, shift_labels)


class RandomTokenBaseline(torch.nn.Module):
    """Budget-matched baseline that selects random next-token positions."""

    def __init__(self, model: torch.nn.Module, ratio: float):
        super().__init__()
        self.model = model
        self.ratio = ratio

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        outputs = self.model(input_ids, **kwargs)
        if labels is None:
            return outputs

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
        shift_labels = labels[:, 1:].contiguous().view(-1)

        total = shift_labels.numel()
        k = max(1, int(total * self.ratio))
        selected = torch.randperm(total, device=shift_labels.device)[:k]

        selected_logits = shift_logits[selected]
        selected_labels = shift_labels[selected]
        return F.cross_entropy(selected_logits, selected_labels)


def build_token_blocks(
    dataset: Iterable[Dict[str, str]],
    tokenizer,
    *,
    text_key: str,
    seq_len: int,
    max_texts: int,
) -> List[torch.Tensor]:
    token_buffer: List[int] = []
    blocks: List[torch.Tensor] = []
    eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or tokenizer.pad_token_id

    count = 0
    for row in dataset:
        if count >= max_texts:
            break
        text = row.get(text_key, "")
        if not text or not text.strip():
            continue
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        token_buffer.extend(token_ids)
        if eos_id is not None:
            token_buffer.append(eos_id)
        while len(token_buffer) >= seq_len:
            blocks.append(torch.tensor(token_buffer[:seq_len], dtype=torch.long))
            token_buffer = token_buffer[seq_len:]
        count += 1

    return blocks


def make_batches(blocks: List[torch.Tensor], batch_size: int) -> List[torch.Tensor]:
    usable = len(blocks) - (len(blocks) % batch_size)
    blocks = blocks[:usable]
    batches = []
    for i in range(0, usable, batch_size):
        batches.append(torch.stack(blocks[i:i + batch_size], dim=0))
    return batches


def load_dataset_batches(config: QualityBenchmarkConfig):
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "canonical_quality_benchmark.py requires datasets and transformers. Install the "
            "benchmark extras or make those packages available in the active environment."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=config.train_split,
    )
    eval_ds = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=config.eval_split,
    )

    text_key = "text"
    train_blocks = build_token_blocks(
        train_ds,
        tokenizer,
        text_key=text_key,
        seq_len=config.seq_len,
        max_texts=config.max_train_texts,
    )
    eval_blocks = build_token_blocks(
        eval_ds,
        tokenizer,
        text_key=text_key,
        seq_len=config.seq_len,
        max_texts=config.max_eval_texts,
    )

    train_batches = make_batches(train_blocks, config.batch_size)
    eval_batches = make_batches(eval_blocks, config.batch_size)

    if not train_batches or not eval_batches:
        raise RuntimeError("Dataset preprocessing produced no usable train/eval batches")

    return tokenizer, train_batches, eval_batches


def build_models(config: QualityBenchmarkConfig, device: str):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "canonical_quality_benchmark.py requires transformers. Install the benchmark extras "
            "or make transformers available in the active environment."
        ) from exc

    def fresh_base_model():
        return AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=torch.float32,
        ).to(device)

    def fresh_cggr_model():
        base = fresh_base_model()
        router = create_truncated_router(base, num_layers=4).to(device)
        return CGGRModel(
            base,
            router=router,
            min_tokens_ratio=config.cggr_ratio,
            warmup_steps=0,
            selection="fixed_quota",
        )

    def fresh_random_model():
        base = fresh_base_model()
        return RandomSequenceBaseline(base, config.cggr_ratio)

    def fresh_random_token_model():
        base = fresh_base_model()
        return RandomTokenBaseline(base, config.cggr_ratio)

    return {
        "standard": fresh_base_model,
        "random_sequence": fresh_random_model,
        "random_token": fresh_random_token_model,
        "cggr": fresh_cggr_model,
    }


def resolve_eval_model(model):
    """Evaluate wrappers with their full base model, but keep plain HF causal LMs intact."""
    if isinstance(model, (CGGRModel, RandomSequenceBaseline, RandomTokenBaseline)):
        return model.model
    return model


def compute_eval_loss(model, eval_batches: List[torch.Tensor], device: str, max_batches: int) -> float:
    model.eval()
    eval_model = resolve_eval_model(model)
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in eval_batches[:max_batches]:
            batch = batch.to(device)
            with autocast_context(device):
                outputs = eval_model(batch, labels=batch)
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss
                    shift_tokens = batch[:, 1:].numel()
                else:
                    if hasattr(outputs, "logits") and outputs.logits is not None:
                        logits = outputs.logits
                    elif isinstance(outputs, (tuple, list)) and outputs:
                        logits = outputs[0]
                    else:
                        logits = outputs
                    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
                    shift_labels = batch[:, 1:].contiguous().view(-1)
                    loss = F.cross_entropy(shift_logits, shift_labels)
                    shift_tokens = shift_labels.numel()
            total_loss += loss.item() * shift_tokens
            total_tokens += shift_tokens

    model.train()
    return total_loss / max(total_tokens, 1)


def run_training(name: str, model, train_batches: List[torch.Tensor], config: QualityBenchmarkConfig, device: str) -> QualityBenchmarkRow:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_history: List[float] = []
    train_steps = min(config.train_steps, len(train_batches))

    model.train()
    for step in range(train_steps):
        batch = train_batches[step].to(device)
        optimizer.zero_grad()
        with autocast_context(device):
            outputs = model(batch, labels=batch)
            loss = outputs if isinstance(outputs, torch.Tensor) else outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if hasattr(model, "step"):
            model.step()
        loss_history.append(loss.item())

    return QualityBenchmarkRow(
        name=name,
        train_steps=train_steps,
        mean_train_loss=sum(loss_history) / len(loss_history),
        final_train_loss=loss_history[-1],
        eval_loss=0.0,
        eval_ppl=0.0,
        loss_history=loss_history,
    )


def run(config: QualityBenchmarkConfig) -> Dict:
    torch.manual_seed(config.seed)
    device = resolve_device(config.device, require_cuda=config.require_cuda)
    _, train_batches, eval_batches = load_dataset_batches(config)
    model_factories = build_models(config, device)

    results = []
    for name, factory in model_factories.items():
        reset_runtime_state()
        model = factory()
        row = run_training(name, model, train_batches, config, device)
        row.eval_loss = compute_eval_loss(model, eval_batches, device, config.eval_batches)
        row.eval_ppl = math.exp(row.eval_loss) if row.eval_loss < 20 else float("inf")
        results.append(row)
        del model
        reset_runtime_state()

    return {
        "timestamp": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "git_commit": get_git_commit(),
        "artifact": build_artifact_metadata(config.artifact_tag, config.benchmark_tier, device, config.require_cuda),
        "platform": collect_platform_metadata(device),
        "config": asdict(config),
        "dataset_protocol": {
            "train_split": config.train_split,
            "eval_split": config.eval_split,
            "eval_metric": "full-loss held-out perplexity",
            "notes": [
                "Quality evaluation always uses full-loss evaluation on held-out text.",
                "Random-sequence baseline is included as a budget-matched control.",
                "Random-token baseline is included as a token-level budget-matched control.",
                "Use --require-cuda for research-grade CUDA runs when local smoke fallback would be misleading.",
            ],
        },
        "results": [asdict(row) for row in results],
    }


def main():
    parser = argparse.ArgumentParser(description="Canonical CGGR quality benchmark")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-texts", type=int, default=2000)
    parser.add_argument("--max-eval-texts", type=int, default=1000)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--artifact-tag", default="default")
    parser.add_argument("--benchmark-tier", default="canonical")
    parser.add_argument("--output", default="benchmark_results/canonical_quality_benchmark.json")
    args = parser.parse_args()

    config = QualityBenchmarkConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        eval_batches=args.eval_batches,
        cggr_ratio=args.ratio,
        learning_rate=args.lr,
        seed=args.seed,
        max_train_texts=args.max_train_texts,
        max_eval_texts=args.max_eval_texts,
        device=args.device,
        require_cuda=args.require_cuda,
        artifact_tag=args.artifact_tag,
        benchmark_tier=args.benchmark_tier,
        output=args.output,
    )

    payload = run(config)
    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote canonical quality benchmark results to {output_path}")


if __name__ == "__main__":
    main()
