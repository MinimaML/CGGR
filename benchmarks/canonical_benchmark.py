#!/usr/bin/env python3
"""
Canonical CGGR benchmark surface.

This script is intentionally narrow:
- throughput and memory only
- synthetic token inputs only
- explicit metadata capture
- includes a random-sequence baseline alongside CGGR

It does NOT make quality or convergence claims.
"""

import argparse
import datetime as dt
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cggr import CGGRModel, create_truncated_router
from canonical_common import (
    build_artifact_metadata,
    collect_platform_metadata,
    get_git_commit,
    resolve_device,
)


@dataclass
class BenchmarkConfig:
    model_name: str = "HuggingFaceTB/SmolLM-135M"
    batch_sizes: List[int] = field(default_factory=lambda: [4, 8, 16])
    seq_len: int = 512
    cggr_ratio: float = 0.25
    warmup_runs: int = 5
    timed_runs: int = 25
    seed: int = 42
    device: str = "auto"
    require_cuda: bool = False
    artifact_tag: str = "default"
    benchmark_tier: str = "canonical"
    output: str = "benchmark_results/canonical_benchmark.json"


@dataclass
class BenchmarkRow:
    name: str
    batch_size: int
    tokens_per_second: float
    steps_per_second: float
    mean_latency_ms: float
    peak_memory_mb: float


def reset_runtime_state():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def synchronize_device(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class RandomSequenceBaseline(torch.nn.Module):
    """Matches CGGRModel's sequence-level second pass, but selects sequences uniformly at random."""

    def __init__(self, model: torch.nn.Module, ratio: float):
        super().__init__()
        self.model = model
        self.ratio = ratio

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        if labels is None:
            return self.model(input_ids, **kwargs)

        batch_size, seq_len = input_ids.shape
        k = max(1, int(batch_size * self.ratio))
        selected = torch.randperm(batch_size, device=input_ids.device)[:k]

        outputs = self.model(input_ids[selected], **kwargs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
        shift_labels = labels[selected][:, 1:].contiguous().view(-1)
        return torch.nn.functional.cross_entropy(shift_logits, shift_labels)


def build_models(config: BenchmarkConfig, device: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "canonical_benchmark.py requires transformers. Install the benchmark extras or "
            "make transformers available in the active environment."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.float32,
    ).to(device)
    router = create_truncated_router(base_model, num_layers=4).to(device)

    cggr_model = CGGRModel(
        base_model,
        router=router,
        min_tokens_ratio=config.cggr_ratio,
        warmup_steps=0,
        selection="fixed_quota",
    )
    random_model = RandomSequenceBaseline(base_model, config.cggr_ratio)

    return tokenizer, base_model, random_model, cggr_model


def make_batch(vocab_size: int, batch_size: int, seq_len: int, device: str):
    input_ids = torch.randint(1, vocab_size - 1, (batch_size, seq_len), device=device)
    return input_ids, input_ids.clone()


def measure_train_step(name: str, model: torch.nn.Module, input_ids: torch.Tensor, labels: torch.Tensor, warmup_runs: int, timed_runs: int) -> BenchmarkRow:
    model.train()

    for _ in range(warmup_runs):
        model.zero_grad()
        loss = model(input_ids, labels=labels)
        loss = loss if isinstance(loss, torch.Tensor) else loss.loss
        loss.backward()
        synchronize_device(input_ids.device.type)

    reset_runtime_state()

    latencies_ms = []
    start = time.perf_counter()
    for _ in range(timed_runs):
        model.zero_grad()
        iter_start = time.perf_counter()
        loss = model(input_ids, labels=labels)
        loss = loss if isinstance(loss, torch.Tensor) else loss.loss
        loss.backward()
        synchronize_device(input_ids.device.type)
        latencies_ms.append((time.perf_counter() - iter_start) * 1000)

    total_time = time.perf_counter() - start
    tokens = timed_runs * input_ids.numel()
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    elif input_ids.device.type == "mps":
        peak_memory_mb = 0.0
    else:
        peak_memory_mb = 0.0

    return BenchmarkRow(
        name=name,
        batch_size=input_ids.shape[0],
        tokens_per_second=tokens / total_time,
        steps_per_second=timed_runs / total_time,
        mean_latency_ms=sum(latencies_ms) / len(latencies_ms),
        peak_memory_mb=peak_memory_mb,
    )


def run(config: BenchmarkConfig) -> Dict:
    torch.manual_seed(config.seed)
    device = resolve_device(config.device, require_cuda=config.require_cuda)
    tokenizer, base_model, random_model, cggr_model = build_models(config, device)

    results: List[BenchmarkRow] = []
    for batch_size in config.batch_sizes:
        input_ids, labels = make_batch(tokenizer.vocab_size, batch_size, config.seq_len, device)
        for name, model in [
            ("standard", base_model),
            ("random_sequence", random_model),
            ("cggr", cggr_model),
        ]:
            reset_runtime_state()
            results.append(
                measure_train_step(
                    name,
                    model,
                    input_ids,
                    labels,
                    warmup_runs=config.warmup_runs,
                    timed_runs=config.timed_runs,
                )
            )

    return {
        "timestamp": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "git_commit": get_git_commit(),
        "artifact": build_artifact_metadata(config.artifact_tag, config.benchmark_tier, device, config.require_cuda),
        "platform": collect_platform_metadata(device),
        "config": asdict(config),
        "results": [asdict(row) for row in results],
        "notes": [
            "Synthetic token inputs only. This benchmark is for throughput/memory, not quality.",
            "Random-sequence baseline is included to avoid comparing CGGR only against full-loss training.",
            "Use --require-cuda for research-grade CUDA runs when local smoke fallback would be misleading.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Canonical CGGR throughput/memory benchmark")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--batch-sizes", default="4,8,16")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--timed-runs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--artifact-tag", default="default")
    parser.add_argument("--benchmark-tier", default="canonical")
    parser.add_argument("--output", default="benchmark_results/canonical_benchmark.json")
    args = parser.parse_args()

    config = BenchmarkConfig(
        model_name=args.model,
        batch_sizes=[int(x) for x in args.batch_sizes.split(",") if x],
        seq_len=args.seq_len,
        cggr_ratio=args.ratio,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
        seed=args.seed,
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
    print(f"Wrote canonical benchmark results to {output_path}")


if __name__ == "__main__":
    main()
