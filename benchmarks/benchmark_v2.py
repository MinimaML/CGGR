#!/usr/bin/env python3
"""
CGGR Benchmark Suite v2
=======================
Professional benchmarking with CLI, modular tests, and rich outputs.

Usage:
    python benchmark_v2.py --model HuggingFaceTB/SmolLM-135M --all
    python benchmark_v2.py --model ... --throughput --memory
    python benchmark_v2.py --model ... --batch-sizes 4,8,16 --output results/
"""

import argparse
import json
import time
import gc
import os
import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich import box

console = Console()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    model_name: str = "HuggingFaceTB/SmolLM-135M"
    batch_sizes: List[int] = field(default_factory=lambda: [4, 8, 16])
    seq_len: int = 512
    num_warmup: int = 5
    num_runs: int = 50
    num_convergence_steps: int = 100
    cggr_ratio: float = 0.25
    output_dir: str = "benchmark_results"
    formats: List[str] = field(default_factory=lambda: ["json", "md"])


@dataclass
class ThroughputResult:
    """Results from throughput benchmark."""
    name: str
    batch_size: int
    seq_len: int
    tps: float  # Tokens per second
    steps_per_sec: float
    latency_ms: float
    latency_p50: float
    latency_p95: float
    latency_p99: float


@dataclass
class MemoryResult:
    """Results from memory benchmark."""
    name: str
    batch_size: int
    peak_vram_mb: float
    allocated_mb: float
    reserved_mb: float
    max_batch_size: int = 0


@dataclass
class QualityResult:
    """Results from quality benchmark."""
    name: str
    final_loss: float
    ppl: float
    loss_history: List[float] = field(default_factory=list)
    hard_token_loss: float = 0.0


@dataclass
class Tier1Result:
    """Results from Tier 1 optimization benchmarks."""
    checkpointing_vram_savings_pct: float = 0.0
    async_latency_hidden_ms: float = 0.0
    dataloader_skip_rate: float = 0.0


@dataclass
class FullBenchmarkResult:
    """Complete benchmark results."""
    model_name: str
    gpu_name: str
    gpu_vram_gb: float
    timestamp: str
    config: Dict[str, Any]
    throughput: List[ThroughputResult] = field(default_factory=list)
    memory: List[MemoryResult] = field(default_factory=list)
    quality: List[QualityResult] = field(default_factory=list)
    tier1: Optional[Tier1Result] = None


# =============================================================================
# UTILITIES
# =============================================================================

def reset_gpu():
    """Reset GPU state for clean measurements."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def get_gpu_info():
    """Get GPU name and VRAM."""
    if torch.cuda.is_available():
        return (
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / (1024**3)
        )
    return "CPU", 0.0


def percentile(data: List[float], p: float) -> float:
    """Compute percentile."""
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


# =============================================================================
# BENCHMARKER CLASS
# =============================================================================

class Benchmarker:
    """Main benchmarking class."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.cggr_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        console.print(f"[yellow]Loading {self.config.model_name}...[/yellow]")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,  # Stability
        ).to(self.device)
        
        # Create CGGR version
        from cggr import CGGRModel, create_truncated_router
        router = create_truncated_router(self.model, num_layers=4)
        self.cggr_model = CGGRModel(
            self.model,
            router=router,
            min_tokens_ratio=self.config.cggr_ratio,
            warmup_steps=0,
            selection='fixed_quota',
        )
        
        console.print(f"[green]✓ Model loaded: {sum(p.numel() for p in self.model.parameters()):,} params[/green]")
    
    def generate_data(self, batch_size: int) -> tuple:
        """Generate random input data."""
        vocab_size = self.tokenizer.vocab_size
        input_ids = torch.randint(1, vocab_size - 1, (batch_size, self.config.seq_len)).to(self.device)
        return input_ids, input_ids.clone()
    
    def benchmark_throughput(self) -> List[ThroughputResult]:
        """Run throughput benchmarks."""
        console.print("\n[bold magenta]THROUGHPUT BENCHMARK[/bold magenta]")
        results = []
        
        for bs in self.config.batch_sizes:
            input_ids, labels = self.generate_data(bs)
            
            # Standard
            result = self._measure_throughput("Standard", self.model, input_ids, labels, bs)
            results.append(result)
            
            # CGGR
            result = self._measure_throughput("CGGR", self.cggr_model, input_ids, labels, bs)
            results.append(result)
            
            reset_gpu()
        
        return results
    
    def _measure_throughput(self, name: str, model, input_ids, labels, bs) -> ThroughputResult:
        """Measure throughput for a single configuration."""
        model.train()
        
        # Warmup
        for _ in range(self.config.num_warmup):
            model.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(input_ids, labels=labels)
                loss = out if isinstance(out, torch.Tensor) else out.loss
            loss.backward()
        
        torch.cuda.synchronize()
        reset_gpu()
        
        # Measure
        latencies = []
        start = time.perf_counter()
        
        for _ in range(self.config.num_runs):
            model.zero_grad()
            iter_start = time.perf_counter()
            
            with torch.amp.autocast('cuda'):
                out = model(input_ids, labels=labels)
                loss = out if isinstance(out, torch.Tensor) else out.loss
            loss.backward()
            
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - iter_start) * 1000)
        
        total_time = time.perf_counter() - start
        total_tokens = self.config.num_runs * input_ids.numel()
        
        return ThroughputResult(
            name=f"{name} (bs={bs})",
            batch_size=bs,
            seq_len=self.config.seq_len,
            tps=total_tokens / total_time,
            steps_per_sec=self.config.num_runs / total_time,
            latency_ms=sum(latencies) / len(latencies),
            latency_p50=percentile(latencies, 50),
            latency_p95=percentile(latencies, 95),
            latency_p99=percentile(latencies, 99),
        )
    
    def benchmark_memory(self) -> List[MemoryResult]:
        """Run memory benchmarks."""
        console.print("\n[bold magenta]MEMORY BENCHMARK[/bold magenta]")
        results = []
        
        for bs in self.config.batch_sizes:
            input_ids, labels = self.generate_data(bs)
            
            # Standard
            result = self._measure_memory("Standard", self.model, input_ids, labels, bs)
            results.append(result)
            reset_gpu()
            
            # CGGR
            result = self._measure_memory("CGGR", self.cggr_model, input_ids, labels, bs)
            results.append(result)
            reset_gpu()
        
        # Find max batch sizes
        std_max = self._find_max_batch_size(self.model)
        cggr_max = self._find_max_batch_size(self.cggr_model)
        
        results.append(MemoryResult("Standard (Max BS)", 0, 0, 0, 0, std_max))
        results.append(MemoryResult("CGGR (Max BS)", 0, 0, 0, 0, cggr_max))
        
        return results
    
    def _measure_memory(self, name: str, model, input_ids, labels, bs) -> MemoryResult:
        """Measure memory for a single configuration."""
        reset_gpu()
        model.train()
        
        with torch.amp.autocast('cuda'):
            out = model(input_ids, labels=labels)
            loss = out if isinstance(out, torch.Tensor) else out.loss
        loss.backward()
        
        torch.cuda.synchronize()
        
        return MemoryResult(
            name=f"{name} (bs={bs})",
            batch_size=bs,
            peak_vram_mb=torch.cuda.max_memory_allocated() / 1024 / 1024,
            allocated_mb=torch.cuda.memory_allocated() / 1024 / 1024,
            reserved_mb=torch.cuda.memory_reserved() / 1024 / 1024,
        )
    
    def _find_max_batch_size(self, model, start=4, max_bs=128) -> int:
        """Binary search for max batch size."""
        low, high = start, max_bs
        best = start
        
        while low <= high:
            mid = (low + high) // 2
            try:
                reset_gpu()
                input_ids, labels = self.generate_data(mid)
                
                with torch.amp.autocast('cuda'):
                    out = model(input_ids, labels=labels)
                    loss = out if isinstance(out, torch.Tensor) else out.loss
                loss.backward()
                
                torch.cuda.synchronize()
                best = mid
                low = mid + 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    high = mid - 1
                    reset_gpu()
                else:
                    raise
        
        return best
    
    def benchmark_quality(self) -> List[QualityResult]:
        """Run quality benchmarks (convergence)."""
        console.print("\n[bold magenta]QUALITY BENCHMARK (Convergence)[/bold magenta]")
        results = []
        
        bs = self.config.batch_sizes[0]  # Use smallest batch
        
        # Standard convergence
        result = self._measure_convergence("Standard", self._fresh_model(), bs)
        results.append(result)
        
        # CGGR convergence
        result = self._measure_convergence("CGGR", self._fresh_cggr_model(), bs)
        results.append(result)
        
        return results
    
    def _fresh_model(self):
        """Get a fresh model instance."""
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_name, torch_dtype=torch.float32
        ).to(self.device)
    
    def _fresh_cggr_model(self):
        """Get a fresh CGGR model instance."""
        from cggr import CGGRModel, create_truncated_router
        model = self._fresh_model()
        router = create_truncated_router(model, num_layers=4)
        return CGGRModel(model, router=router, min_tokens_ratio=self.config.cggr_ratio, warmup_steps=0)
    
    def _measure_convergence(self, name: str, model, bs: int) -> QualityResult:
        """Measure convergence over N steps."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        loss_history = []
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=console) as progress:
            task = progress.add_task(f"Training {name}...", total=self.config.num_convergence_steps)
            
            for step in range(self.config.num_convergence_steps):
                input_ids, labels = self.generate_data(bs)
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    out = model(input_ids, labels=labels)
                    loss = out if isinstance(out, torch.Tensor) else out.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                if hasattr(model, 'step'):
                    model.step()
                
                loss_history.append(loss.item())
                progress.update(task, description=f"Training {name} [loss: {loss.item():.4f}]")
                progress.advance(task)
        
        del model
        reset_gpu()
        
        return QualityResult(
            name=name,
            final_loss=loss_history[-1],
            ppl=torch.exp(torch.tensor(loss_history[-1])).item(),
            loss_history=loss_history,
        )
    
    def benchmark_tier1(self) -> Tier1Result:
        """Benchmark Tier 1 optimizations."""
        console.print("\n[bold magenta]TIER 1 OPTIMIZATIONS BENCHMARK[/bold magenta]")
        
        bs = self.config.batch_sizes[0]
        input_ids, labels = self.generate_data(bs)
        
        # 1. Checkpointing VRAM savings (compare SAME model with/without checkpointing)
        try:
            from cggr_checkpointing import CGGRCheckpointedModel
            
            # Standard model WITHOUT checkpointing
            reset_gpu()
            base1 = self._fresh_model()
            with torch.amp.autocast('cuda'):
                outputs = base1(input_ids, labels=labels)
                loss = outputs.loss
            loss.backward()
            vram_no_ckpt = torch.cuda.max_memory_allocated() / 1024**2
            del base1
            
            # Standard model WITH gradient checkpointing enabled
            reset_gpu()
            base2 = self._fresh_model()
            if hasattr(base2, 'gradient_checkpointing_enable'):
                base2.gradient_checkpointing_enable()
            with torch.amp.autocast('cuda'):
                outputs = base2(input_ids, labels=labels)
                loss = outputs.loss
            loss.backward()
            vram_ckpt = torch.cuda.max_memory_allocated() / 1024**2
            del base2
            
            ckpt_savings = (vram_no_ckpt - vram_ckpt) / vram_no_ckpt * 100
            console.print(f"[dim]Checkpointing: {vram_no_ckpt:.0f}MB -> {vram_ckpt:.0f}MB ({ckpt_savings:.1f}% savings)[/dim]")
        except Exception as e:
            console.print(f"[yellow]Checkpointing test skipped: {e}[/yellow]")
            ckpt_savings = 0.0
        
        # 2. Async latency (simplified measurement)
        async_latency = 0.0  # Would need multi-batch test
        
        # 3. DataLoader skip rate
        try:
            from cggr_dataloader import NanoRouter, DifficultyFilteredDataLoader
            from torch.utils.data import DataLoader, TensorDataset
            
            data = TensorDataset(torch.randint(0, 1000, (100, 64)), torch.randint(0, 1000, (100, 64)))
            loader = DataLoader(data, batch_size=10)
            nano = NanoRouter(vocab_size=1000, hidden_dim=32)
            filtered = DifficultyFilteredDataLoader(loader, nano, threshold=0.4, input_key=0)
            
            count = sum(1 for _ in filtered)
            skip_rate = filtered.get_stats()['skip_rate']
        except Exception as e:
            console.print(f"[yellow]DataLoader test skipped: {e}[/yellow]")
            skip_rate = 0.0
        
        reset_gpu()
        
        return Tier1Result(
            checkpointing_vram_savings_pct=ckpt_savings,
            async_latency_hidden_ms=async_latency,
            dataloader_skip_rate=skip_rate,
        )


# =============================================================================
# RESULT EXPORTER
# =============================================================================

class ResultExporter:
    """Export benchmark results."""
    
    @staticmethod
    def to_json(result: FullBenchmarkResult, output_dir: str):
        """Export to JSON."""
        path = Path(output_dir) / "benchmark_results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclasses to dicts
        data = {
            "model_name": result.model_name,
            "gpu_name": result.gpu_name,
            "gpu_vram_gb": result.gpu_vram_gb,
            "timestamp": result.timestamp,
            "config": result.config,
            "throughput": [asdict(r) for r in result.throughput],
            "memory": [asdict(r) for r in result.memory],
            "quality": [asdict(r) for r in result.quality],
            "tier1": asdict(result.tier1) if result.tier1 else None,
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        console.print(f"[green]✓ JSON saved to {path}[/green]")
    
    @staticmethod
    def to_markdown(result: FullBenchmarkResult, output_dir: str):
        """Export to Markdown report."""
        path = Path(output_dir) / "benchmark_report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate speedups
        std_tps = next((r.tps for r in result.throughput if "Standard" in r.name), 1)
        cggr_tps = next((r.tps for r in result.throughput if "CGGR" in r.name), 1)
        speedup = cggr_tps / std_tps if std_tps > 0 else 1.0
        
        std_max = next((r.max_batch_size for r in result.memory if "Standard" in r.name and r.max_batch_size > 0), 0)
        cggr_max = next((r.max_batch_size for r in result.memory if "CGGR" in r.name and r.max_batch_size > 0), 0)
        
        md = f"""# CGGR Benchmark Report

**Model:** {result.model_name}
**GPU:** {result.gpu_name} ({result.gpu_vram_gb:.1f} GB)  
**Timestamp:** {result.timestamp}

---

## Executive Summary

| Metric | Standard | CGGR | Improvement |
|:--|:--|:--|:--|
| **Throughput (TPS)** | {std_tps:.0f} | {cggr_tps:.0f} | **{speedup:.2f}x** |
| **Max Batch Size** | {std_max} | {cggr_max} | **{cggr_max/max(std_max,1):.1f}x** |

---

## Throughput

| Configuration | TPS | Latency (ms) | P95 (ms) | P99 (ms) |
|:--|--:|--:|--:|--:|
"""
        for r in result.throughput:
            md += f"| {r.name} | {r.tps:.0f} | {r.latency_ms:.1f} | {r.latency_p95:.1f} | {r.latency_p99:.1f} |\n"
        
        md += "\n---\n\n## Memory\n\n| Configuration | Peak VRAM (MB) | Allocated (MB) |\n|:--|--:|--:|\n"
        for r in result.memory:
            if r.peak_vram_mb > 0:
                md += f"| {r.name} | {r.peak_vram_mb:.0f} | {r.allocated_mb:.0f} |\n"
        
        md += "\n---\n\n## Quality (Convergence)\n\n| Configuration | Final Loss | PPL |\n|:--|--:|--:|\n"
        for r in result.quality:
            md += f"| {r.name} | {r.final_loss:.4f} | {r.ppl:.2f} |\n"
        
        if result.tier1:
            md += f"""
---

## Tier 1 Optimizations

| Optimization | Result |
|:--|:--|
| Checkpointing VRAM Savings | **{result.tier1.checkpointing_vram_savings_pct:.1f}%** |
| DataLoader Skip Rate | **{result.tier1.dataloader_skip_rate:.1%}** |
"""
        
        with open(path, "w") as f:
            f.write(md)
        
        console.print(f"[green]✓ Markdown saved to {path}[/green]")
    
    @staticmethod
    def to_csv(result: FullBenchmarkResult, output_dir: str):
        """Export to CSV."""
        path = Path(output_dir) / "benchmark_results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = ["metric,configuration,value"]
        
        for r in result.throughput:
            lines.append(f"tps,{r.name},{r.tps:.2f}")
            lines.append(f"latency_ms,{r.name},{r.latency_ms:.2f}")
        
        for r in result.memory:
            if r.peak_vram_mb > 0:
                lines.append(f"peak_vram_mb,{r.name},{r.peak_vram_mb:.2f}")
        
        for r in result.quality:
            lines.append(f"final_loss,{r.name},{r.final_loss:.4f}")
        
        with open(path, "w") as f:
            f.write("\n".join(lines))
        
        console.print(f"[green]✓ CSV saved to {path}[/green]")


# =============================================================================
# DISPLAY
# =============================================================================

def display_results(result: FullBenchmarkResult):
    """Display results with rich tables."""
    console.print(Panel("[bold cyan]BENCHMARK RESULTS[/bold cyan]", expand=False))
    
    # Throughput table
    if result.throughput:
        table = Table(title="🚀 Throughput", box=box.ROUNDED)
        table.add_column("Configuration", style="cyan")
        table.add_column("TPS", justify="right")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("P95 (ms)", justify="right")
        
        baseline_tps = result.throughput[0].tps if result.throughput else 1
        for r in result.throughput:
            speedup = r.tps / baseline_tps
            table.add_row(
                r.name,
                f"{r.tps:.0f} ({speedup:.2f}x)",
                f"{r.latency_ms:.1f}",
                f"{r.latency_p95:.1f}",
            )
        console.print(table)
    
    # Memory table
    if result.memory:
        table = Table(title="💾 Memory", box=box.ROUNDED)
        table.add_column("Configuration", style="cyan")
        table.add_column("Peak VRAM (MB)", justify="right")
        table.add_column("Max Batch Size", justify="right", style="green")
        
        for r in result.memory:
            table.add_row(
                r.name,
                f"{r.peak_vram_mb:.0f}" if r.peak_vram_mb > 0 else "-",
                f"{r.max_batch_size}" if r.max_batch_size > 0 else "-",
            )
        console.print(table)
    
    # Quality table
    if result.quality:
        table = Table(title="📊 Quality", box=box.ROUNDED)
        table.add_column("Configuration", style="cyan")
        table.add_column("Final Loss", justify="right")
        table.add_column("PPL", justify="right")
        
        for r in result.quality:
            table.add_row(r.name, f"{r.final_loss:.4f}", f"{r.ppl:.2f}")
        console.print(table)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CGGR Benchmark Suite v2")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M", help="Model to benchmark")
    parser.add_argument("--batch-sizes", default="4,8,16", help="Comma-separated batch sizes")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--num-runs", type=int, default=50, help="Number of runs per test")
    parser.add_argument("--num-steps", type=int, default=100, help="Convergence steps")
    parser.add_argument("--cggr-ratio", type=float, default=0.25, help="CGGR token ratio")
    parser.add_argument("--output", default="benchmark_results", help="Output directory")
    parser.add_argument("--format", default="json,md", help="Output formats (json,md,csv)")
    
    # Test selection
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--throughput", action="store_true", help="Run throughput benchmark")
    parser.add_argument("--memory", action="store_true", help="Run memory benchmark")
    parser.add_argument("--quality", action="store_true", help="Run quality benchmark")
    parser.add_argument("--tier1", action="store_true", help="Run Tier 1 optimization benchmark")
    
    args = parser.parse_args()
    
    # Parse config
    config = BenchmarkConfig(
        model_name=args.model,
        batch_sizes=[int(x) for x in args.batch_sizes.split(",")],
        seq_len=args.seq_len,
        num_runs=args.num_runs,
        num_convergence_steps=args.num_steps,
        cggr_ratio=args.cggr_ratio,
        output_dir=args.output,
        formats=args.format.split(","),
    )
    
    # Default to all if no specific test selected
    run_all = args.all or not any([args.throughput, args.memory, args.quality, args.tier1])
    
    console.print(Panel("[bold cyan]CGGR BENCHMARK SUITE v2[/bold cyan]", expand=False))
    console.print(f"Model: {config.model_name}")
    console.print(f"Batch sizes: {config.batch_sizes}")
    console.print(f"Sequence length: {config.seq_len}")
    console.print()
    
    # Initialize
    benchmarker = Benchmarker(config)
    benchmarker.load_model()
    
    gpu_name, gpu_vram = get_gpu_info()
    
    result = FullBenchmarkResult(
        model_name=config.model_name,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram,
        timestamp=datetime.datetime.now().isoformat(),
        config=asdict(config),
    )
    
    # Run selected benchmarks
    if run_all or args.throughput:
        result.throughput = benchmarker.benchmark_throughput()
    
    if run_all or args.memory:
        result.memory = benchmarker.benchmark_memory()
    
    if run_all or args.quality:
        result.quality = benchmarker.benchmark_quality()
    
    if run_all or args.tier1:
        result.tier1 = benchmarker.benchmark_tier1()
    
    # Display results
    display_results(result)
    
    # Export
    if "json" in config.formats:
        ResultExporter.to_json(result, config.output_dir)
    if "md" in config.formats:
        ResultExporter.to_markdown(result, config.output_dir)
    if "csv" in config.formats:
        ResultExporter.to_csv(result, config.output_dir)
    
    console.print("\n[bold green]✓ Benchmark complete![/bold green]")


if __name__ == "__main__":
    main()
