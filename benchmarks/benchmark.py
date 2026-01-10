"""
CGGR Benchmark Suite
====================
Comprehensive benchmarking with SmolLM-135m.
Measures timing, memory, and token routing behavior.
"""

import torch
import torch.nn.functional as F
import time
import gc
from dataclasses import dataclass
from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich import box

console = Console()


@dataclass
class BenchmarkResult:
    name: str
    forward_ms: float
    loss_ms: float
    backward_ms: float
    total_ms: float
    memory_mb: float
    tokens_total: int
    tokens_selected: int
    avg_confidence: float
    avg_entropy: float
    avg_difficulty: float
    difficulty_histogram: Dict[str, int]


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def compute_difficulty_histogram(difficulty: torch.Tensor, num_bins: int = 5) -> Dict[str, int]:
    """Bin tokens by difficulty level."""
    flat = difficulty.view(-1)
    min_val, max_val = flat.min().item(), flat.max().item()
    
    if max_val == min_val:
        return {"all": len(flat)}
    
    bins = torch.linspace(min_val, max_val, num_bins + 1, device=flat.device)
    histogram = {}
    labels = ["very_easy", "easy", "medium", "hard", "very_hard"]
    
    for i in range(num_bins):
        mask = (flat >= bins[i]) & (flat < bins[i + 1] if i < num_bins - 1 else flat <= bins[i + 1])
        histogram[labels[i]] = mask.sum().item()
    
    return histogram


def benchmark_standard_loss(model, input_ids, labels, num_warmup=5, num_runs=20):
    """Benchmark standard CrossEntropyLoss."""
    vocab_size = model.config.vocab_size
    
    # Warmup
    for _ in range(num_warmup):
        with torch.amp.autocast('cuda'):
            outputs = model(input_ids)
            logits = outputs.logits[:, :-1, :].contiguous()
            targets = labels[:, 1:].contiguous()
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        model.zero_grad()
    
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    
    # Benchmark
    forward_times = []
    loss_times = []
    backward_times = []
    
    for _ in range(num_runs):
        model.zero_grad()
        torch.cuda.synchronize()
        
        # Forward
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda'):
            outputs = model(input_ids)
            logits = outputs.logits[:, :-1, :].contiguous()
            targets = labels[:, 1:].contiguous()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        # Loss
        with torch.amp.autocast('cuda'):
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        # Backward
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        
        forward_times.append((t1 - t0) * 1000)
        loss_times.append((t2 - t1) * 1000)
        backward_times.append((t3 - t2) * 1000)
    
    memory_mb = get_memory_mb()
    num_tokens = targets.numel()
    
    return BenchmarkResult(
        name="Standard CE",
        forward_ms=sum(forward_times) / len(forward_times),
        loss_ms=sum(loss_times) / len(loss_times),
        backward_ms=sum(backward_times) / len(backward_times),
        total_ms=sum(forward_times) / len(forward_times) + 
                 sum(loss_times) / len(loss_times) + 
                 sum(backward_times) / len(backward_times),
        memory_mb=memory_mb,
        tokens_total=num_tokens,
        tokens_selected=num_tokens,
        avg_confidence=0.0,
        avg_entropy=0.0,
        avg_difficulty=0.0,
        difficulty_histogram={},
    )


def benchmark_cggr_loss(
    model, 
    input_ids, 
    labels, 
    criterion,
    num_warmup=5, 
    num_runs=20
):
    """Benchmark CGGRLoss with detailed token tracking."""
    from triton_kernels import fused_difficulty_score
    
    vocab_size = model.config.vocab_size
    
    # Warmup
    for _ in range(num_warmup):
        with torch.amp.autocast('cuda'):
            outputs = model(input_ids)
            logits = outputs.logits[:, :-1, :].contiguous()
            targets = labels[:, 1:].contiguous()
            loss = criterion(logits, targets)
        loss.backward()
        model.zero_grad()
        criterion.step()
    
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reset for fair comparison
    criterion.step_count.fill_(criterion.warmup_steps)  # Skip warmup
    
    # Benchmark
    forward_times = []
    loss_times = []
    backward_times = []
    all_metrics = []
    all_histograms = []
    
    for _ in range(num_runs):
        model.zero_grad()
        torch.cuda.synchronize()
        
        # Forward
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda'):
            outputs = model(input_ids)
            logits = outputs.logits[:, :-1, :].contiguous()
            targets = labels[:, 1:].contiguous()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        # Loss (includes CGGR overhead)
        with torch.amp.autocast('cuda'):
            loss = criterion(logits, targets)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        # Compute difficulty histogram for this batch
        with torch.no_grad():
            difficulty, confidence, entropy = fused_difficulty_score(
                logits.float(), targets
            )
            histogram = compute_difficulty_histogram(difficulty)
            all_histograms.append(histogram)
        
        # Backward
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        
        forward_times.append((t1 - t0) * 1000)
        loss_times.append((t2 - t1) * 1000)
        backward_times.append((t3 - t2) * 1000)
        all_metrics.append(criterion.get_metrics())
    
    memory_mb = get_memory_mb()
    
    # Aggregate metrics
    avg_metrics = {
        k: sum(m.get(k, 0) for m in all_metrics) / len(all_metrics)
        for k in ['avg_confidence', 'avg_entropy', 'avg_difficulty', 'tokens_selected', 'tokens_total']
    }
    
    # Aggregate histogram
    agg_histogram = {}
    for h in all_histograms:
        for k, v in h.items():
            agg_histogram[k] = agg_histogram.get(k, 0) + v
    
    return BenchmarkResult(
        name=f"CGGR ({criterion.scoring}/{criterion.selection})",
        forward_ms=sum(forward_times) / len(forward_times),
        loss_ms=sum(loss_times) / len(loss_times),
        backward_ms=sum(backward_times) / len(backward_times),
        total_ms=sum(forward_times) / len(forward_times) + 
                 sum(loss_times) / len(loss_times) + 
                 sum(backward_times) / len(backward_times),
        memory_mb=memory_mb,
        tokens_total=int(avg_metrics['tokens_total']),
        tokens_selected=int(avg_metrics['tokens_selected']),
        avg_confidence=avg_metrics['avg_confidence'],
        avg_entropy=avg_metrics['avg_entropy'],
        avg_difficulty=avg_metrics['avg_difficulty'],
        difficulty_histogram=agg_histogram,
    )


def print_results(results: List[BenchmarkResult], baseline: BenchmarkResult):
    """Print benchmark results with rich formatting."""
    
    # Timing table
    timing_table = Table(title="⏱️ Timing Results", box=box.ROUNDED)
    timing_table.add_column("Configuration", style="cyan")
    timing_table.add_column("Forward (ms)", justify="right")
    timing_table.add_column("Loss (ms)", justify="right")
    timing_table.add_column("Backward (ms)", justify="right")
    timing_table.add_column("Total (ms)", justify="right")
    timing_table.add_column("Speedup", justify="right", style="green")
    
    for r in results:
        speedup = baseline.backward_ms / r.backward_ms if r.backward_ms > 0 else 0
        speedup_str = f"{speedup:.2f}x" if r != baseline else "1.0x (baseline)"
        timing_table.add_row(
            r.name,
            f"{r.forward_ms:.2f}",
            f"{r.loss_ms:.2f}",
            f"{r.backward_ms:.2f}",
            f"{r.total_ms:.2f}",
            speedup_str,
        )
    
    console.print(timing_table)
    console.print()
    
    # Token routing table
    routing_table = Table(title="🎯 Token Routing", box=box.ROUNDED)
    routing_table.add_column("Configuration", style="cyan")
    routing_table.add_column("Tokens Selected", justify="right")
    routing_table.add_column("Tokens Total", justify="right")
    routing_table.add_column("Selection %", justify="right")
    routing_table.add_column("Avg Confidence", justify="right")
    routing_table.add_column("Avg Entropy", justify="right")
    
    for r in results:
        pct = (r.tokens_selected / r.tokens_total * 100) if r.tokens_total > 0 else 100
        routing_table.add_row(
            r.name,
            str(r.tokens_selected),
            str(r.tokens_total),
            f"{pct:.1f}%",
            f"{r.avg_confidence:.3f}" if r.avg_confidence else "-",
            f"{r.avg_entropy:.3f}" if r.avg_entropy else "-",
        )
    
    console.print(routing_table)
    console.print()
    
    # Difficulty distribution for CGGR results
    for r in results:
        if r.difficulty_histogram:
            dist_table = Table(title=f"📊 Difficulty Distribution: {r.name}", box=box.ROUNDED)
            dist_table.add_column("Difficulty Level", style="cyan")
            dist_table.add_column("Token Count", justify="right")
            dist_table.add_column("Percentage", justify="right")
            
            total = sum(r.difficulty_histogram.values())
            for level, count in r.difficulty_histogram.items():
                pct = count / total * 100 if total > 0 else 0
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                dist_table.add_row(level, str(count), f"{bar} {pct:.1f}%")
            
            console.print(dist_table)
            console.print()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from cggr import CGGRLoss
    
    console.print("[bold cyan]CGGR Benchmark Suite[/bold cyan]")
    console.print("=" * 50)
    console.print()
    
    # Load model
    console.print("[yellow]Loading SmolLM-135m...[/yellow]")
    model_name = "HuggingFaceTB/SmolLM-135M"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).cuda()
    model.train()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    console.print(f"[green]✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} params[/green]")
    console.print()
    
    # Load real data from FineWeb-Edu
    console.print("[yellow]Loading FineWeb-Edu (streaming)...[/yellow]")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    
    batch_size = 4
    seq_len = 512
    
    # Get real text samples
    texts = []
    for i, sample in enumerate(dataset):
        if i >= batch_size:
            break
        texts.append(sample['text'][:seq_len * 4])  # Get enough text
    
    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=seq_len,
        padding='max_length',
        return_tensors='pt',
    )
    input_ids = encodings['input_ids'].cuda()
    labels = input_ids.clone()
    
    console.print(f"[green]✓ Loaded {batch_size}x{seq_len} = {batch_size * seq_len:,} real tokens[/green]")
    console.print()
    
    # Benchmark configurations
    results = []
    
    # 1. Standard CrossEntropy baseline
    console.print("\n[bold]Benchmarking Standard CrossEntropy...[/bold]")
    baseline = benchmark_standard_loss(model, input_ids, labels)
    results.append(baseline)
    
    # 2. CGGR with different strategies
    configs = [
        {"scoring": "entropy", "selection": "topk", "min_tokens_ratio": 0.25},
        {"scoring": "combined", "selection": "topk", "min_tokens_ratio": 0.25},
        {"scoring": "combined", "selection": "stratified", "min_tokens_ratio": 0.25},
        {"scoring": "combined", "selection": "topk", "min_tokens_ratio": 0.50},
    ]
    
    for config in configs:
        console.print(f"\n[bold]Benchmarking CGGR ({config['scoring']}/{config['selection']}, {int(config['min_tokens_ratio']*100)}%)...[/bold]")
        criterion = CGGRLoss(
            warmup_steps=0,  # Skip warmup for benchmark
            **config
        )
        result = benchmark_cggr_loss(model, input_ids, labels, criterion)
        results.append(result)
    
    # 3. CGGRModel with batch splitting (the real speedup)
    console.print("\n[bold magenta]Benchmarking CGGRModel (batch splitting, 25%)...[/bold magenta]")
    from cggr import CGGRModel
    
    cggr_model = CGGRModel(
        model, 
        min_tokens_ratio=0.25,
        warmup_steps=0,
    )
    
    # Warmup
    for _ in range(5):
        loss = cggr_model(input_ids, labels)
        loss.backward()
        model.zero_grad()
        cggr_model.step()
    
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    cggr_model.scorer.step_count.fill_(1000)  # Skip warmup
    
    # Benchmark CGGRModel
    forward_times = []
    backward_times = []
    
    for _ in range(20):
        model.zero_grad()
        torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        loss = cggr_model(input_ids, labels)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        forward_times.append((t1 - t0) * 1000)
        backward_times.append((t2 - t1) * 1000)
    
    metrics = cggr_model.get_metrics()
    results.append(BenchmarkResult(
        name="CGGRModel (batch split)",
        forward_ms=sum(forward_times) / len(forward_times),
        loss_ms=0,  # Loss included in forward for CGGRModel
        backward_ms=sum(backward_times) / len(backward_times),
        total_ms=sum(forward_times) / len(forward_times) + sum(backward_times) / len(backward_times),
        memory_mb=get_memory_mb(),
        tokens_total=metrics.get('tokens_total', batch_size * seq_len),
        tokens_selected=metrics.get('tokens_selected', 0),
        avg_confidence=metrics.get('avg_confidence', 0),
        avg_entropy=metrics.get('avg_entropy', 0),
        avg_difficulty=0,
        difficulty_histogram={},
    ))
    
    # 4. TRUNCATED CGGRModel (Optimized Forward)
    console.print("\n[bold magenta]Benchmarking CGGRModel (Truncated Router, 25%)...[/bold magenta]")
    from cggr import create_truncated_router
    
    # Create lightweight router (2 layers)
    router = create_truncated_router(model, num_layers=4)  # 4 layers for better accuracy
    router.cuda()
    
    truncated_model = CGGRModel(
        model, 
        router=router,
        min_tokens_ratio=0.25,
        warmup_steps=0,
    )
    
    # Warmup
    for _ in range(5):
        loss = truncated_model(input_ids, labels)
        loss.backward()
        model.zero_grad()
        truncated_model.step()
    
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    truncated_model.scorer.step_count.fill_(1000)
    
    # Benchmark
    forward_times = []
    backward_times = []
    
    for _ in range(20):
        model.zero_grad()
        torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        loss = truncated_model(input_ids, labels)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        forward_times.append((t1 - t0) * 1000)
        backward_times.append((t2 - t1) * 1000)
    
    metrics = truncated_model.get_metrics()
    results.append(BenchmarkResult(
        name="CGGRModel (Truncated)",
        forward_ms=sum(forward_times) / len(forward_times),
        loss_ms=0,
        backward_ms=sum(backward_times) / len(backward_times),
        total_ms=sum(forward_times) / len(forward_times) + sum(backward_times) / len(backward_times),
        memory_mb=get_memory_mb(),
        tokens_total=metrics.get('tokens_total', batch_size * seq_len),
        tokens_selected=metrics.get('tokens_selected', 0),
        avg_confidence=metrics.get('avg_confidence', 0),
        avg_entropy=metrics.get('avg_entropy', 0),
        avg_difficulty=0,
        difficulty_histogram={},
    ))
    
    # Print results
    console.print("\n")
    console.print("=" * 50)
    console.print("[bold cyan]RESULTS[/bold cyan]")
    console.print("=" * 50)
    console.print()
    
    print_results(results, baseline)
    
    # Summary
    best = min(results[1:], key=lambda r: r.backward_ms)
    speedup = baseline.backward_ms / best.backward_ms
    
    console.print(f"[bold green]Best backward speedup: {speedup:.2f}x with {best.name}[/bold green]")


if __name__ == "__main__":
    main()

