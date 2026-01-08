"""
CGGR Refined Benchmark Suite
============================
High-precision measurements of throughput, VRAM, and convergence.
Addresses data loading bottlenecks and dtype inconsistencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich import box
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from cggr import CGGRModel, create_truncated_router

console = Console()

@dataclass
class BenchmarkResult:
    name: str
    mode: str
    tps: float
    steps_per_sec: float
    latency_ms: float
    max_memory_mb: float
    final_loss: float = 0.0

def reset_gpu():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()

def find_max_batch_size(model_fn, tokenizer, seq_len=512, start_bs=4):
    """Finds the maximum batch size before OOM."""
    bs = start_bs
    last_success = 0
    console.print(f"[dim]Searching for Max BS...")
    
    while True:
        try:
            reset_gpu()
            model = model_fn()
            # Test forward + backward
            dummy_ids = torch.randint(0, 1000, (bs, seq_len)).cuda()
            with torch.cuda.amp.autocast():
                outputs = model(dummy_ids, labels=dummy_ids)
                loss = outputs if isinstance(outputs, torch.Tensor) else outputs.loss
            loss.backward()
            
            last_success = bs
            bs *= 2
            del model
            if bs > 1024: break # Safety cap
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                break
            else:
                raise e
    
    return last_success

def benchmark_run(model, input_ids, labels, num_runs=50, mode='train'):
    model.train() if mode == 'train' else model.eval()
    
    # Warmup
    for _ in range(5):
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, labels=labels)
            loss = outputs if isinstance(outputs, torch.Tensor) else outputs.loss
        if mode == 'train':
            loss.backward()
            model.zero_grad()
            
    reset_gpu()
    torch.cuda.synchronize()
    
    latencies = []
    start_time = time.perf_counter()
    
    for _ in range(num_runs):
        iter_start = time.perf_counter()
        if mode == 'train':
            model.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, labels=labels)
                loss = outputs if isinstance(outputs, torch.Tensor) else outputs.loss
            loss.backward()
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    model(input_ids)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - iter_start) * 1000)
        
    total_time = time.perf_counter() - start_time
    total_tokens = num_runs * input_ids.numel()
    
    return BenchmarkResult(
        name="", mode=mode,
        tps=total_tokens / total_time,
        steps_per_sec=num_runs / total_time,
        latency_ms=sum(latencies) / len(latencies),
        max_memory_mb=torch.cuda.max_memory_allocated() / 1024 / 1024
    )

def benchmark_convergence(model, preloaded_data, num_steps=50, name="Model"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    
    start_time = time.perf_counter()
    reset_gpu()
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=console) as progress:
        task = progress.add_task(f"Training {name} (Convergence)...", total=num_steps)
        
        last_loss = 0.0
        for i in range(num_steps):
            batch = preloaded_data[i % len(preloaded_data)]
            input_ids, labels = batch[0].cuda(), batch[1].cuda()
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, labels=labels)
                loss = outputs if isinstance(outputs, torch.Tensor) else outputs.loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if hasattr(model, 'step'):
                model.step()
                
            last_loss = loss.item()
            if i % 5 == 0:
                progress.update(task, description=f"Training {name} [loss: {last_loss:.4f}]")
            progress.advance(task)
            
    total_time = time.perf_counter() - start_time
    total_tokens = num_steps * preloaded_data[0][0].numel()
    
    return BenchmarkResult(
        name=name, mode="convergence",
        tps=total_tokens / total_time,
        steps_per_sec=num_steps / total_time,
        latency_ms=(total_time / num_steps) * 1000,
        max_memory_mb=torch.cuda.max_memory_allocated() / 1024 / 1024,
        final_loss=last_loss
    )

def main():
    console.print(Panel("[bold cyan]CGGR REFINED BENCHMARK SUITE[/bold cyan]"))
    
    model_name = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 1. PRE-LOAD DATA (To remove IO bottlenecks)
    console.print("[yellow]Pre-loading data for accuracy...[/yellow]")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    preloaded_batches = []
    iterator = iter(dataset)
    for _ in range(50):
        texts = [next(iterator)['text'][:2048] for _ in range(4)] # bs=4
        enc = tokenizer(texts, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
        preloaded_batches.append((enc['input_ids'], enc['input_ids']))
        
    input_ids, labels = preloaded_batches[0][0].cuda(), preloaded_batches[0][1].cuda()
    
    # 2. MODELS
    def get_model():
        # Force float16 loading to avoid massive VRAM usage of HF default float32
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

    results = []

    # 2. AUTO-SCALE COMPARISON (The "Level Playing Field")
    console.print("\n[bold magenta]2. SEARCHING FOR PEAK THROUGHPUT (MATCHED VRAM)[/bold magenta]")
    
    max_bs_std = find_max_batch_size(get_model, tokenizer)
    console.print(f"[green]Standard Max Batch Size: {max_bs_std}[/green]")
    
    def get_cggr_fn():
        m = get_model()
        r = create_truncated_router(m, num_layers=4).cuda()
        return CGGRModel(m, router=r, min_tokens_ratio=0.25, warmup_steps=0)
    
    max_bs_cggr = find_max_batch_size(get_cggr_fn, tokenizer)
    console.print(f"[green]CGGR Max Batch Size: {max_bs_cggr}[/green]")

    # Run Peak Tests
    ids_std = torch.randint(0, 1000, (max_bs_std, 512)).cuda()
    model_std = get_model()
    res_peak_std = benchmark_run(model_std, ids_std, ids_std)
    res_peak_std.name = f"Standard (Peak, BS={max_bs_std})"
    results.append(res_peak_std)
    del model_std
    reset_gpu()

    ids_cggr = torch.randint(0, 1000, (max_bs_cggr, 512)).cuda()
    model_cggr = get_cggr_fn()
    res_peak_cggr = benchmark_run(model_cggr, ids_cggr, ids_cggr)
    res_peak_cggr.name = f"CGGR (Peak, BS={max_bs_cggr})"
    results.append(res_peak_cggr)
    del model_cggr
    reset_gpu()

    # 3. CONVERGENCE COMPARISON (LOSS VS TIME)
    console.print("\n[bold magenta]3. CONVERGENCE EFFICIENCY (LOSS VS TIME)[/bold magenta]")
    # Use the same mini 50-step run but emphasize the Loss/Time trade-off
    
    model = get_model()
    std_conv = benchmark_convergence(model, preloaded_batches, name="Standard")
    results.append(std_conv)
    del model
    reset_gpu()

    model = get_cggr_fn()
    cggr_conv = benchmark_convergence(model, preloaded_batches, name="CGGR (25%)")
    results.append(cggr_conv)
    del model
    reset_gpu()

    # 3. REPORT
    table = Table(title="🏆 CGGR vs Standard: Final Comparison", box=box.HORIZONTALS)
    table.add_column("Configuration", style="cyan")
    table.add_column("TPS", justify="right")
    table.add_column("Speedup", justify="right", style="green")
    table.add_column("VRAM (MB)", justify="right")
    table.add_column("VRAM Save", justify="right", style="magenta")
    table.add_column("Iter/Sec", justify="right")
    table.add_column("Loss", justify="right")

    baseline = results[0]
    for r in results:
        speedup = r.tps / baseline.tps
        vram_save = (baseline.max_memory_mb - r.max_memory_mb) / baseline.max_memory_mb * 100 if r.max_memory_mb < baseline.max_memory_mb else 0
        table.add_row(
            r.name,
            f"{r.tps:.0f}",
            f"{speedup:.2f}x",
            f"{r.max_memory_mb:.0f}",
            f"{vram_save:.0f}%" if vram_save > 0 else "-",
            f"{r.steps_per_sec:.2f}",
            f"{r.final_loss:.4f}" if r.final_loss > 0 else "-"
        )
    console.print(table)

if __name__ == "__main__":
    main()
