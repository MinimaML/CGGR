#!/usr/bin/env python3
"""
Code Fine-tune Benchmark
========================
Compare Standard vs CGGR fine-tuning on high-quality code datasets.

Usage:
    python code_finetune_benchmark.py --mode cggr --hours 12 --wandb
"""

import sys
import os
# Ensure we use the local cggr module, not the installed package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import gc
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# Optional W&B
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

# Flash Attention
try:
    from cggr_flash import get_flash_attention_info, HAS_SDPA
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False
    HAS_SDPA = False


# =============================================================================
# MUON OPTIMIZER
# =============================================================================

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    
    A simplified, fast optimizer that works well for LLM training.
    Based on: https://github.com/KellerJordan/modded-nanogpt
    """
    
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, 
                 ns_steps=5, adamw_fallback_lr=3e-4, adamw_betas=(0.9, 0.95), 
                 adamw_eps=1e-8, adamw_wd=0.01):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            adamw_fallback_lr=adamw_fallback_lr, adamw_betas=adamw_betas,
            adamw_eps=adamw_eps, adamw_wd=adamw_wd
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad
                
                # Use Muon for 2D+ params, AdamW for 1D (biases, norms)
                if p.dim() >= 2:
                    # Newton-Schulz orthogonalization
                    g = d_p
                    if p.dim() > 2:
                        g = g.view(g.shape[0], -1)
                    
                    # Orthogonalize
                    g = self._newton_schulz(g, ns_steps)
                    
                    if p.dim() > 2:
                        g = g.view_as(d_p)
                    
                    # Momentum
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    
                    p.add_(g, alpha=-lr)
                else:
                    # AdamW fallback for 1D params
                    state = self.state[p]
                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    
                    state['step'] += 1
                    beta1, beta2 = group['adamw_betas']
                    
                    state['exp_avg'].mul_(beta1).add_(d_p, alpha=1 - beta1)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(d_p, d_p, value=1 - beta2)
                    
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    step_size = group['adamw_fallback_lr'] / bias_correction1
                    denom = (state['exp_avg_sq'].sqrt() / (bias_correction2 ** 0.5)).add_(group['adamw_eps'])
                    
                    p.addcdiv_(state['exp_avg'], denom, value=-step_size)
                    p.add_(p, alpha=-group['adamw_wd'] * group['adamw_fallback_lr'])
        
        return loss
    
    def _newton_schulz(self, G, steps=5):
        """Newton-Schulz iteration for orthogonalization."""
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G / (G.norm() + 1e-7)
        
        if X.shape[0] > X.shape[1]:
            X = X.T
            transpose = True
        else:
            transpose = False
        
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        
        if transpose:
            X = X.T
        
        return X


# =============================================================================
# HIGH-QUALITY CODE DATASETS
# =============================================================================

class CodeDataset(Dataset):
    """Combined high-quality code dataset."""
    
    def __init__(self, tokenizer, max_length: int = 2048, max_samples: int = 100000):
        from datasets import load_dataset
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        console.print("[yellow]Loading high-quality code datasets...[/yellow]")
        
        # 1. Code Alpaca - Instruction-following code
        try:
            console.print("[cyan]Loading sahil2801/CodeAlpaca-20k...[/cyan]")
            alpaca = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
            for item in alpaca:
                text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
                self.samples.append({
                    "code": text,
                    "source": "code-alpaca",
                    "language": "mixed"
                })
            console.print(f"[green]✓ Code Alpaca: {len(alpaca)} samples[/green]")
        except Exception as e:
            console.print(f"[yellow]Code Alpaca skipped: {e}[/yellow]")
        
        # 2. Evol-Instruct-Code - High-quality evolved code instructions
        try:
            console.print("[cyan]Loading theblackcat102/evol-codealpaca-v1...[/cyan]")
            evol = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
            for item in evol:
                text = f"### Instruction:\n{item.get('instruction', '')}\n\n### Response:\n{item.get('output', '')}"
                if len(text) > 100:
                    self.samples.append({
                        "code": text,
                        "source": "evol-code",
                        "language": "mixed"
                    })
            console.print(f"[green]✓ Evol-Instruct-Code: {len(evol)} samples[/green]")
        except Exception as e:
            console.print(f"[yellow]Evol-Instruct-Code skipped: {e}[/yellow]")
        
        # 3. CodeContests - Competitive programming
        try:
            console.print("[cyan]Loading deepmind/code_contests...[/cyan]")
            contests = load_dataset("deepmind/code_contests", split="train")
            count = 0
            for item in contests:
                if count >= 10000:
                    break
                solutions = item.get("solutions", {}).get("solution", [])
                if solutions:
                    sol = solutions[0] if isinstance(solutions, list) else solutions
                    text = f"### Problem:\n{item.get('description', '')[:2000]}\n\n### Solution:\n{sol}"
                    self.samples.append({
                        "code": text,
                        "source": "code-contests",
                        "language": "mixed"
                    })
                    count += 1
            console.print(f"[green]✓ CodeContests: {count} samples[/green]")
        except Exception as e:
            console.print(f"[yellow]CodeContests skipped: {e}[/yellow]")
        
        console.print(f"[bold green]Total code samples: {len(self.samples)}[/bold green]")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["code"]
        
        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0),
        }


# =============================================================================
# METRICS HELPERS
# =============================================================================

def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def get_gpu_memory_max_mb():
    """Get max GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0

def get_gradient_norm(model):
    """Calculate total gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_humaneval_style(model, tokenizer, device: str, num_samples: int = 50) -> Dict[str, Any]:
    """Simple code completion evaluation."""
    model.eval()
    
    test_prompts = [
        "def fibonacci(n):\n    '''Return the nth fibonacci number.'''\n",
        "def binary_search(arr, target):\n    '''Binary search for target in sorted array.'''\n",
        "def merge_sort(arr):\n    '''Sort array using merge sort.'''\n",
        "def is_palindrome(s):\n    '''Check if string is a palindrome.'''\n",
        "class Stack:\n    '''Simple stack implementation.'''\n    def __init__(self):\n",
        "def quicksort(arr):\n    '''Sort array using quicksort.'''\n",
        "def gcd(a, b):\n    '''Return greatest common divisor.'''\n",
        "def prime_factors(n):\n    '''Return list of prime factors.'''\n",
    ]
    
    valid_completions = 0
    total = min(num_samples, len(test_prompts))
    
    with torch.no_grad():
        for prompt in test_prompts[:total]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            try:
                outputs = model.generate(
                    **inputs, max_new_tokens=128, do_sample=True, temperature=0.2,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                completion = response[len(prompt):]
                if len(completion) > 20 and ("return" in completion or "self" in completion or "=" in completion):
                    valid_completions += 1
            except:
                pass
    
    model.train()
    return {"completion_rate": valid_completions / total if total > 0 else 0.0}


# =============================================================================
# TRAINING
# =============================================================================

@dataclass
class TrainingState:
    start_time: float
    total_steps: int = 0
    total_tokens: int = 0
    current_loss: float = 0.0
    eval_history: List[Dict] = None
    tokens_this_interval: int = 0
    last_interval_time: float = 0
    
    def __post_init__(self):
        if self.eval_history is None:
            self.eval_history = []
        self.last_interval_time = self.start_time


def run_training(
    mode: str,
    hours: float,
    model_name: str,
    output_dir: str,
    batch_size: int = 1,
    gradient_accumulation: int = 8,
    learning_rate: float = 0.02,  # Muon uses higher LR
    eval_interval_min: int = 60,
    checkpoint_interval_min: int = 120,
    use_wandb: bool = False,
    wandb_project: str = "cggr-code-benchmark",
    flash_attention: bool = True,
) -> Dict[str, Any]:
    """Run extended code training with Muon optimizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]CODE FINE-TUNE - {mode.upper()} MODE (MUON)[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"Model: {model_name}")
    console.print(f"Duration: {hours} hours")
    console.print(f"Optimizer: Muon (lr={learning_rate})")
    console.print()
    
    # W&B init with enhanced config
    if use_wandb and HAS_WANDB:
        wandb.init(
            project=wandb_project,
            name=f"{mode}_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M')}",
            config={
                "mode": mode, 
                "hours": hours, 
                "model": model_name,
                "batch_size": batch_size, 
                "grad_accum": gradient_accumulation, 
                "lr": learning_rate,
                "flash_attention": flash_attention,
                "optimizer": "Muon",
                "dtype": "bfloat16",
            }
        )
        console.print("[green]✓ W&B initialized[/green]")
    
    # Load model with Flash Attention
    console.print("[yellow]Loading model...[/yellow]")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine attention implementation
    attn_impl = 'eager'
    if flash_attention and HAS_FLASH:
        info = get_flash_attention_info()
        attn_impl = info['recommended_backend']
        console.print(f"[green]✓ Flash Attention: {attn_impl}[/green]")
    
    # Use BFloat16 (no grad scaler needed!)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    console.print(f"[green]✓ Using dtype: {dtype}[/green]")
    
    # Load model
    if attn_impl in ['flash_attention_2', 'sdpa']:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            trust_remote_code=True,
            device_map="auto",
            use_cache=False,  # Disable cache for gradient checkpointing
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            use_cache=False,
        )
    
    # Enable gradient checkpointing (after model load, before setting requires_grad)
    if hasattr(base_model, 'gradient_checkpointing_enable'):
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        console.print("[green]✓ Gradient checkpointing enabled[/green]")
    
    # Ensure requires_grad is set
    for param in base_model.parameters():
        param.requires_grad = True
    
    if mode == "cggr":
        from cggr import CGGRModel, create_truncated_router
        router = create_truncated_router(base_model, num_layers=4)
        model = CGGRModel(base_model, router=router, min_tokens_ratio=0.25, warmup_steps=500, selection='fixed_quota')
        console.print("[green]✓ CGGR model initialized[/green]")
    else:
        model = base_model
        console.print("[green]✓ Standard model initialized[/green]")
    
    # Load code dataset
    train_dataset = CodeDataset(tokenizer, max_length=2048, max_samples=100000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    # MUON optimizer
    optimizer = Muon(model.parameters(), lr=learning_rate, momentum=0.95, nesterov=True)
    console.print("[green]✓ Muon optimizer initialized[/green]")
    
    total_steps_estimate = int((hours * 3600) / 2.0)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps_estimate)
    
    # NO GradScaler for BFloat16!
    use_amp = dtype == torch.float16  # Only use AMP with FP16
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        console.print("[yellow]Using GradScaler for FP16[/yellow]")
    else:
        console.print("[green]✓ BFloat16 mode (no GradScaler needed)[/green]")
    
    # Training loop
    start_time = time.time()
    end_time = start_time + (hours * 3600)
    state = TrainingState(start_time=start_time)
    
    last_eval_time = start_time
    last_checkpoint_time = start_time
    eval_interval = eval_interval_min * 60
    checkpoint_interval = checkpoint_interval_min * 60
    
    console.print(f"\n[bold green]Training started. Expected end: {(datetime.now() + timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')}[/bold green]\n")
    
    epoch = 0
    while time.time() < end_time:
        epoch += 1
        model.train()
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            if time.time() >= end_time:
                break
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            batch_tokens = input_ids.numel()
            
            # Forward pass with autocast
            with torch.amp.autocast('cuda', dtype=dtype):
                if mode == "cggr":
                    loss = model(input_ids, labels=labels)
                else:
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss
                loss = loss / gradient_accumulation
            
            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation == 0:
                # Gradient clipping
                grad_norm = get_gradient_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step
                if use_amp:
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                
                if mode == "cggr" and hasattr(model, 'step'):
                    model.step()
                
                state.total_steps += 1
                state.current_loss = loss.item() * gradient_accumulation
                state.total_tokens += batch_tokens * gradient_accumulation
                state.tokens_this_interval += batch_tokens * gradient_accumulation
                
                # W&B logging with enhanced metrics
                if use_wandb and HAS_WANDB and state.total_steps % 10 == 0:
                    current_time = time.time()
                    elapsed = current_time - state.last_interval_time
                    tokens_per_sec = state.tokens_this_interval / elapsed if elapsed > 0 else 0
                    
                    log_data = {
                        "loss": state.current_loss,
                        "step": state.total_steps,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "gradient_norm": grad_norm,
                        "vram_mb": get_gpu_memory_mb(),
                        "vram_max_mb": get_gpu_memory_max_mb(),
                        "tokens_per_sec": tokens_per_sec,
                        "total_tokens": state.total_tokens,
                    }
                    
                    # CGGR-specific metrics
                    if mode == "cggr" and hasattr(model, 'get_metrics'):
                        metrics = model.get_metrics()
                        token_ratio = metrics.get("token_ratio", 1.0)
                        tokens_selected = metrics.get("tokens_selected", 0)
                        tokens_total = metrics.get("tokens_total", 1)
                        
                        log_data["cggr/token_ratio"] = token_ratio
                        log_data["cggr/token_savings_pct"] = (1.0 - token_ratio) * 100
                        log_data["cggr/tokens_selected"] = tokens_selected
                        log_data["cggr/tokens_skipped"] = tokens_total - tokens_selected
                    
                    wandb.log(log_data)
                    
                    # Reset interval counters
                    state.tokens_this_interval = 0
                    state.last_interval_time = current_time
                
                # Console logging
                if state.total_steps % 50 == 0:
                    elapsed = time.time() - start_time
                    vram = get_gpu_memory_mb()
                    cggr_info = ""
                    if mode == "cggr" and hasattr(model, 'get_metrics'):
                        m = model.get_metrics()
                        cggr_info = f", Savings: {(1-m.get('token_ratio', 1))*100:.1f}%"
                    console.print(f"Step {state.total_steps}: Loss={state.current_loss:.4f}, VRAM={vram:.0f}MB, GradNorm={grad_norm:.2f}{cggr_info}")
        
        current_time = time.time()
        elapsed = current_time - start_time
        console.print(f"Epoch {epoch}: Steps={state.total_steps}, Loss={state.current_loss:.4f}, Elapsed={str(timedelta(seconds=int(elapsed)))}")
        
        # Periodic evaluation
        if current_time - last_eval_time >= eval_interval:
            console.print("[yellow]Evaluating...[/yellow]")
            eval_model = model.model if mode == "cggr" else model
            eval_result = evaluate_humaneval_style(eval_model, tokenizer, device)
            
            state.eval_history.append({
                "elapsed_hours": elapsed / 3600, "steps": state.total_steps,
                "completion_rate": eval_result["completion_rate"], "loss": state.current_loss,
            })
            
            console.print(f"[bold green]Code Completion Rate: {eval_result['completion_rate']:.2%}[/bold green]")
            
            if use_wandb and HAS_WANDB:
                wandb.log({"eval/completion_rate": eval_result["completion_rate"]})
            
            last_eval_time = current_time
        
        # Periodic checkpoint
        if current_time - last_checkpoint_time >= checkpoint_interval:
            ckpt_path = output_path / f"checkpoint_step{state.total_steps}"
            save_model = model.model if mode == "cggr" else model
            save_model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            console.print(f"[green]✓ Checkpoint saved: {ckpt_path}[/green]")
            last_checkpoint_time = current_time
    
    # Final save
    final_path = output_path / "final_model"
    save_model = model.model if mode == "cggr" else model
    save_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    results = {
        "mode": mode, "model": model_name, "hours": hours,
        "total_steps": state.total_steps, "total_tokens": state.total_tokens,
        "final_loss": state.current_loss, "eval_history": state.eval_history,
        "optimizer": "Muon",
    }
    
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    if use_wandb and HAS_WANDB:
        wandb.log({"final_loss": state.current_loss})
        wandb.finish()
    
    console.print(f"\n[bold green]COMPLETE: {mode.upper()} - Final Loss: {state.current_loss:.4f}[/bold green]")
    
    # Cleanup
    del model, base_model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def compare_results(std_results: Dict, cggr_results: Dict, output_dir: str):
    """Compare and generate final report."""
    console.print("\n" + "="*60)
    console.print("[bold cyan]FINAL COMPARISON REPORT[/bold cyan]")
    console.print("="*60)
    
    table = Table(title="Standard vs CGGR Code Training", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Standard", justify="right")
    table.add_column("CGGR", justify="right")
    table.add_column("Δ", justify="right", style="green")
    
    std_loss = std_results.get("final_loss", 0)
    cggr_loss = cggr_results.get("final_loss", 0)
    std_steps = std_results.get("total_steps", 0)
    cggr_steps = cggr_results.get("total_steps", 0)
    
    table.add_row("Final Loss", f"{std_loss:.4f}", f"{cggr_loss:.4f}", f"{(cggr_loss - std_loss):+.4f}")
    table.add_row("Total Steps", f"{std_steps:,}", f"{cggr_steps:,}", f"{cggr_steps - std_steps:+,}")
    table.add_row("Tokens Processed", f"{std_results.get('total_tokens', 0):,}", f"{cggr_results.get('total_tokens', 0):,}", "")
    
    console.print(table)
    
    if std_steps > 0 and cggr_steps > 0:
        throughput_ratio = cggr_steps / std_steps
        console.print(f"\n[bold]Throughput Ratio: {throughput_ratio:.2f}x[/bold]")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "standard": std_results,
        "cggr": cggr_results,
    }
    
    with open(output_path / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Code Fine-tune Benchmark")
    parser.add_argument("--mode", choices=["standard", "cggr"], help="Training mode")
    parser.add_argument("--all", action="store_true", help="Run BOTH modes")
    parser.add_argument("--hours", type=float, default=12.0, help="Training duration per mode")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM3-3B-Base", help="Model to fine-tune")
    parser.add_argument("--output", default="results/code_finetune", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate for Muon")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", default="cggr-code-benchmark", help="W&B project")
    parser.add_argument("--no-flash-attention", dest="flash_attention", action="store_false", default=True)
    
    args = parser.parse_args()
    
    if args.all:
        console.print("[bold cyan]Running FULL CODE BENCHMARK (Standard + CGGR)[/bold cyan]\n")
        
        std_results = run_training(
            mode="standard", hours=args.hours, model_name=args.model,
            output_dir=f"{args.output}/standard_{args.hours}h",
            batch_size=args.batch_size, gradient_accumulation=args.grad_accum,
            learning_rate=args.lr, use_wandb=args.wandb, wandb_project=args.wandb_project,
            flash_attention=args.flash_attention,
        )
        
        cggr_results = run_training(
            mode="cggr", hours=args.hours, model_name=args.model,
            output_dir=f"{args.output}/cggr_{args.hours}h",
            batch_size=args.batch_size, gradient_accumulation=args.grad_accum,
            learning_rate=args.lr, use_wandb=args.wandb, wandb_project=args.wandb_project,
            flash_attention=args.flash_attention,
        )
        
        compare_results(std_results, cggr_results, args.output)
        
    elif args.mode:
        run_training(
            mode=args.mode, hours=args.hours, model_name=args.model,
            output_dir=f"{args.output}/{args.mode}_{args.hours}h",
            batch_size=args.batch_size, gradient_accumulation=args.grad_accum,
            learning_rate=args.lr, use_wandb=args.wandb, wandb_project=args.wandb_project,
            flash_attention=args.flash_attention,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
