#!/usr/bin/env python3
"""
Extended Math Fine-tune Benchmark
=================================
Compare Standard vs CGGR fine-tuning on GSM8K over extended training runs.

Usage:
    python math_finetune_benchmark.py --all --hours 6          # Run both automatically
    python math_finetune_benchmark.py --mode standard --hours 6
    python math_finetune_benchmark.py --mode cggr --hours 6
"""

import argparse
import json
import time
import re
import os
import gc
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

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


# =============================================================================
# DATA
# =============================================================================

class MathDataset(Dataset):
    """Combined math reasoning dataset."""
    
    def __init__(self, tokenizer, max_length: int = 512, split: str = "train"):
        from datasets import load_dataset
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        console.print(f"[yellow]Loading {split} datasets...[/yellow]")
        
        # GSM8K
        try:
            gsm8k = load_dataset("openai/gsm8k", "main", split=split)
            for item in gsm8k:
                self.samples.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "source": "gsm8k"
                })
            console.print(f"[green]✓ GSM8K: {len(gsm8k)} samples[/green]")
        except Exception as e:
            console.print(f"[red]GSM8K failed: {e}[/red]")
        
        # MetaMathQA (only for training)
        if split == "train":
            try:
                metamath = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
                count = 0
                max_metamath = 50000
                for item in metamath:
                    if count >= max_metamath:
                        break
                    self.samples.append({
                        "question": item["query"],
                        "answer": item["response"],
                        "source": "metamath"
                    })
                    count += 1
                console.print(f"[green]✓ MetaMathQA: {count} samples[/green]")
            except Exception as e:
                console.print(f"[yellow]MetaMathQA skipped: {e}[/yellow]")
        
        console.print(f"[bold green]Total samples: {len(self.samples)}[/bold green]")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = f"Question: {sample['question']}\n\nAnswer: {sample['answer']}"
        
        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0),
        }


class GSM8KTestDataset(Dataset):
    """GSM8K test set for evaluation."""
    
    def __init__(self, tokenizer, max_length: int = 512):
        from datasets import load_dataset
        
        self.tokenizer = tokenizer
        self.samples = []
        
        gsm8k = load_dataset("openai/gsm8k", "main", split="test")
        for item in gsm8k:
            answer_match = re.search(r"####\s*([\d,.-]+)", item["answer"])
            numerical_answer = answer_match.group(1).replace(",", "") if answer_match else ""
            
            self.samples.append({
                "question": item["question"],
                "full_answer": item["answer"],
                "numerical_answer": numerical_answer,
            })
        
        console.print(f"[green]✓ GSM8K test: {len(self.samples)} samples[/green]")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# =============================================================================
# EVALUATION
# =============================================================================

def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from model output."""
    match = re.search(r"####\s*([\d,.-]+)", text)
    if match:
        return match.group(1).replace(",", "")
    
    match = re.search(r"(?:answer|result|equals?|is)[:\s]*([\d,.-]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    
    numbers = re.findall(r"[\d,.-]+", text)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return None


def evaluate_gsm8k(model, tokenizer, test_dataset, device: str, num_samples: int = 100) -> Dict[str, Any]:
    """Evaluate model on GSM8K test set."""
    model.eval()
    correct = 0
    total = 0
    
    indices = list(range(min(num_samples, len(test_dataset))))
    
    with torch.no_grad():
        for idx in indices:
            sample = test_dataset[idx]
            prompt = f"Question: {sample['question']}\n\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            try:
                outputs = model.generate(
                    **inputs, max_new_tokens=256, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                predicted = extract_answer(response)
                expected = sample["numerical_answer"]
                
                if predicted == expected:
                    correct += 1
            except:
                pass
            
            total += 1
    
    model.train()
    return {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}


# =============================================================================
# TRAINING
# =============================================================================

@dataclass
class TrainingState:
    start_time: float
    total_steps: int = 0
    total_tokens: int = 0
    current_loss: float = 0.0
    best_accuracy: float = 0.0
    eval_history: List[Dict] = None
    
    def __post_init__(self):
        if self.eval_history is None:
            self.eval_history = []


def run_training(
    mode: str,
    hours: float,
    model_name: str,
    output_dir: str,
    batch_size: int = 2,
    gradient_accumulation: int = 4,
    learning_rate: float = 5e-5,
    eval_interval_min: int = 30,
    checkpoint_interval_min: int = 60,
    use_wandb: bool = False,
    wandb_project: str = "cggr-math-benchmark",
) -> Dict[str, Any]:
    """Run extended training."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]MATH FINE-TUNE - {mode.upper()} MODE[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"Model: {model_name}")
    console.print(f"Duration: {hours} hours")
    console.print()
    
    # W&B init
    if use_wandb and HAS_WANDB:
        wandb.init(
            project=wandb_project,
            name=f"{mode}_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M')}",
            config={
                "mode": mode, "hours": hours, "model": model_name,
                "batch_size": batch_size, "grad_accum": gradient_accumulation, "lr": learning_rate,
            }
        )
        console.print("[green]✓ W&B initialized[/green]")
    
    # Load model
    console.print("[yellow]Loading model...[/yellow]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32  # Use float32, let autocast handle precision
    ).to(device)
    
    if hasattr(base_model, 'gradient_checkpointing_enable'):
        base_model.gradient_checkpointing_enable()
    
    if mode == "cggr":
        from cggr import CGGRModel, create_truncated_router
        router = create_truncated_router(base_model, num_layers=4)
        model = CGGRModel(base_model, router=router, min_tokens_ratio=0.25, warmup_steps=500, selection='fixed_quota')
        console.print("[green]✓ CGGR model initialized[/green]")
    else:
        model = base_model
        console.print("[green]✓ Standard model initialized[/green]")
    
    # Load datasets
    train_dataset = MathDataset(tokenizer, split="train")
    test_dataset = GSM8KTestDataset(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps_estimate = int((hours * 3600) / 0.5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps_estimate)
    scaler = torch.amp.GradScaler('cuda')  # Fixed deprecation warning
    
    # Training loop
    start_time = time.time()
    end_time = start_time + (hours * 3600)
    state = TrainingState(start_time=start_time)
    
    last_eval_time = start_time
    last_checkpoint_time = start_time
    eval_interval = eval_interval_min * 60
    checkpoint_interval = checkpoint_interval_min * 60
    
    console.print(f"\n[bold green]Training started. Expected end: {(datetime.now() + timedelta(hours=hours)).strftime('%H:%M:%S')}[/bold green]\n")
    
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
            
            with torch.amp.autocast('cuda'):
                if mode == "cggr":
                    loss = model(input_ids, labels=labels)
                else:
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss
                loss = loss / gradient_accumulation
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                if mode == "cggr" and hasattr(model, 'step'):
                    model.step()
                
                state.total_steps += 1
                state.current_loss = loss.item() * gradient_accumulation
                
                # W&B logging
                if use_wandb and HAS_WANDB and state.total_steps % 10 == 0:
                    wandb.log({"loss": state.current_loss, "step": state.total_steps})
            
            state.total_tokens += input_ids.numel()
        
        current_time = time.time()
        elapsed = current_time - start_time
        console.print(f"Epoch {epoch}: Steps={state.total_steps}, Loss={state.current_loss:.4f}, Elapsed={str(timedelta(seconds=int(elapsed)))}")
        
        # Periodic evaluation
        if current_time - last_eval_time >= eval_interval:
            console.print("[yellow]Evaluating...[/yellow]")
            eval_model = model.model if mode == "cggr" else model
            eval_result = evaluate_gsm8k(eval_model, tokenizer, test_dataset, device, num_samples=100)
            
            state.eval_history.append({
                "elapsed_hours": elapsed / 3600, "steps": state.total_steps,
                "accuracy": eval_result["accuracy"], "loss": state.current_loss,
            })
            
            if eval_result["accuracy"] > state.best_accuracy:
                state.best_accuracy = eval_result["accuracy"]
            
            console.print(f"[bold green]GSM8K Accuracy: {eval_result['accuracy']:.2%} (Best: {state.best_accuracy:.2%})[/bold green]")
            
            if use_wandb and HAS_WANDB:
                wandb.log({"gsm8k_accuracy": eval_result["accuracy"], "best_accuracy": state.best_accuracy})
            
            last_eval_time = current_time
        
        # Periodic checkpoint
        if current_time - last_checkpoint_time >= checkpoint_interval:
            ckpt_path = output_path / f"checkpoint_step{state.total_steps}"
            save_model = model.model if mode == "cggr" else model
            save_model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            last_checkpoint_time = current_time
    
    # Final evaluation
    console.print("\n[yellow]Final evaluation...[/yellow]")
    eval_model = model.model if mode == "cggr" else model
    final_eval = evaluate_gsm8k(eval_model, tokenizer, test_dataset, device, num_samples=200)
    
    # Save final model
    final_path = output_path / "final_model"
    save_model = model.model if mode == "cggr" else model
    save_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    results = {
        "mode": mode, "model": model_name, "hours": hours,
        "total_steps": state.total_steps, "total_tokens": state.total_tokens,
        "final_loss": state.current_loss, "final_accuracy": final_eval["accuracy"],
        "best_accuracy": state.best_accuracy, "eval_history": state.eval_history,
    }
    
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    if use_wandb and HAS_WANDB:
        wandb.log({"final_accuracy": final_eval["accuracy"]})
        wandb.finish()
    
    console.print(f"\n[bold green]COMPLETE: {mode.upper()} - Final Accuracy: {final_eval['accuracy']:.2%}[/bold green]")
    
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
    
    table = Table(title="Standard vs CGGR Comparison", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Standard", justify="right")
    table.add_column("CGGR", justify="right")
    table.add_column("Δ", justify="right", style="green")
    
    std_acc = std_results.get("final_accuracy", 0)
    cggr_acc = cggr_results.get("final_accuracy", 0)
    std_steps = std_results.get("total_steps", 0)
    cggr_steps = cggr_results.get("total_steps", 0)
    
    table.add_row("Final Accuracy", f"{std_acc:.2%}", f"{cggr_acc:.2%}", f"{(cggr_acc - std_acc):+.2%}")
    table.add_row("Best Accuracy", f"{std_results.get('best_accuracy', 0):.2%}", f"{cggr_results.get('best_accuracy', 0):.2%}", "")
    table.add_row("Total Steps", f"{std_steps:,}", f"{cggr_steps:,}", f"{cggr_steps - std_steps:+,}")
    table.add_row("Final Loss", f"{std_results.get('final_loss', 0):.4f}", f"{cggr_results.get('final_loss', 0):.4f}", "")
    
    console.print(table)
    
    # Speedup analysis
    if std_steps > 0 and cggr_steps > 0:
        throughput_ratio = cggr_steps / std_steps
        console.print(f"\n[bold]Throughput Ratio: {throughput_ratio:.2f}x (CGGR completed {throughput_ratio:.2f}x more steps in same time)[/bold]")
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "standard": std_results,
        "cggr": cggr_results,
        "comparison": {
            "accuracy_delta": cggr_acc - std_acc,
            "steps_delta": cggr_steps - std_steps,
            "throughput_ratio": cggr_steps / std_steps if std_steps > 0 else 0,
        }
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Markdown report
    md_report = f"""# Math Fine-tune Benchmark Results

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Summary

| Metric | Standard | CGGR | Delta |
|:--|--:|--:|--:|
| Final Accuracy | {std_acc:.2%} | {cggr_acc:.2%} | {(cggr_acc - std_acc):+.2%} |
| Total Steps | {std_steps:,} | {cggr_steps:,} | {cggr_steps - std_steps:+,} |
| Throughput Ratio | 1.0x | {cggr_steps / max(std_steps, 1):.2f}x | - |

## Conclusion
CGGR achieved **{cggr_steps / max(std_steps, 1):.2f}x more training steps** in the same wall-clock time, 
resulting in a **{(cggr_acc - std_acc):+.2%}** accuracy difference on GSM8K.
"""
    
    with open(output_path / "comparison_report.md", "w") as f:
        f.write(md_report)
    
    console.print(f"\n[green]Reports saved to {output_path}[/green]")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Extended Math Fine-tune Benchmark")
    parser.add_argument("--mode", choices=["standard", "cggr"], help="Training mode (use --all for both)")
    parser.add_argument("--all", action="store_true", help="Run BOTH standard and CGGR automatically")
    parser.add_argument("--hours", type=float, default=6.0, help="Training duration per mode")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M", help="Model to fine-tune")
    parser.add_argument("--output", default="results/math_finetune", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2 for 12GB VRAM)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--eval-interval", type=int, default=30, help="Eval interval in minutes")
    parser.add_argument("--checkpoint-interval", type=int, default=60, help="Checkpoint interval in minutes")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", default="cggr-math-benchmark", help="W&B project name")
    
    args = parser.parse_args()
    
    if args.all:
        # Run both modes automatically
        console.print("[bold cyan]Running FULL BENCHMARK (Standard + CGGR)[/bold cyan]\n")
        
        # Standard first
        std_results = run_training(
            mode="standard", hours=args.hours, model_name=args.model,
            output_dir=f"{args.output}/standard_{args.hours}h",
            batch_size=args.batch_size, gradient_accumulation=args.grad_accum,
            learning_rate=args.lr, eval_interval_min=args.eval_interval,
            checkpoint_interval_min=args.checkpoint_interval,
            use_wandb=args.wandb, wandb_project=args.wandb_project,
        )
        
        # Then CGGR
        cggr_results = run_training(
            mode="cggr", hours=args.hours, model_name=args.model,
            output_dir=f"{args.output}/cggr_{args.hours}h",
            batch_size=args.batch_size, gradient_accumulation=args.grad_accum,
            learning_rate=args.lr, eval_interval_min=args.eval_interval,
            checkpoint_interval_min=args.checkpoint_interval,
            use_wandb=args.wandb, wandb_project=args.wandb_project,
        )
        
        # Compare
        compare_results(std_results, cggr_results, args.output)
        
    elif args.mode:
        run_training(
            mode=args.mode, hours=args.hours, model_name=args.model,
            output_dir=f"{args.output}/{args.mode}_{args.hours}h",
            batch_size=args.batch_size, gradient_accumulation=args.grad_accum,
            learning_rate=args.lr, eval_interval_min=args.eval_interval,
            checkpoint_interval_min=args.checkpoint_interval,
            use_wandb=args.wandb, wandb_project=args.wandb_project,
        )
    else:
        parser.print_help()
        console.print("\n[yellow]Use --all to run both modes, or --mode to run one.[/yellow]")


if __name__ == "__main__":
    main()
