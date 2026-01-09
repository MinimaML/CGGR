#!/usr/bin/env python3
"""
Extended Math Fine-tune Benchmark
=================================
Compare Standard vs CGGR fine-tuning on GSM8K over extended training runs.

Usage:
    python math_finetune_benchmark.py --mode standard --hours 6 --output results/standard/
    python math_finetune_benchmark.py --mode cggr --hours 6 --output results/cggr/
"""

import argparse
import json
import time
import re
import os
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich import box

console = Console()


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
                max_metamath = 50000  # Limit to avoid memory issues
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
        
        # Format as instruction
        text = f"Question: {sample['question']}\n\nAnswer: {sample['answer']}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
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
        self.max_length = max_length
        self.samples = []
        
        gsm8k = load_dataset("openai/gsm8k", "main", split="test")
        for item in gsm8k:
            # Extract numerical answer from #### pattern
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
    # Look for #### pattern
    match = re.search(r"####\s*([\d,.-]+)", text)
    if match:
        return match.group(1).replace(",", "")
    
    # Look for "the answer is" pattern
    match = re.search(r"(?:answer|result|equals?|is)[:\s]*([\d,.-]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    
    # Last number in the text
    numbers = re.findall(r"[\d,.-]+", text)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return None


def evaluate_gsm8k(model, tokenizer, test_dataset: GSM8KTestDataset, device: str, num_samples: int = 100) -> Dict[str, Any]:
    """Evaluate model on GSM8K test set."""
    model.eval()
    
    correct = 0
    total = 0
    results = []
    
    # Sample subset for faster evaluation
    indices = list(range(min(num_samples, len(test_dataset))))
    
    with torch.no_grad():
        for idx in indices:
            sample = test_dataset[idx]
            
            # Prepare prompt
            prompt = f"Question: {sample['question']}\n\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                predicted = extract_answer(response)
                expected = sample["numerical_answer"]
                
                is_correct = predicted == expected
                if is_correct:
                    correct += 1
                
                results.append({
                    "question": sample["question"][:100],
                    "expected": expected,
                    "predicted": predicted,
                    "correct": is_correct,
                })
            except Exception as e:
                results.append({
                    "question": sample["question"][:100],
                    "expected": sample["numerical_answer"],
                    "predicted": None,
                    "correct": False,
                    "error": str(e),
                })
            
            total += 1
    
    model.train()
    
    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results[:10],  # First 10 for logging
    }


# =============================================================================
# TRAINING
# =============================================================================

@dataclass
class TrainingState:
    """Track training progress."""
    start_time: float
    total_steps: int = 0
    total_tokens: int = 0
    current_loss: float = 0.0
    best_accuracy: float = 0.0
    eval_history: List[Dict] = None
    
    def __post_init__(self):
        if self.eval_history is None:
            self.eval_history = []


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    state: TrainingState,
    end_time: float,
    scaler,
    use_cggr: bool = False,
    gradient_accumulation: int = 4,
) -> bool:
    """Train for one epoch or until time limit. Returns True if time exhausted."""
    model.train()
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        if time.time() >= end_time:
            return True  # Time exhausted
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        with torch.cuda.amp.autocast():
            if use_cggr:
                loss = model(input_ids, labels=labels)
            else:
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation
        
        scaler.scale(loss).backward()
        
        # Step every N batches
        if (batch_idx + 1) % gradient_accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            if use_cggr and hasattr(model, 'step'):
                model.step()
            
            state.total_steps += 1
        
        state.total_tokens += input_ids.numel()
        state.current_loss = loss.item() * gradient_accumulation  # Rescale for logging
    
    return False  # Epoch complete, time remains


def run_training(
    mode: str,
    hours: float,
    model_name: str,
    output_dir: str,
    batch_size: int = 2,  # Conservative for 12GB VRAM
    gradient_accumulation: int = 4,  # Effective batch = 8
    learning_rate: float = 5e-5,
    eval_interval_min: int = 30,
    checkpoint_interval_min: int = 60,
):
    """Run extended training."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]MATH FINE-TUNE BENCHMARK - {mode.upper()} MODE[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"Model: {model_name}")
    console.print(f"Duration: {hours} hours")
    console.print(f"Output: {output_dir}")
    console.print()
    
    # Load tokenizer and model
    console.print("[yellow]Loading model...[/yellow]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use fp16 to save VRAM
    ).to(device)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(base_model, 'gradient_checkpointing_enable'):
        base_model.gradient_checkpointing_enable()
        console.print("[green]✓ Gradient checkpointing enabled[/green]")
    
    if mode == "cggr":
        from cggr import CGGRModel, create_truncated_router
        router = create_truncated_router(base_model, num_layers=4)
        model = CGGRModel(
            base_model,
            router=router,
            min_tokens_ratio=0.25,
            warmup_steps=500,
            selection='fixed_quota',
        )
        console.print("[green]✓ CGGR model initialized[/green]")
    else:
        model = base_model
        console.print("[green]✓ Standard model initialized[/green]")
    
    # Load datasets
    train_dataset = MathDataset(tokenizer, split="train")
    test_dataset = GSM8KTestDataset(tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps_estimate = int((hours * 3600) / 0.5)  # Rough estimate
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=total_steps_estimate,
    )
    scaler = torch.cuda.amp.GradScaler()
    
    # Training state
    start_time = time.time()
    end_time = start_time + (hours * 3600)
    state = TrainingState(start_time=start_time)
    
    last_eval_time = start_time
    last_checkpoint_time = start_time
    eval_interval = eval_interval_min * 60
    checkpoint_interval = checkpoint_interval_min * 60
    
    console.print(f"\n[bold green]Training started at {datetime.now().strftime('%H:%M:%S')}[/bold green]")
    console.print(f"[bold green]Expected end: {(datetime.now() + timedelta(hours=hours)).strftime('%H:%M:%S')}[/bold green]\n")
    
    epoch = 0
    while time.time() < end_time:
        epoch += 1
        console.print(f"[cyan]Epoch {epoch}...[/cyan]")
        
        time_exhausted = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            state=state,
            end_time=end_time,
            scaler=scaler,
            use_cggr=(mode == "cggr"),
            gradient_accumulation=gradient_accumulation,
        )
        
        current_time = time.time()
        elapsed = current_time - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        console.print(f"  Steps: {state.total_steps}, Loss: {state.current_loss:.4f}, Elapsed: {elapsed_str}")
        
        # Periodic evaluation
        if current_time - last_eval_time >= eval_interval:
            console.print("\n[yellow]Running evaluation...[/yellow]")
            eval_result = evaluate_gsm8k(
                model.model if mode == "cggr" else model,
                tokenizer,
                test_dataset,
                device,
                num_samples=100,
            )
            
            state.eval_history.append({
                "elapsed_hours": elapsed / 3600,
                "steps": state.total_steps,
                "accuracy": eval_result["accuracy"],
                "loss": state.current_loss,
            })
            
            if eval_result["accuracy"] > state.best_accuracy:
                state.best_accuracy = eval_result["accuracy"]
            
            console.print(f"[bold green]GSM8K Accuracy: {eval_result['accuracy']:.2%} ({eval_result['correct']}/{eval_result['total']})[/bold green]")
            console.print(f"[bold green]Best Accuracy: {state.best_accuracy:.2%}[/bold green]\n")
            
            last_eval_time = current_time
        
        # Periodic checkpoint
        if current_time - last_checkpoint_time >= checkpoint_interval:
            ckpt_path = output_path / f"checkpoint_step{state.total_steps}"
            console.print(f"[yellow]Saving checkpoint: {ckpt_path}[/yellow]")
            
            save_model = model.model if mode == "cggr" else model
            save_model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            
            last_checkpoint_time = current_time
        
        if time_exhausted:
            break
    
    # Final evaluation
    console.print("\n[bold yellow]Running final evaluation...[/bold yellow]")
    final_eval = evaluate_gsm8k(
        model.model if mode == "cggr" else model,
        tokenizer,
        test_dataset,
        device,
        num_samples=200,  # More samples for final eval
    )
    
    # Save final model
    final_path = output_path / "final_model"
    save_model = model.model if mode == "cggr" else model
    save_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Save results
    results = {
        "mode": mode,
        "model": model_name,
        "hours": hours,
        "total_steps": state.total_steps,
        "total_tokens": state.total_tokens,
        "final_loss": state.current_loss,
        "final_accuracy": final_eval["accuracy"],
        "best_accuracy": state.best_accuracy,
        "eval_history": state.eval_history,
        "final_eval": final_eval,
    }
    
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    console.print("\n" + "="*60)
    console.print("[bold cyan]TRAINING COMPLETE[/bold cyan]")
    console.print("="*60)
    console.print(f"Mode: {mode.upper()}")
    console.print(f"Total Steps: {state.total_steps}")
    console.print(f"Total Tokens: {state.total_tokens:,}")
    console.print(f"Final Loss: {state.current_loss:.4f}")
    console.print(f"[bold green]Final GSM8K Accuracy: {final_eval['accuracy']:.2%}[/bold green]")
    console.print(f"[bold green]Best GSM8K Accuracy: {state.best_accuracy:.2%}[/bold green]")
    console.print("="*60)
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extended Math Fine-tune Benchmark")
    parser.add_argument("--mode", choices=["standard", "cggr"], required=True, help="Training mode")
    parser.add_argument("--hours", type=float, default=6.0, help="Training duration in hours")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M", help="Model to fine-tune")
    parser.add_argument("--output", default="results/math_finetune", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2 for 12GB VRAM)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--eval-interval", type=int, default=30, help="Eval interval in minutes")
    parser.add_argument("--checkpoint-interval", type=int, default=60, help="Checkpoint interval in minutes")
    
    args = parser.parse_args()
    
    output_dir = f"{args.output}/{args.mode}_{args.hours}h"
    
    results = run_training(
        mode=args.mode,
        hours=args.hours,
        model_name=args.model,
        output_dir=output_dir,
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accum,
        learning_rate=args.lr,
        eval_interval_min=args.eval_interval,
        checkpoint_interval_min=args.checkpoint_interval,
    )
    
    return results


if __name__ == "__main__":
    main()
