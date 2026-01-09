"""
CGGR Expert Specialization Training
===================================
Fine-tunes a pre-trained model using selective routing and a curriculum.
Starts with full tokens (ratio=1.0) and anneals to sparse tokens (ratio=0.25).
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from cggr import CGGRModel, create_truncated_router
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

console = Console()

def main():
    # 1. Configuration
    model_name = "HuggingFaceTB/SmolLM-135M"
    data_name = "HuggingFaceFW/fineweb-edu"
    batch_size = 4
    seq_len = 512
    learning_rate = 2e-5
    num_steps = 1000
    warmup_steps = 200 # Steps to anneal from 1.0 to 0.25 tokens
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold green]Target Device:[/bold green] {device}")

    # 2. Setup Model & Router
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print(f"[cyan]Loading Pre-trained {model_name}...[/cyan]")
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    
    # Create truncated proxy router (4 layers)
    router = create_truncated_router(base_model, num_layers=4).to(device)
    
    # Wrap in CGGRModel with Curriculum
    model = CGGRModel(
        base_model, 
        router=router, 
        min_tokens_ratio=0.25, 
        warmup_steps=warmup_steps
    ).to(device)
    
    # 3. Setup Dataset
    console.print(f"[cyan]Streaming Dataset: {data_name}...[/cyan]")
    
    def get_dataset_iterator(name, split="train"):
        max_retries = 5
        for i in range(max_retries):
            try:
                ds = load_dataset(name, name="sample-10BT", split=split, streaming=True)
                return iter(ds)
            except Exception as e:
                console.print(f"[yellow]Dataset loading attempt {i+1} failed: {e}. Retrying...[/yellow]")
                time.sleep(2)
        # Fallback to wikitext if fineweb is unreachable
        console.print("[red]All fineweb-edu attempts failed. Falling back to Wikitext-2 for metrics.[/red]")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, streaming=True)
        return iter(ds)

    def get_dataloader():
        iterator = get_dataset_iterator(data_name)
        while True:
            texts = []
            try:
                for _ in range(batch_size):
                    sample = next(iterator)
                    texts.append(sample.get('text', sample.get('content', "")))
            except StopIteration:
                iterator = get_dataset_iterator(data_name)
                continue
            
            enc = tokenizer(
                texts, 
                truncation=True, 
                max_length=seq_len, 
                padding='max_length', 
                return_tensors='pt'
            )
            yield enc['input_ids'].to(device), enc['input_ids'].to(device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_steps)
    scaler = torch.cuda.amp.GradScaler()

    # 5. Evaluation Function (The Level Playing Field)
    def evaluate_ppl(model, dataset_iterator, num_batches=10):
        model.eval()
        total_loss = 0
        tokens_count = 0
        with torch.no_grad():
            for _ in range(num_batches):
                try:
                    texts = [next(dataset_iterator)['text'] for _ in range(batch_size)]
                    enc = tokenizer(texts, truncation=True, max_length=seq_len, padding='max_length', return_tensors='pt')
                    ids = enc['input_ids'].to(device)
                    
                    # We use the raw base model for PPL to avoid selective loss effects
                    # This ensures the PPL is comparable to a standard baseline
                    with torch.cuda.amp.autocast():
                        outputs = model.model(ids, labels=ids)
                        loss = outputs.loss
                    
                    total_loss += loss.item() * ids.numel()
                    tokens_count += ids.numel()
                except StopIteration:
                    break
        model.train()
        avg_loss = total_loss / tokens_count if tokens_count > 0 else 0
        return torch.exp(torch.tensor(avg_loss)).item()

    # 6. Training Loop
    model.train()
    dataloader = get_dataloader()
    
    # Separate iterator for PPL evaluation to not consume training data
    eval_iterator = get_dataset_iterator(data_name)
    
    console.print(f"\n[bold magenta]Starting Specialization Training ({num_steps} steps)[/bold magenta]")
    console.print(f"[dim]Curriculum: Ratio will anneal from 1.0 to 0.25 over {warmup_steps} steps.[/dim]")
    console.print(f"[dim]Metrics: PPL (Perplexity) is measured on the full token distribution.[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[yellow]Fine-tuning...", total=num_steps)
        
        current_ppl = 0.0
        for step in range(num_steps):
            try:
                input_ids, labels = next(dataloader)
            except Exception as e:
                console.print(f"[red]Batch loading error: {e}[/red]")
                continue
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, labels=labels)
                loss = outputs if isinstance(outputs, torch.Tensor) else outputs.loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Step the CGGR curriculum
            model.step()
            
            # Periodic PPL Evaluation
            if step % 50 == 0:
                current_ppl = evaluate_ppl(model, eval_iterator)
                
            if step % 5 == 0:
                metrics = model.get_metrics()
                ratio = metrics.get('current_ratio', 1.0)
                ppl_str = f" | PPL: {current_ppl:.2f}" if current_ppl > 0 else ""
                progress.update(
                    task, 
                    description=f"Step {step} | Train Loss: {loss.item():.2f}{ppl_str} | Ratio: {ratio:.2f}"
                )
            
            progress.advance(task)

    console.print("\n[bold green]✓ Specialization Training Complete![/bold green]")
    
    # Save the model
    # Note: CGGRModel.model holds the specialized weights
    base_model.save_pretrained("./specialized_model")
    tokenizer.save_pretrained("./specialized_model")
    console.print("[dim]Weights saved to ./specialized_model[/dim]")

if __name__ == "__main__":
    main()
