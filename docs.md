# CGGR Documentation

Complete documentation for Confidence-Gated Gradient Routing.

---

## Table of Contents

1. [Installation](#installation)
2. [Core Concepts](#core-concepts)
3. [CGGRLoss API](#cggrloss-api)
4. [Scoring Strategies](#scoring-strategies)
5. [Selection Strategies](#selection-strategies)
6. [Dynamic Thresholding](#dynamic-thresholding)
7. [Complete Examples](#complete-examples)
8. [Integration with SRDE](#integration-with-srde)
9. [Troubleshooting](#troubleshooting)

---

## Installation

```bash
pip install cggr
```

**Requirements:**
- Python ≥ 3.8
- PyTorch ≥ 2.0
- Triton ≥ 2.0
- CUDA GPU

---

## Core Concepts

### The Problem
In language model training, not all tokens are equally valuable:
- **Easy tokens**: Model is confident → low learning signal
- **Hard tokens**: Model is uncertain → high learning signal

Standard training wastes compute on easy tokens that don't improve the model.

### The Solution
CGGR excludes easy tokens from the loss function. Tokens not in the loss don't generate gradients, saving backward pass compute.

```
Standard:  loss = CrossEntropy(ALL tokens)     → 100% backward FLOPs
CGGR:      loss = CrossEntropy(HARD tokens)    → 25% backward FLOPs
```

### Comparison

| Metric              | Standard Training            | CGGR (Batch Split)       | Benefit                               |
| :------------------ | :--------------------------- | :----------------------- | :------------------------------------ |
| **Backward Pass**   | 100% of tokens               | 25% of tokens            | **4x cheaper** backward pass          |
| **Forward Pass**    | 1.0x cost                    | ~1.1x cost (Pass 1 + 2)  | Negligible overhead (~9ms)            |
| **Total Speed**     | 1.0x (Baseline)              | **1.4x - 2.0x faster**   | Significant training acceleration     |
| **Data Efficiency** | Learns from all tokens       | Prioritizes hard tokens  | Learns faster from hard examples      |
| **Gradient Noise**  | High (easy tokens add noise) | Low                      | Cleaner gradients, faster convergence |
| **Memory**          | High (full graph)            | **Lower** (sparse graph) | Can increase batch size               |

---

## Batch Splitting (Recommended)

The most efficient way to use CGGR is via `CGGRModel`, which implements "Batch Splitting".

### Why?
- **Speed:** Only runs the backward pass on hard tokens (1.4x - 2x faster).
- **Efficiency:** Uses a lightweight truncated router (4 layers) to score difficulty.

### Usage

```python
from cggr import CGGRModel, create_truncated_router
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("...").cuda()

# 1. Create a lightweight router (shares weights with model, 0 extra memory)
# Uses first 4 layers to estimate difficulty
router = create_truncated_router(model, num_layers=4)

# 2. Wrap the model
cggr_model = CGGRModel(
    model, 
    router=router,           # Optimized difficulty scoring
    min_tokens_ratio=0.25,   # Keep top 25% hardest tokens
    warmup_steps=1000        # Gradually reduce to 25%
)

# 3. Train as usual
loss = cggr_model(input_ids, labels=labels)
loss.backward()
```

---

## CGGRLoss API (Manual Integration)

If you cannot use `CGGRModel` (e.g. specialized architectures), you can use `CGGRLoss`.
*Note: This effectively sparsifies gradients but does NOT save backward pass FLOPs unless combined with custom slicing.*

```python
from cggr import CGGRLoss
import torch.nn as nn

# Create loss function
criterion = CGGRLoss()

# Training loop
for batch in dataloader:
    logits = model(input_ids)  # (batch, seq, vocab)
    targets = batch['labels']   # (batch, seq)
    
    loss = criterion(logits, targets)
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    criterion.step()  # Important: advance curriculum
```

### Full Constructor

```python
CGGRLoss(
    # Scoring method
    scoring='combined',           # How to measure token difficulty
    
    # Selection method
    selection='topk',             # How to choose which tokens to keep
    
    # Dynamic adjustment
    dynamic_threshold=True,       # Adjust ratio based on batch confidence
    threshold_sensitivity=0.5,    # How much to adjust (0-1)
    
    # Curriculum
    min_tokens_ratio=0.25,        # Target: keep 25% of tokens
    warmup_steps=1000,            # Steps to reach target ratio
    
    # Strategy-specific
    num_strata=4,                 # For stratified selection
    min_tokens_per_sequence=1,    # For sequence_aware selection
    
    # Base loss
    base_loss=nn.CrossEntropyLoss(reduction='none'),
)
```

### Methods

| Method                     | Description                                    |
| -------------------------- | ---------------------------------------------- |
| `forward(logits, targets)` | Compute selective loss                         |
| `step()`                   | Advance curriculum (call after optimizer.step) |
| `get_metrics()`            | Get current statistics                         |

### Metrics

```python
metrics = criterion.get_metrics()
print(metrics)
# {
#     'step': 500,
#     'token_ratio': 0.625,        # Current ratio (during warmup)
#     'tokens_selected': 160,      # Tokens in this batch's loss
#     'tokens_total': 256,
#     'avg_confidence': 0.72,
#     'avg_entropy': 1.34,
#     'avg_difficulty': 0.82,
#     'selection': 'topk',
#     'scoring': 'combined',
# }
```

---

## Scoring Strategies

How CGGR measures token difficulty.

### `'entropy'`
Pure uncertainty-based. High entropy = hard.

$$\text{difficulty} = -\sum_i p_i \log p_i$$

```python
CGGRLoss(scoring='entropy')
```

**Best for:** General language modeling where uncertainty is the main signal.

---

### `'margin'`
Distance between top-2 predictions. Small margin = hard.

$$\text{difficulty} = 1 - (p_{\text{top1}} - p_{\text{top2}})$$

```python
CGGRLoss(scoring='margin')
```

**Best for:** Classification tasks, multiple-choice, when the model is "torn" between options.

---

### `'loss'`
Direct use of per-token cross-entropy loss. High loss = hard.

$$\text{difficulty} = -\log p_{\text{target}}$$

```python
CGGRLoss(scoring='loss')
```

**Best for:** Direct optimization, when you want to focus on what the model gets wrong.

---

### `'combined'` (default)
All signals combined: entropy + margin + loss.

```python
CGGRLoss(scoring='combined')
```

**Best for:** Most scenarios. Robust across different training phases.

---

## Selection Strategies

How CGGR chooses which tokens to keep.

### `'topk'` (default)
Simple top-k selection by difficulty.

```python
CGGRLoss(selection='topk')
```

**Pros:** Fast, simple, predictable
**Cons:** May exclude entire difficulty ranges

---

### `'stratified'`
Samples proportionally from difficulty buckets.

```python
CGGRLoss(
    selection='stratified',
    num_strata=4,  # 4 difficulty buckets
)
```

**How it works:**
1. Divide tokens into N buckets by difficulty
2. Sample more from hard buckets, fewer from easy
3. Distribution: [40%, 30%, 20%, 10%] by default

**Pros:** Prevents catastrophic forgetting, balanced learning
**Cons:** Slightly more overhead

---

### `'sequence_aware'`
Ensures minimum coverage per sequence.

```python
CGGRLoss(
    selection='sequence_aware',
    min_tokens_per_sequence=1,
)
```

**How it works:**
1. First does top-k selection
2. Then ensures at least N tokens per sequence

**Pros:** Preserves sequence structure, prevents excluding entire sentences
**Cons:** May select more tokens than target ratio

---

## Dynamic Thresholding

Automatically adjusts token ratio based on batch confidence.

```python
CGGRLoss(
    dynamic_threshold=True,
    threshold_sensitivity=0.5,  # 0-1
)
```

**Behavior:**
- Low batch confidence → increase ratio (more tokens)
- High batch confidence → decrease ratio (fewer tokens)

**Formula:**
```
adjusted_ratio = base_ratio × (1 + (1 - avg_confidence) × sensitivity)
```

**Example:**
| Avg Confidence  | Base Ratio | Sensitivity | Adjusted Ratio |
| --------------- | ---------- | ----------- | -------------- |
| 0.3 (learning)  | 0.25       | 0.5         | 0.34           |
| 0.7 (trained)   | 0.25       | 0.5         | 0.29           |
| 0.9 (converged) | 0.25       | 0.5         | 0.26           |

---

## Complete Examples

### Basic Language Model Training

```python
from cggr import CGGRLoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

criterion = CGGRLoss(
    scoring='combined',
    selection='stratified',
    min_tokens_ratio=0.25,
    warmup_steps=1000,
)

for epoch in range(10):
    for batch in dataloader:
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()
        
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1, :]  # Shift for next-token prediction
        targets = labels[:, 1:]
        
        loss = criterion(logits, targets)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        criterion.step()
        
        if step % 100 == 0:
            print(f"Step {step}: loss={loss.item():.4f}, "
                  f"tokens={criterion.get_metrics()['tokens_selected']}")
```

### With Hugging Face Trainer

```python
from transformers import Trainer, TrainingArguments
from cggr import CGGRLoss

class CGGRTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cggr = CGGRLoss(min_tokens_ratio=0.25)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = inputs['labels'][:, 1:]
        
        loss = self.cggr(logits, labels)
        self.cggr.step()
        
        return (loss, outputs) if return_outputs else loss
```

### Monitoring Training

```python
import wandb

for step, batch in enumerate(dataloader):
    # ... training code ...
    
    if step % 10 == 0:
        metrics = criterion.get_metrics()
        wandb.log({
            'loss': loss.item(),
            'cggr/token_ratio': metrics['token_ratio'],
            'cggr/tokens_selected': metrics['tokens_selected'],
            'cggr/avg_confidence': metrics['avg_confidence'],
            'cggr/avg_difficulty': metrics['avg_difficulty'],
        }, step=step)
```

---

## Integration with SRDE

CGGR was designed to work with [SRDE](https://github.com/MinimaML/SRDE) (Sparse Routed Delta Experts) for "Double Sparsity":

- **Forward Sparsity:** SRDE routes tokens to specialized experts (sparse MoE)
- **Backward Sparsity:** CGGR excludes easy tokens from loss (sparse gradients)

```python
from srde import SRDEModel
from cggr import CGGRLoss

model = SRDEModel(config).cuda()
criterion = CGGRLoss(scoring='combined', min_tokens_ratio=0.25)

# Combined: sparse forward + sparse backward = maximum efficiency
```

---

## Troubleshooting

### "Triton not found"
```bash
pip install triton
```
CGGR requires Triton for CUDA kernels. CPU training is not supported.

### Loss is NaN
- Check that `min_tokens_ratio` > 0
- Ensure logits and targets have matching shapes
- Verify targets are valid token indices

### Training instability
- Increase `warmup_steps` (try 2000-5000)
- Increase `min_tokens_ratio` (try 0.5)
- Use `selection='stratified'` instead of `topk`

### Low tokens_selected
This is expected after warmup completes. Check `token_ratio` in metrics.

### "Tokens per sequence" warnings
Use `selection='sequence_aware'` to ensure coverage:
```python
CGGRLoss(selection='sequence_aware', min_tokens_per_sequence=2)
```

---

## Citation

```bibtex
@software{cggr2026,
  title={CGGR: Confidence-Gated Gradient Routing},
  author={MinimaML},
  year={2026},
  url={https://github.com/MinimaML/CGGR}
}
```
