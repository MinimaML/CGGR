# CGGR - Confidence-Gated Gradient Routing

Selective loss computation for Transformer training. Only hard tokens contribute to loss, providing actual backward pass savings.

## Installation

```bash
pip install cggr
```

> Requires CUDA + Triton.

## Why CGGR?

| Metric              | Standard Training      | CGGR (Batch Split)      | Benefit                           |
| :------------------ | :--------------------- | :---------------------- | :-------------------------------- |
| **Backward Pass**   | 100% of tokens         | 25% of tokens           | 4x cheaper backward pass          |
| **Forward Pass**    | 1.0x cost              | ~1.1x cost (Pass 1 + 2) | Negligible overhead (~9ms)        |
| **Total Speed**     | 1.0x (Baseline)        | 1.4x - 2.0x faster      | Significant training acceleration |
| **Data Efficiency** | Learns from all tokens | Prioritizes hard tokens | Learns faster from hard examples  |
| **Memory**          | High (full graph)      | Lower (sparse graph)    | Can increase batch size           |

## Benchmarks

**Hardware:** RTX 3060
**Model:** SmolLM-135M (Llama architecture)  
**Dataset:** FineWeb-Edu

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'fontSize': '16px'}}}%%
xychart-beta
    title "Training Step Time (Lower is Better)"
    x-axis [Standard, CGGR (Optimized)]
    y-axis "Time (ms)" 0 --> 350
    bar [309, 220]
```

| Configuration         | Forward (ms) | Backward (ms) | Total Step (ms) | **Speedup** |
| :-------------------- | :----------- | :------------ | :-------------- | :---------- |
| **Standard Training** | 118 ms       | 185 ms        | 309 ms          | 1.0x        |
| **CGGR (Optimized)**  | **127 ms**   | **93 ms**     | **220 ms**      | **1.40x**   |

## Quick Start

### 1. Batch Splitting (Recommended)

The most efficient way to use CGGR is via `CGGRModel`. It uses a lightweight router to score difficulty and only computes gradients for hard tokens.

```python
from cggr import CGGRModel, create_truncated_router
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("...").cuda()

# Create lightweight router (shares weights, 0 extra memory)
router = create_truncated_router(model, num_layers=4)

# Wrap model
cggr_model = CGGRModel(
    model, 
    router=router, 
    min_tokens_ratio=0.25
)

# Train
loss = cggr_model(input_ids, labels=labels)
loss.backward()
```

### 2. Manual Integration (CGGRLoss)


```python
from cggr import CGGRLoss

criterion = CGGRLoss(
    scoring='combined',      # 'entropy', 'margin', 'loss', 'combined'
    selection='stratified',  # 'topk', 'stratified', 'sequence_aware'
    min_tokens_ratio=0.25,
    warmup_steps=1000,
)

for batch in dataloader:
    logits = model(input_ids)
    loss = criterion(logits, targets)  # Only hard tokens
    loss.backward()
    optimizer.step()
    criterion.step()
```

## Scoring Strategies

| Strategy   | Description               | Best For            |
| ---------- | ------------------------- | ------------------- |
| `entropy`  | High entropy = hard       | General training    |
| `margin`   | Small top-2 margin = hard | Classification      |
| `loss`     | High loss = hard          | Direct optimization |
| `combined` | All signals combined      | Best overall        |

## Selection Strategies

| Strategy         | Description                    | Benefit             |
| ---------------- | ------------------------------ | ------------------- |
| `topk`           | Top-k hardest tokens           | Simple, fast        |
| `stratified`     | Sample from difficulty buckets | Prevents forgetting |
| `sequence_aware` | Ensure coverage per sequence   | Preserves structure |

## Dynamic Thresholding

Automatically adjusts token ratio based on batch confidence:
- Low confidence → more tokens (model is learning)
- High confidence → fewer tokens (model has converged)

```python
CGGRLoss(dynamic_threshold=True, threshold_sensitivity=0.5)
```

## Full API

```python
CGGRLoss(
    # Scoring
    scoring='combined',
    
    # Selection
    selection='topk',
    num_strata=4,                  # For stratified
    min_tokens_per_sequence=1,     # For sequence_aware
    
    # Thresholding
    dynamic_threshold=True,
    threshold_sensitivity=0.5,
    
    # Curriculum
    min_tokens_ratio=0.25,
    warmup_steps=1000,
)
```

## Performance

| Config                | Backward FLOPs | Overhead |
| --------------------- | -------------- | -------- |
| Standard Loss         | 100%           | 0%       |
| **CGGR (25% tokens)** | **~25%**       | **~0%**  |
