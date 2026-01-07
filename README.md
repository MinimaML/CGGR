# CGGR - Confidence-Gated Gradient Routing

Selective loss computation for Transformer training. Only hard tokens contribute to loss, providing actual backward pass savings.

## Installation

```bash
pip install cggr
```

> Requires CUDA + Triton.

## Quick Start

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
