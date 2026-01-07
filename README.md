# Confidence-Gated Gradient Routing (CGGR)

CGGR is a PyTorch library that provides **selective loss computation** for Transformer training. By excluding easy tokens from loss, gradients are only computed for hard tokens - providing actual backward pass savings.

## Quick Start

```python
from cggr import CGGRLoss

# Replace your loss function
criterion = CGGRLoss(min_tokens_ratio=0.25, warmup_steps=1000)

for batch in dataloader:
    logits = model(input_ids)
    
    # Only hard tokens contribute to loss and backprop
    loss = criterion(logits, targets)
    loss.backward()
    
    optimizer.step()
    criterion.step()
    
    print(criterion.get_metrics())  # tokens_used, avg_confidence, etc.
```

## How It Works

### Token Selection
Each token gets a **difficulty score** based on model confidence:
$$D_t = \text{Entropy}(P_t) - \text{Confidence}(P_t)$$

High difficulty = model is uncertain → include in loss
Low difficulty = model is confident → exclude from loss

### Curriculum Learning
Starts with all tokens, gradually reduces to `min_tokens_ratio`:
- Step 0: 100% of tokens in loss
- Step N (warmup): `min_tokens_ratio` of tokens (e.g., 25%)

### Actual Compute Savings
Unlike gradient masking (which adds overhead), excluding tokens from loss **prevents gradient computation entirely**:

| Approach           | Backward FLOPs | Overhead |
| ------------------ | -------------- | -------- |
| Gradient Hooks     | 100%           | +4%      |
| **Selective Loss** | **25-75%**     | **~0%**  |

## Configuration

```python
CGGRLoss(
    base_loss=nn.CrossEntropyLoss(reduction='none'),  # Any per-token loss
    min_tokens_ratio=0.25,  # Target: 25% of tokens
    warmup_steps=1000,      # Steps to reach target
)
```

## Compatibility

- **Triton Required**: CUDA GPU with Triton for fused difficulty scoring
- **Any Transformer**: Works with any model that outputs logits
- **SRDE Optimized**: "Double Sparsity" - sparse forward (MoE) + sparse backward (CGGR)

## Installation

```bash
pip install cggr
```

> [!IMPORTANT]
> Requires CUDA GPU and Triton. CPU training not supported.
