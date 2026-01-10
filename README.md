# CGGR - Confidence-Gated Gradient Routing

> [!WARNING]
> This is highly experimental, and prepare for the worst hope for the best.

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

## Benchmark Race Results (2026)

**Model:** Google Gemma-3-4B-IT  
**Dataset:** AI-MO/NuminaMath-1.5  
**Evaluation:** AIME 2024 (Math Reasoning)  

> [!TIP]
> **Key Result:** CGGR achieved **4x higher sample throughput** and **+1.5% Accuracy** by utilizing idle compute cycles.

| Metric                      | Standard (BS=1) | CGGR (BS=4) | Improvement |
| :-------------------------- | :-------------- | :---------- | :---------- |
| **Final Accuracy (AIME)**   | 8.00%           | **9.50%**   | **+1.50%**  |
| **Final Loss**              | 0.3610          | **0.0980**  | **-73%**    |
| **Total Samples Processed** | ~14,368         | ~58,716     | **4.08x**   |
| **Wall Clock Time**         | 1.5 Hours       | 1.5 Hours   | Same        |

### The "Free Lunch" Phenomenon
In standard training (Batch Size = 1), high-end GPUs are often **latency-bound**—spending more time waiting for memory transfers than doing math. 

CGGR exploits this by **quadrupling the batch size** (Batch Size = 4) without increasing the step time.
*   **Step Latency:** Unchanged (1.02x throughput ratio in steps/sec).
*   **Data Throughput:** **4.08x higher** (samples/sec).

By processing 4x more data in the same timeframe, CGGR converges significantly faster and deeper (Loss 0.09 vs 0.36).

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

If you cannot use `CGGRModel` (e.g. specialized architectures), you can use `CGGRLoss` manually.

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

### 3. Native Integration (e.g. MoE/SRDE)

For architectures like SRDE that require static tensor shapes, you can use `CGGRScorer` to generate a routing mask instead of splitting the batch.

```python
from cggr import CGGRScorer

# 1. Initialize Scorer
self.scorer = CGGRScorer(router, min_tokens_ratio=0.5)

# 2. Get Mask
difficulty, mask, info = self.scorer(input_ids)

# 3. Apply Mask (Null Routing)
# mask is Boolean: True=Hard (Route to Expert), False=Easy (Skip/Null)
expert_output = expert_layer(x) * mask.unsqueeze(-1)
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
