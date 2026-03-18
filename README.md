# CGGR - Confidence-Gated Gradient Routing

> [!WARNING]
> CGGR is still experimental. The core selective-loss path is usable for research and prototyping, but large parts of the repo remain exploratory.

Selective loss computation for Transformer-style training. The core idea is to score examples by difficulty, then spend backward-pass work on the hardest part of the batch.

## Status

### Stable Core

The most credible, currently maintained surface is:

- `CGGRLoss`
- `CGGRModel`
- `CGGRScorer`
- `create_truncated_router`
- PyTorch fallback kernels in `triton_kernels.py`

### Experimental / Best-Effort

The following modules exist, but should be treated as research code rather than stable product surface:

- `cggr_async.py`
- `cggr_checkpointing.py`
- `cggr_dataloader.py`
- `cggr_flash.py`
- `persistent_cggr_kernels.py`
- benchmark and race scripts under `benchmarks/`

### Current Behavior Notes

- `CGGRLoss` selects tokens for loss computation.
- `CGGRModel` currently scores token difficulty but performs its second training pass on the hardest sequences, not arbitrary token slices.
- Architecture auto-detection works best on common Hugging Face causal LM layouts. Non-causal and less common architectures should be treated as best-effort until they are covered by dedicated tests.

## Installation

```bash
pip install cggr
```

For optional CUDA acceleration with Triton kernels:

```bash
pip install cggr[cuda]
```

## Platform Compatibility

| Platform            | Triton Kernels | PyTorch Fallback | Status |
| ------------------- | :------------: | :--------------: | ------ |
| CUDA (Linux)        |       ✓        |        ✓         | Best-supported path |
| CUDA (Windows)      |       ✓        |        ✓         | Best-effort |
| ROCm (AMD)          |       ✗        |        ✓         | Fallback-only |
| MPS (Apple Silicon) |       ✗        |        ✓         | Fallback-only |
| CPU                 |       ✗        |        ✓         | Fallback-only |

## Model Architecture Support

| Architecture                   | Auto-Detect | Notes |
| ------------------------------ | :---------: | ----- |
| Llama/Mistral/Qwen/Gemma/Phi-3 |      ✓      | Primary target, `model.layers` style |
| GPT-2/GPT-J/Falcon/GPT-NeoX    |      ✓      | Implemented, `transformer.h` style |
| BERT/RoBERTa                   | Best-effort | Not yet covered by dedicated tests |
| Mamba/SSM                      | Best-effort | Not yet covered by dedicated tests |
| Other                          | Passthrough | Uses full model as router, usually slower |

## Flash Attention Utilities

The repo includes experimental helpers for enabling Flash Attention / SDPA on Hugging Face models:

```bash
pip install cggr[flash]
```

```python
from cggr_flash import load_model_with_flash_attention, enable_flash_attention

# Option 1: Load model with Flash Attention
model = load_model_with_flash_attention("microsoft/phi-2")

# Option 2: Enable on existing model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("...")
model = enable_flash_attention(model)  # Auto-selects best backend
```

| Backend             | Requirements       | Notes |
| ------------------- | ------------------ | ----- |
| `flash_attention_2` | flash-attn library | Experimental helper path |
| `sdpa`              | PyTorch 2.0+       | Experimental helper path |
| `eager`             | None               | Baseline |

## What CGGR Tries To Improve

| Metric            | Standard Training | CGGR Goal | Caveat |
| :---------------- | :---------------- | :-------- | :----- |
| Backward compute  | Full batch/tokens | Selective loss on hardest part of batch | Savings depend on wrapper and routing ratio |
| Forward overhead  | Single pass       | Router pass + selective main pass | Can erase gains on some models/hardware |
| Memory use        | Full graph        | Smaller active backward graph | Most visible with `CGGRModel` |
| Throughput        | Baseline          | Potential improvement | Must be benchmarked per setup |
| Quality           | Baseline          | Similar or better with good routing | Not yet established as a universal result |

## Benchmark Status

Checked-in benchmark artifacts are exploratory, not canonical proof of the method. They are useful for local investigation, but they should not be read as a production benchmark suite yet.

The current repository contains:

- one canonical synthetic throughput/memory entry point: `benchmarks/canonical_benchmark.py`
- one canonical dataset-backed quality entry point: `benchmarks/canonical_quality_benchmark.py`
- one reusable single-GPU CUDA wrapper: `scripts/run_canonical_l40s.sh`
- additional exploratory benchmark scripts under `benchmarks/`
- one checked-in benchmark report under `benchmark_results/`
- early experiments across throughput, memory, and convergence behavior

Phase 0 hardening rule: treat benchmark claims as setup-specific unless they are backed by a reproducible command, fixed config, and matching artifact in the repo.

Current hardening benchmark convention:

- local MPS/CPU outputs are pipeline-validation artifacts only
- CUDA outputs with explicit artifact tags are the canonical evidence path for this repo
- dataset-backed canonical quality runs need meaningful host RAM in Slurm (`--mem=16G` minimum, `--mem=32G` recommended)

See `benchmarks/README.md` for the canonical vs experimental split.
The latest checked-in single-L40S canonical interpretation is in `benchmark_results/cuda_single_l40s_smollm135_summary.md`.
Release gating checklist is in `RELEASE_CHECKLIST.md`.
Compatibility policy is in `COMPATIBILITY.md`.

## Quick Start

### 1. Batch Splitting (Recommended Core Path)

The main end-to-end wrapper is `CGGRModel`. It uses a lightweight router to score difficulty, then trains on the hardest sequences in the batch.

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
| `loss`     | Intended to use per-token loss | Experimental semantics |
| `combined` | Mix of uncertainty signals | Default research setting |

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

## Performance Notes

For `CGGRLoss`, selective loss reduces how many tokens contribute to the final loss tensor, but it does not automatically guarantee end-to-end wall-clock speedup.

For `CGGRModel`, the intended win comes from training the main model on a smaller subset of the batch after a router pass. Whether that produces a net gain depends on:

- model architecture
- batch shape
- routing ratio
- GPU characteristics
- router cost vs main-model savings

## Persistent CGGR Kernels (Advanced, Experimental)

For Token-Routed MLP and Mixture-of-Experts architectures, the repo includes experimental persistent-kernel work:

1. **Persistent Kernels** - Keep SM threads active across expert batches
2. **Cooperative Thread Groups** - Better SM utilization through cooperative scheduling
3. **Warp-Specialized Streaming** - Overlap memory loads with computation
4. **Software Pipelining** - Multi-stage async prefetching for latency hiding

```python
from cggr import PersistentTRMLP, create_persistent_tr_mlp

# Create MoE layer with persistent optimizations
moe_layer = create_persistent_tr_mlp(
    hidden_dim=1024,
    intermediate_dim=4096,
    num_experts=8,
    top_k=2,
)

# Forward pass (routes tokens to experts)
output, aux_loss = moe_layer(hidden_states)
total_loss = language_modeling_loss + 0.01 * aux_loss
```

This code path is source-tree experimental work. It is not part of the stable Phase 0 surface and should be validated independently before use.
