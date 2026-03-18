# Compatibility Policy

This document defines what is considered supported for the hardened CGGR surface.

## Stable Surface

The following APIs are the stable target:

- `CGGRLoss`
- `CGGRScorer`
- `CGGRModel`
- `create_truncated_router`

Everything else should be treated as experimental unless explicitly promoted.

## Python Support

- Supported: `3.10`, `3.11`, `3.12`
- Best effort: `3.9`
- Not guaranteed by CI: `3.8`, `3.13+`

## Device Support

- Canonical evidence path: CUDA (`--require-cuda`)
- Local smoke path: CPU / MPS
- Triton acceleration requires CUDA-compatible environments

## Model Family Support

### Supported for canonical causal-LM benchmarks

- Hugging Face causal LMs with standard logits outputs

### Best effort / partially validated

- BERT/RoBERTa-style masked LM layouts via router adaptation
- Architecture auto-detection fallback paths

## Operational Requirements

- Dataset-backed canonical quality runs require meaningful host memory:
  - minimum: `--mem=16G`
  - recommended: `--mem=32G`
- Use `/scratch` or equivalent job-local working area on HPC systems.

## Benchmark Claim Policy

- Throughput-only synthetic runs may support throughput/memory claims only.
- Quality/convergence claims require dataset-backed canonical artifacts.
- Any claim should point to:
  - exact command
  - checked-in artifact(s)
  - hardware context
  - baseline controls
