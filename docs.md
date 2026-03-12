# CGGR Documentation

> [!WARNING]
> This document describes the current repository state after Phase 0 hardening. Where the research vision and the implementation differ, the implementation takes precedence.

## Overview

CGGR is an experimental training-time technique for spending backward-pass work on the hardest part of a batch instead of all tokens equally.

The repo currently has two main usage paths:

- `CGGRLoss`: selects tokens for loss computation
- `CGGRModel`: scores difficulty first, then trains on the hardest sequences in the batch

The core stable-ish surface for further hardening is:

- `cggr.py`
- `triton_kernels.py`

Everything else should be treated as experimental until it has matching tests and reproducible benchmarks.

## Installation

Core install:

```bash
pip install cggr
```

Optional CUDA acceleration:

```bash
pip install cggr[cuda]
```

Optional Hugging Face helper utilities:

```bash
pip install cggr[flash]
```

## Support Status

| Area | Status | Notes |
| ---- | ------ | ----- |
| Core selective loss | Primary focus | `CGGRLoss` is the clearest implementation |
| Batch-splitting wrapper | Usable, needs hardening | `CGGRModel` trains on hard sequences in pass 2 |
| Triton acceleration | Optional | Falls back to PyTorch on unsupported devices |
| CUDA/Linux | Best-supported path | Most likely to reflect intended performance |
| CPU / MPS / ROCm | Fallback-only | Functional path, limited validation |
| Async / checkpointing / dataloader helpers | Experimental | Research code |
| Flash Attention helpers | Experimental | Convenience utilities, not core API |
| Persistent kernels / MoE work | Experimental | Validate separately before use |

## Current Behavior

### `CGGRLoss`

`CGGRLoss` computes difficulty scores, selects a subset of tokens, and computes cross-entropy only on those selected tokens.

Supported public modes:

- `scoring`: `entropy`, `margin`, `loss`, `combined`
- `selection`: `topk`, `stratified`, `sequence_aware`, `fixed_quota`

Phase 0 caveat:

- `loss` and `combined` should still be treated as research semantics, because target-conditioned routing is not yet consistently represented across all wrappers.

### `CGGRModel`

`CGGRModel` does a two-pass training flow:

1. router/scorer estimates difficulty
2. main model trains on the hardest sequences in the batch

This means the current implementation is not arbitrary token-sliced training in the main wrapper. It is sequence-level selection in the second pass.

### `CGGRScorer`

`CGGRScorer` exposes a broader API than the fully hardened behavior. Use it for research and custom integrations, but assume non-default selection semantics outside `CGGRLoss` still need verification.

### `TruncatedRouter`

The router auto-detection path is most credible on common Hugging Face causal LM layouts:

- `model.layers`
- `transformer.h`

Other architectures are best-effort until covered by dedicated tests.

## Guidance

### Best current entry point

Use `CGGRLoss` if you want the clearest, most inspectable implementation.

Use `CGGRModel` if you specifically want the two-pass wrapper and accept that it is still being hardened.

### What CGGR may improve

- active backward work
- peak memory during training
- throughput on some hardware / batch shapes

### What CGGR does not guarantee

- universal speedup
- universal quality improvement
- support for every claimed architecture in the docs history
- reproducible benchmark wins across setups

## Benchmarks

The checked-in benchmark scripts and artifacts are exploratory. They are useful for local investigation, but they are not yet a canonical benchmark suite.

Phase 0 hardening rule:

- do not make headline claims from a benchmark unless the repo contains
  - the exact command
  - the config
  - the hardware context
  - the generated artifact

The canonical benchmark entry points are now:

- `benchmarks/canonical_benchmark.py` for synthetic throughput/memory
- `benchmarks/canonical_quality_benchmark.py` for dataset-backed quality evaluation
- `scripts/run_canonical_l40s.sh` for single-GPU CUDA canonical runs

Other scripts under `benchmarks/` remain experimental until they meet the same standard.

Current benchmark artifact rule:

- local CPU/MPS artifacts are useful to validate the pipeline
- canonical research artifacts should be tagged CUDA runs with explicit hardware metadata and `--require-cuda`
- dataset-backed canonical quality runs should use Slurm host memory of at least `--mem=16G` (prefer `--mem=32G`)

Current single-L40S canonical interpretation is tracked in:

- `benchmark_results/cuda_single_l40s_smollm135_summary.md`
- regenerate via `python benchmark_results/update_cuda_summary.py` after syncing new artifacts

## Stable vs Experimental Roadmap Boundary

### Stable target for Phase 1-3 hardening

- `CGGRLoss`
- `CGGRModel`
- `CGGRScorer`
- `create_truncated_router`
- PyTorch fallback path in `triton_kernels.py`

### Explicitly experimental for now

- `cggr_async.py`
- `cggr_checkpointing.py`
- `cggr_dataloader.py`
- `cggr_flash.py`
- `persistent_cggr_kernels.py`
- most scripts under `benchmarks/`

CI and release gates:

- GitHub Actions workflow: `.github/workflows/ci.yml`
- Release checklist: `RELEASE_CHECKLIST.md`
- Compatibility policy: `COMPATIBILITY.md`

## Known Phase 0 Gaps

- Some public API descriptions are broader than the fully verified behavior.
- Benchmark methodology is not yet strict enough for strong claims.
- Auxiliary modules exist before the stable surface is fully hardened.
- Packaging and extras still need a separate hardening pass.

## Next Hardening Step

Phase 1 should focus on core correctness:

- align API semantics with implementation
- harden `CGGRScorer` and `CGGRModel`
- add missing tests for real wrapper behavior
