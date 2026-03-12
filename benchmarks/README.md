# Benchmarks

This directory now has an explicit split between **canonical** and **experimental** benchmark surfaces.

## Canonical

Use these when you want benchmark results that are reasonable to cite inside this repo.

### Throughput / Memory

Use [canonical_benchmark.py](/Users/craigchirara/CGGR-main/benchmarks/canonical_benchmark.py).

Scope:

- synthetic throughput and memory only
- exact config capture
- hardware metadata capture
- includes a random-sequence baseline in addition to standard full-loss training and CGGR

Non-goals:

- no quality claims
- no convergence claims
- no “headline” speedup claims outside the saved artifact

Example:

```bash
python benchmarks/canonical_benchmark.py \
  --device cuda \
  --require-cuda \
  --artifact-tag cuda-single-l40s-throughput \
  --benchmark-tier cuda-canonical \
  --model HuggingFaceTB/SmolLM-135M \
  --batch-sizes 4,8,16 \
  --seq-len 512 \
  --ratio 0.25 \
  --output benchmark_results/cuda_single_l40s_throughput.json
```

### Quality

Use [canonical_quality_benchmark.py](/Users/craigchirara/CGGR-main/benchmarks/canonical_quality_benchmark.py).

Scope:

- dataset-backed train/eval protocol
- held-out full-loss evaluation
- fixed config capture
- includes random-sequence and random-token baselines in addition to standard full-loss training and CGGR

Example:

```bash
python benchmarks/canonical_quality_benchmark.py \
  --device cuda \
  --require-cuda \
  --artifact-tag cuda-single-l40s-quality \
  --benchmark-tier cuda-canonical \
  --model HuggingFaceTB/SmolLM-135M \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --seq-len 256 \
  --batch-size 4 \
  --train-steps 20 \
  --eval-batches 16 \
  --ratio 0.25 \
  --output benchmark_results/cuda_single_l40s_quality.json
```

Reusable single-GPU CUDA entry point:

```bash
bash scripts/run_canonical_l40s.sh
```

Default launcher ratio is `RATIO=0.40` (override with `RATIO=<value>`).

Latest checked-in single-L40S canonical summary:

- [cuda_single_l40s_smollm135_summary.md](/Users/craigchirara/CGGR-main/benchmark_results/cuda_single_l40s_smollm135_summary.md)
- regenerate from artifacts with `python benchmark_results/update_cuda_summary.py`

Notes:

- `--require-cuda` is recommended for research-grade runs so the script fails instead of silently falling back to CPU/MPS.
- Use tagged artifact names to keep local smoke outputs separate from CUDA canonical outputs.
- The current hardening target is single-GPU CUDA evidence first, then multi-GPU scaling later.
- For Slurm jobs, allocate at least `--mem=16G` (prefer `--mem=32G`) for the dataset-backed quality benchmark.
- Current setup-specific recommendation for single `L40S` + `SmolLM-360M`: start with `RATIO=0.40` as the quality/speed default, and treat `0.10` (speed-lean) / `0.60` (quality-lean) as comparison points.

## Experimental

These scripts remain useful for research and iteration, but they are not yet hardened enough to be treated as canonical evidence:

- `benchmark.py`
- `benchmark_suite.py`
- `benchmark_v2.py`
- `benchmark_persistent.py`
- `math_finetune_benchmark.py`
- `code_finetune_benchmark.py`
- `race_runner.py`

## Evaluation Rules

Phase 1 stricter evaluation rules:

1. A CGGR benchmark should not be compared only against full-loss training. Include budget-matched random controls (sequence and/or token level).
2. Synthetic-token benchmarks may support throughput or memory claims only.
3. Quality or convergence claims require:
   - a named dataset
   - fixed seed(s)
   - saved config
   - hardware context
   - a reproducible command
4. If a result is not reproducible from a checked-in command and artifact, it should be treated as exploratory.
