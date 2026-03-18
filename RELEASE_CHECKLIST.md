# Release Checklist

Use this checklist before tagging a release or publishing benchmark claims.

## Core Correctness

- [ ] `pytest -q` passes on a clean environment.
- [ ] `python -m compileall -q cggr.py benchmarks scripts tests setup.py` passes.
- [ ] Public API behavior matches docs for `CGGRLoss`, `CGGRScorer`, `CGGRModel`, and `create_truncated_router`.

## Packaging

- [ ] `python -m build` succeeds.
- [ ] Built wheel installs in a fresh environment.
- [ ] Optional extras (`dev`, `benchmark`, `flash`, `cuda`) resolve expected dependencies.

## Canonical Benchmarks

- [ ] Canonical throughput benchmark artifact exists for current release.
- [ ] Canonical quality benchmark artifact exists for current release.
- [ ] Artifacts include `artifact` and `platform` metadata with hardware context.
- [ ] At least one budget-matched random baseline is present for each claim.
- [ ] Any headline benchmark statement links to a checked-in artifact.

## CUDA Evidence Path

- [ ] CUDA runs use `--require-cuda`.
- [ ] Slurm host memory for quality runs is at least `--mem=16G` (prefer `--mem=32G`).
- [ ] Latest CUDA summary markdown is updated with the latest checked-in artifacts.

## Documentation

- [ ] README and docs only claim behavior validated by tests/artifacts.
- [ ] Experimental scripts/features remain labeled as experimental.
- [ ] Any changed benchmark policy is reflected in `benchmarks/README.md`.
- [ ] Compatibility policy updates are reflected in `COMPATIBILITY.md`.

## CI / Project Hygiene

- [ ] `.github/workflows/ci.yml` passes on current branch.
- [ ] Benchmark smoke CI job generates expected smoke artifacts.
- [ ] Issue templates are present and up to date under `.github/ISSUE_TEMPLATE/`.
