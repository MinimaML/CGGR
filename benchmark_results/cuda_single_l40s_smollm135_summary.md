# Canonical CUDA Summary (Single L40S)

This summary is generated from checked-in canonical artifact JSON files.

## SmolLM-135M

### Throughput (synthetic, seq_len=512)

| Mode | Batch 4 TPS | Batch 8 TPS | Batch 16 TPS |
| --- | ---: | ---: | ---: |
| standard | 28,693.9703 | 29,476.4684 | 25,514.0124 |
| random_sequence | 32,261.5351 | 62,847.5801 | 126,923.9702 |
| cggr | 23,588.7608 | 25,615.6089 | 22,756.6622 |

### Quality (wikitext/wikitext-2-raw-v1)

| Mode | Eval Loss | Eval PPL |
| --- | ---: | ---: |
| standard | 2.8458 | 17.2150 |
| random_sequence | 2.8858 | 17.9174 |
| random_token | 2.8855 | 17.9120 |
| cggr | 2.9002 | 18.1782 |

## SmolLM-360M

### Throughput (synthetic, seq_len=512)

| Mode | Batch 4 TPS | Batch 8 TPS | Batch 16 TPS |
| --- | ---: | ---: | ---: |
| standard | 16,246.3575 | 14,038.8330 | 12,734.0821 |
| random_sequence | 29,971.4932 | 55,192.0861 | 63,419.8047 |
| cggr | 14,535.4221 | 12,863.4226 | 11,711.1498 |

### Quality (wikitext/wikitext-2-raw-v1)

| Mode | Eval Loss | Eval PPL |
| --- | ---: | ---: |
| standard | 2.5879 | 13.3023 |
| random_sequence | 2.6410 | 14.0274 |
| random_token | 2.6301 | 13.8750 |
| cggr | 2.6627 | 14.3353 |

## 135M vs 360M Comparison

| Mode | 135M Eval PPL | 360M Eval PPL | Δ (360M-135M) |
| --- | ---: | ---: | ---: |
| standard | 17.2150 | 13.3023 | -3.9127 |
| random_sequence | 17.9174 | 14.0274 | -3.8900 |
| random_token | 17.9120 | 13.8750 | -4.0370 |
| cggr | 18.1782 | 14.3353 | -3.8429 |

## SmolLM-360M Ratio Sweep (v2)

These rows come from `cuda_single_l40s_smollm360_ratio*_v2` artifacts.

| Ratio | CGGR Eval Loss | CGGR Eval PPL | CGGR TPS @ Batch 16 |
| ---: | ---: | ---: | ---: |
| 0.10 | 2.7088 | 15.0114 | 74,921.5318 |
| 0.25 | 2.7088 | 15.0114 | 46,916.8061 |
| 0.40 | 2.6730 | 14.4828 | 26,967.9005 |
| 0.60 | 2.6635 | 14.3462 | 18,808.4298 |

Recommended default ratio for this setup: **0.40**

Rationale: `0.40` gives a much better quality/speed balance than `0.10/0.25`, while avoiding the steep throughput drop at `0.60`.


## Hardening Notes

- Canonical quality includes sequence-level and token-level random controls.
- Use `--require-cuda` for research-grade runs.
- For Slurm quality runs, use `--mem=16G` minimum and `--mem=32G` preferred.
