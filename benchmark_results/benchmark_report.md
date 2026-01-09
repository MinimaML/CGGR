# CGGR Benchmark Report

**Model:** HuggingFaceTB/SmolLM-135M
**GPU:** NVIDIA GeForce RTX 3060 (12.0 GB)  
**Timestamp:** 2026-01-09T16:01:29.956388

---

## Executive Summary

| Metric | Standard | CGGR | Improvement |
|:--|:--|:--|:--|
| **Throughput (TPS)** | 6327 | 8352 | **1.32x** |
| **Max Batch Size** | 18 | 69 | **3.8x** |

---

## Throughput

| Configuration | TPS | Latency (ms) | P95 (ms) | P99 (ms) |
|:--|--:|--:|--:|--:|
| Standard (bs=4) | 6327 | 322.4 | 330.5 | 356.6 |
| CGGR (bs=4) | 8352 | 244.2 | 292.4 | 319.3 |
| Standard (bs=8) | 7060 | 579.1 | 572.0 | 1001.9 |
| CGGR (bs=8) | 16498 | 247.2 | 285.3 | 301.2 |
| Standard (bs=16) | 1745 | 4692.7 | 5362.9 | 5527.1 |
| CGGR (bs=16) | 19579 | 417.3 | 421.6 | 487.5 |

---

## Memory

| Configuration | Peak VRAM (MB) | Allocated (MB) |
|:--|--:|--:|
| Standard (bs=4) | 5936 | 1387 |
| CGGR (bs=4) | 2525 | 1095 |
| Standard (bs=8) | 10589 | 1667 |
| CGGR (bs=8) | 3870 | 1095 |
| Standard (bs=16) | 19753 | 2229 |
| CGGR (bs=16) | 6558 | 1095 |

---

## Quality (Convergence)

| Configuration | Final Loss | PPL |
|:--|--:|--:|
| Standard | 13.3899 | 653376.50 |
| CGGR | 11.8403 | 138737.73 |

---

## Tier 1 Optimizations

| Optimization | Result |
|:--|:--|
| Checkpointing VRAM Savings | **-15.2%** |
| DataLoader Skip Rate | **0.0%** |
