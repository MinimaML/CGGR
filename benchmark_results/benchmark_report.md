# CGGR Benchmark Report

**Model:** HuggingFaceTB/SmolLM-135M
**GPU:** NVIDIA GeForce RTX 3060 (12.0 GB)  
**Timestamp:** 2026-01-09T16:41:31.866738

---

## Executive Summary

| Metric | Standard | CGGR | Improvement |
|:--|:--|:--|:--|
| **Throughput (TPS)** | 6763 | 8694 | **1.29x** |
| **Max Batch Size** | 18 | 69 | **3.8x** |

---

## Throughput

| Configuration | TPS | Latency (ms) | P95 (ms) | P99 (ms) |
|:--|--:|--:|--:|--:|
| Standard (bs=4) | 6763 | 301.7 | 306.6 | 306.8 |
| CGGR (bs=4) | 8694 | 234.6 | 287.8 | 292.8 |
| Standard (bs=8) | 7045 | 580.3 | 574.9 | 985.1 |
| CGGR (bs=8) | 16564 | 246.3 | 269.8 | 309.5 |
| Standard (bs=16) | 1759 | 4655.8 | 5275.7 | 5538.2 |
| CGGR (bs=16) | 19554 | 417.8 | 421.7 | 529.7 |

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
| Standard | 13.4612 | 701655.50 |
| CGGR | 11.7510 | 126876.68 |

---

## Tier 1 Optimizations

| Optimization | Result |
|:--|:--|
| Checkpointing VRAM Savings | **-15.2%** |
| DataLoader Skip Rate | **0.0%** |
