# CGGR Benchmark Report

**Model:** HuggingFaceTB/SmolLM-135M
**GPU:** NVIDIA GeForce RTX 3060 (12.0 GB)  
**Timestamp:** 2026-01-09T15:45:09.104103

---

## Executive Summary

| Metric | Standard | CGGR | Improvement |
|:--|:--|:--|:--|
| **Throughput (TPS)** | 6482 | 8077 | **1.25x** |
| **Max Batch Size** | 18 | 69 | **3.8x** |

---

## Throughput

| Configuration | TPS | Latency (ms) | P95 (ms) | P99 (ms) |
|:--|--:|--:|--:|--:|
| Standard (bs=4) | 6482 | 314.7 | 323.3 | 332.3 |
| CGGR (bs=4) | 8077 | 252.5 | 295.1 | 304.5 |
| Standard (bs=8) | 6087 | 671.7 | 754.1 | 761.8 |
| CGGR (bs=8) | 15077 | 270.5 | 298.8 | 318.2 |
| Standard (bs=16) | 1478 | 5540.4 | 6228.6 | 6274.8 |
| CGGR (bs=16) | 19222 | 424.9 | 444.3 | 477.4 |

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
| Standard | 13.3914 | 654376.06 |
| CGGR | 11.7896 | 131873.38 |

---

## Tier 1 Optimizations

| Optimization | Result |
|:--|:--|
| Checkpointing VRAM Savings | **-112.0%** |
| DataLoader Skip Rate | **0.0%** |
