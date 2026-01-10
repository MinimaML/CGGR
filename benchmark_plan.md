# 3-Hour Efficiency Race: CGGR vs. Standard Baseline

## Objective
To benchmark the efficiency and performance of **CGGR (Confidence-Gated Gradient Routing)** against a **Standard Fine-Tuning** baseline. The goal is to evaluate learning speed, saturation points, and downstream performance metrics (AIME) under a fixed time constraint.

## Experimental Setup

### Core Constraints
*   **Total Duration:** 3 Hours (1.5 Hours per Run)
*   **Hardware:** Single NVIDIA B200 (Ensures massive throughput and capacity for Full Fine-Tuning)
*   **Model:** `google/gemma-3-4b-it`
*   **Dataset:** `AI-MO/NuminaMath-1.5`
*   **Optimizer:** Muon (Momentumized User-Defined Optimizer).

### Conditions

#### 1. Baseline: Standard Fine-Tuning (Control)
*   **Method:** Full Parameter Fine-Tuning.
*   **Batch Size:** $B$ (Maximum fitable into VRAM).
*   **Gradient Accumulation:** Adjusted to match target logical batch size if VRAM is limited.
*   **Compute Budget:** 1.5 Hours Wall-clock.

#### 2. Experimental: CGGR Accelerated
*   **Method:** CGGR (Confidence-Gated Gradient Routing).
*   **Mechanism:** utilizes VRAM savings from selective gradient computation to scale batch size.
*   **Batch Size:** $\approx 4 \times B$ (Leveraging VRAM overhead reduction).
*   **Compute Budget:** 1.5 Hours Wall-clock.

### 3. Controls & Fairness
*   **Seed Control:** Fixed random seed (e.g., `42`) for data shuffling to ensure identical data order.
*   **Learning Rate Scaling:** LR will be scaled (e.g., Linear or Sqrt rule) to account for the $\approx 4\times$ larger batch size in the CGGR run (`LR_cggr = LR_base * sqrt(4)` or similar).
*   **Warmup Strategy:** To ensure fairness in optimization dynamics:
    *   **Baseline:** Linear warmup for the first **200 steps** (or approx. 5% of training).
    *   **CGGR:** Linear warmup for **50 steps** (Scaling by 4x batch size to match token count).
    *   *Constraint:* Both runs must see the same amount of data (tokens) during the warmup phase to strictly isolate the efficiency of the main training phase.

## Experimental Protocol

### 1. Preparation & calibration
Before the clock starts, we must determine $B$ (max batch size for Baseline).
1.  Run Baseline training loop. Increase micro-batch size until OOM. Set $B_{micro}$.
2.  Run CGGR training loop. Increase micro-batch size until OOM. Verify $\approx 4 \times B_{micro}$ capacity.
3.  Fix hyperparameters (Learning Rate, etc.) appropriately. *Note: With Muon, ensure parameters are scaled if batch size affects update dynamics, though Muon is often robust.*

### 2. The Race (3 Hours Each)

#### Run A: Baseline
*   Start Time: $T_0$
*   End Time: $T_0 + 3h$
*   Logging: 
    *   Loss (step-wise)
    *   Tokens/sec
    *   VRAM Usage
    *   Grad Norms (if applicable for Muon)

#### Run B: CGGR
*   Start Time: $T_1$
*   End Time: $T_1 + 3h$
*   Logging:
    *   Same as Baseline
    *   **Router Metrics:** Sparsity %, Router Entropy, Expert/Token routing distribution.

### 4. Telemetry & Logging (Weights & Biases)
**Mandatory: All metrics must be logged to a shared W&B project for real-time comparison.**

**Essential Metrics:**
*   **Loss Dynamics:** Training Loss, Validation Loss, Per-token Loss.
*   **Throughput & Efficiency:** 
    *   `train/tokens_per_sec`
    *   `train/samples_per_sec`
    *   `system/gpu_memory_allocated` (VRAM)
    *   `system/gpu_utilization`
*   **Optimization State:**
    *   `train/learning_rate`
    *   `train/grad_norm` (Global and per-layer if feasible)
*   **CGGR Exclusive:**
    *   `router/entropy`: To monitor routing confidence/collapse.
    *   `router/expert_usage`: Histogram of expert hits.
    *   `router/active_params`: Count of parameters used in backward pass.
    *   `router/difficulty_score_dist`: Distribution of token difficulty scores.

### 5. Evaluation Strategy

To measure "saturation" and "how quickly they learn", we will perform evaluations at fixed time intervals.

**Intervals:**
*   **30-Minute Benchmarks:** Save checkpoint and run AIME benchmark at **T=30m**, **T=60m**, and **T=90m** (Final).
*   **Progress Plotting:** Plot AIME `pass@1` Score vs. Training Time for both runs on the same chart.

**Benchmarks:**
*   **Primary Metric:** AIME (American Invitational Mathematics Examination) Benchmark Suite.
*   **Metrics:** 
    *   `pass@1` Accuracy.
    *   Total valid solutions generated.
*   **Secondary Metric:** 
    *   **GSM8K 8-Shot Accuracy:** To validate basic reasoning capabilities (Baseline: 1.0% -> Current: 6.0%).
    *   Validation Loss on NuminaMath held-out set.

## Analysis Plan

The final report will compare:
1.  **Performance vs. Time:** AIME Score over 3 hours. Does CGGR reach peak performance faster?
2.  **Performance vs. Data:** AIME Score over total tokens seen. Does the 4x Batch Size lead to better sample efficiency per wall-clock second?
3.  **Saturation:** Do the curves flatten? Does CGGR saturate at a higher or lower level than the baseline within the timeframe?
4.  **Efficiency:** 
    *   Total Tokens Processed in 3 hours.
    *   VRAM Peak usage comparison.

## Deliverables
*   `training_log_baseline.json`
*   `training_log_cggr.json`
*   `benchmark_results.md` (Comparative plots and tables)
