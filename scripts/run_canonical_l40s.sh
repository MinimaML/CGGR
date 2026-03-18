#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolLM-135M}"
DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-2-raw-v1}"
ARTIFACT_PREFIX="${ARTIFACT_PREFIX:-cuda_single_l40s}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results}"
RATIO="${RATIO:-0.40}"
MIN_MEM_MB="${MIN_MEM_MB:-16000}"

mkdir -p "${OUTPUT_DIR}"

if ! command -v python >/dev/null 2>&1; then
  echo "python is not available in PATH. Activate your environment first." >&2
  exit 1
fi

python - <<'PY'
import sys
missing = []
for pkg in ("torch", "transformers", "datasets"):
    try:
        __import__(pkg)
    except Exception:
        missing.append(pkg)
if missing:
    raise SystemExit(
        "Missing required packages for canonical benchmark run: "
        + ", ".join(missing)
        + ". Install benchmark dependencies in the active environment."
    )
PY

to_mb() {
  local raw="$1"
  case "${raw}" in
    *[Gg]) echo $(( ${raw%[Gg]} * 1024 )) ;;
    *[Mm]) echo $(( ${raw%[Mm]} )) ;;
    *[Kk]) echo $(( ${raw%[Kk]} / 1024 )) ;;
    *) echo "${raw}" ;;
  esac
}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  job_info="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | tr '\n' ' ')"
  min_mem_node="$(printf '%s' "${job_info}" | sed -n 's/.*MinMemoryNode=\([^ ]*\).*/\1/p')"
  if [[ -n "${min_mem_node}" ]]; then
    alloc_mem_mb="$(to_mb "${min_mem_node}")"
    if [[ "${alloc_mem_mb}" -lt "${MIN_MEM_MB}" ]]; then
      echo "Refusing to run canonical quality benchmark: Slurm MinMemoryNode=${min_mem_node} (< ${MIN_MEM_MB}M)." >&2
      echo "Request a larger allocation, e.g. --mem=16G or --mem=32G, then retry." >&2
      exit 1
    fi
  fi
fi

echo "Running canonical throughput benchmark on a single CUDA-visible GPU"
python benchmarks/canonical_benchmark.py \
  --device cuda \
  --require-cuda \
  --artifact-tag "${ARTIFACT_PREFIX}_throughput" \
  --benchmark-tier cuda-canonical \
  --model "${MODEL_NAME}" \
  --batch-sizes 4,8,16 \
  --seq-len 512 \
  --ratio "${RATIO}" \
  --warmup-runs 5 \
  --timed-runs 25 \
  --output "${OUTPUT_DIR}/${ARTIFACT_PREFIX}_throughput.json"

echo "Running canonical quality benchmark on a single CUDA-visible GPU"
python benchmarks/canonical_quality_benchmark.py \
  --device cuda \
  --require-cuda \
  --artifact-tag "${ARTIFACT_PREFIX}_quality" \
  --benchmark-tier cuda-canonical \
  --model "${MODEL_NAME}" \
  --dataset "${DATASET_NAME}" \
  --dataset-config "${DATASET_CONFIG}" \
  --seq-len 256 \
  --batch-size 4 \
  --train-steps 20 \
  --eval-batches 16 \
  --ratio "${RATIO}" \
  --lr 5e-5 \
  --max-train-texts 2000 \
  --max-eval-texts 1000 \
  --output "${OUTPUT_DIR}/${ARTIFACT_PREFIX}_quality.json"

echo "Wrote artifacts to ${OUTPUT_DIR}"
