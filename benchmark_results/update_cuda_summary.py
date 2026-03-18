#!/usr/bin/env python3
"""Update canonical CUDA benchmark summary markdown from checked-in artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import glob


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmark_results"
SUMMARY_PATH = BENCH_DIR / "cuda_single_l40s_smollm135_summary.md"

ARTIFACTS = {
    "smollm135": {
        "throughput": BENCH_DIR / "cuda_single_l40s_smollm135_v2_throughput.json",
        "quality": BENCH_DIR / "cuda_single_l40s_smollm135_v2_quality.json",
        "label": "SmolLM-135M",
        "prefix": "cuda_single_l40s_smollm135_v2",
    },
    "smollm360": {
        "throughput": BENCH_DIR / "cuda_single_l40s_smollm360_v1_throughput.json",
        "quality": BENCH_DIR / "cuda_single_l40s_smollm360_v1_quality.json",
        "label": "SmolLM-360M",
        "prefix": "cuda_single_l40s_smollm360_v1",
    },
}


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_float(v: float) -> str:
    return f"{v:,.4f}"


def render_model_section(label: str, throughput: dict, quality: dict) -> str:
    t_rows = {f"{r['name']}/b{r['batch_size']}": r for r in throughput["results"]}
    q_rows = {r["name"]: r for r in quality["results"]}

    return f"""## {label}

### Throughput (synthetic, seq_len={throughput["config"]["seq_len"]})

| Mode | Batch 4 TPS | Batch 8 TPS | Batch 16 TPS |
| --- | ---: | ---: | ---: |
| standard | {fmt_float(t_rows["standard/b4"]["tokens_per_second"])} | {fmt_float(t_rows["standard/b8"]["tokens_per_second"])} | {fmt_float(t_rows["standard/b16"]["tokens_per_second"])} |
| random_sequence | {fmt_float(t_rows["random_sequence/b4"]["tokens_per_second"])} | {fmt_float(t_rows["random_sequence/b8"]["tokens_per_second"])} | {fmt_float(t_rows["random_sequence/b16"]["tokens_per_second"])} |
| cggr | {fmt_float(t_rows["cggr/b4"]["tokens_per_second"])} | {fmt_float(t_rows["cggr/b8"]["tokens_per_second"])} | {fmt_float(t_rows["cggr/b16"]["tokens_per_second"])} |

### Quality ({quality["config"]["dataset_name"]}/{quality["config"]["dataset_config"]})

| Mode | Eval Loss | Eval PPL |
| --- | ---: | ---: |
| standard | {fmt_float(q_rows["standard"]["eval_loss"])} | {fmt_float(q_rows["standard"]["eval_ppl"])} |
| random_sequence | {fmt_float(q_rows["random_sequence"]["eval_loss"])} | {fmt_float(q_rows["random_sequence"]["eval_ppl"])} |
| random_token | {fmt_float(q_rows["random_token"]["eval_loss"])} | {fmt_float(q_rows["random_token"]["eval_ppl"])} |
| cggr | {fmt_float(q_rows["cggr"]["eval_loss"])} | {fmt_float(q_rows["cggr"]["eval_ppl"])} |
"""


def render_comparison(sm135_quality: dict, sm360_quality: dict) -> str:
    q135 = {r["name"]: r for r in sm135_quality["results"]}
    q360 = {r["name"]: r for r in sm360_quality["results"]}
    lines = [
        "## 135M vs 360M Comparison",
        "",
        "| Mode | 135M Eval PPL | 360M Eval PPL | Δ (360M-135M) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ["standard", "random_sequence", "random_token", "cggr"]:
        p135 = q135[name]["eval_ppl"]
        p360 = q360[name]["eval_ppl"]
        lines.append(f"| {name} | {fmt_float(p135)} | {fmt_float(p360)} | {fmt_float(p360 - p135)} |")
    lines.append("")
    return "\n".join(lines)


def load_ratio_sweep_rows():
    rows = []
    quality_paths = sorted(glob.glob(str(BENCH_DIR / "cuda_single_l40s_smollm360_ratio*_v2_quality.json")))
    for quality_path_str in quality_paths:
        quality_path = Path(quality_path_str)
        throughput_path = Path(str(quality_path).replace("_quality.json", "_throughput.json"))
        if not throughput_path.exists():
            continue

        quality = load_json(quality_path)
        throughput = load_json(throughput_path)
        if quality is None or throughput is None:
            continue

        ratio = float(quality["config"]["cggr_ratio"])
        cggr_q = [r for r in quality["results"] if r["name"] == "cggr"][0]
        cggr_t_b16 = [
            r for r in throughput["results"] if r["name"] == "cggr" and int(r["batch_size"]) == 16
        ][0]
        rows.append(
            {
                "ratio": ratio,
                "cggr_ppl": float(cggr_q["eval_ppl"]),
                "cggr_eval_loss": float(cggr_q["eval_loss"]),
                "cggr_tps_b16": float(cggr_t_b16["tokens_per_second"]),
                "artifact_quality": quality_path.name,
                "artifact_throughput": throughput_path.name,
            }
        )
    rows.sort(key=lambda r: r["ratio"])
    return rows


def render_ratio_sweep_section(rows):
    lines = [
        "## SmolLM-360M Ratio Sweep (v2)",
        "",
        "These rows come from `cuda_single_l40s_smollm360_ratio*_v2` artifacts.",
        "",
        "| Ratio | CGGR Eval Loss | CGGR Eval PPL | CGGR TPS @ Batch 16 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['ratio']:.2f} | {fmt_float(row['cggr_eval_loss'])} | {fmt_float(row['cggr_ppl'])} | {fmt_float(row['cggr_tps_b16'])} |"
        )
    lines.extend(
        [
            "",
            "Recommended default ratio for this setup: **0.40**",
            "",
            "Rationale: `0.40` gives a much better quality/speed balance than `0.10/0.25`, "
            "while avoiding the steep throughput drop at `0.60`.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    sm135_t = load_json(ARTIFACTS["smollm135"]["throughput"])
    sm135_q = load_json(ARTIFACTS["smollm135"]["quality"])
    sm360_t = load_json(ARTIFACTS["smollm360"]["throughput"])
    sm360_q = load_json(ARTIFACTS["smollm360"]["quality"])

    if sm135_t is None or sm135_q is None:
        raise RuntimeError("Missing required SmolLM-135M v2 artifacts for summary generation.")

    content = [
        "# Canonical CUDA Summary (Single L40S)",
        "",
        "This summary is generated from checked-in canonical artifact JSON files.",
        "",
        render_model_section(ARTIFACTS["smollm135"]["label"], sm135_t, sm135_q),
    ]

    if sm360_t is not None and sm360_q is not None:
        content.append(render_model_section(ARTIFACTS["smollm360"]["label"], sm360_t, sm360_q))
        content.append(render_comparison(sm135_q, sm360_q))
    else:
        content.extend([
            "## 135M vs 360M Comparison",
            "",
            "Pending SmolLM-360M artifacts. Once generated and synced, run:",
            "",
            "```bash",
            "python benchmark_results/update_cuda_summary.py",
            "```",
        ])

    ratio_rows = load_ratio_sweep_rows()
    if ratio_rows:
        content.append(render_ratio_sweep_section(ratio_rows))
    else:
        content.extend([
            "## SmolLM-360M Ratio Sweep (v2)",
            "",
            "Pending `cuda_single_l40s_smollm360_ratio*_v2` artifacts. After syncing them, run:",
            "",
            "```bash",
            "python benchmark_results/update_cuda_summary.py",
            "```",
            "",
            "Target default ratio for this setup is currently **0.40**.",
            "",
        ])

    content.extend([
        "",
        "## Hardening Notes",
        "",
        "- Canonical quality includes sequence-level and token-level random controls.",
        "- Use `--require-cuda` for research-grade runs.",
        "- For Slurm quality runs, use `--mem=16G` minimum and `--mem=32G` preferred.",
    ])

    SUMMARY_PATH.write_text("\n".join(content).strip() + "\n", encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
