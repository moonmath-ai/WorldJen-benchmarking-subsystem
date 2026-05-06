# WorldJen Benchmarking Subsystem

Code for reproducing all experiments in the paper:
*WorldJen: An End-to-End Multi-Dimensional Benchmark for Generative Video Models.*

**Dataset (HuggingFace):** [ik6626/WorldJen-benchmarking-subsystem](https://huggingface.co/datasets/ik6626/WorldJen-benchmarking-subsystem)

**Paper (ArXiv):** [https://arxiv.org/abs/2605.03475](https://arxiv.org/abs/2605.03475)

**Project Page:** [https://moonmath.ai/worldjen/](https://moonmath.ai/worldjen/#paper-demo)

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-org/WorldJen-benchmarking-subsystem
cd WorldJen-benchmarking-subsystem

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Set up API keys
cp .env.example .env
# Edit .env — fill in GEMINI_API_KEY, ANTHROPIC_API_KEY, HF_TOKEN

# 4. Download the dataset from HuggingFace
python download_data.py          # text + pre-computed results (~500 MB)
python download_data.py --include-videos   # also download videos (~3 GB)

# 5. Run the full reproduction pipeline
bash reproduce.sh
```

Results land in `data/results/`. Figures land in `figures/`.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in values.

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes (prompt pipeline + VLM eval) | Google Gemini API key |
| `ANTHROPIC_API_KEY` | Yes (A4 cross-VLM ablation only) | Anthropic Claude API key |
| `HF_TOKEN` | Yes (dataset download) | HuggingFace token to download the dataset |
| `WORLDJEN_DATA_ROOT` | No | Path to dataset root (default: `./data`) |
| `VBENCH_DIR` | VBench eval only | Path to cloned VBench repo |
| `GEMMA4_MODEL_PATH` | Gemma 4 ablation only | Path to Gemma 4 31B model weights |

---
## Repository Structure

```
WorldJen-benchmarking-subsystem/
│
├── reproduce.sh              ← Master reproduction script (run this)
├── download_data.py          ← Download HuggingFace dataset → data/
├── requirements.txt
├── .env.example
│
├── prompt_pipeline/          ← prompt judging & enhancement pipeline
│   ├── config.py
│   ├── run_pipeline.sh
│   └── scripts/
│       ├── gemini_judge.py
│       ├── gemini_fix_review_prompts.py
│       ├── gemini_enhance.py
│       ├── gemini_enhance_pass2.py
│       ├── gemini_judge_enhanced.py
│       ├── generate_statistics.py
│       └── compare_suitability_stats.py
│
├── semantic_adherence/       ← Gemini embedding similarity validation
│   ├── compute_semantic_adherence.py
│   └── generate_report.py
│
├── vlm_eval/                 ← VLM evaluation (Gemini 3 Flash)
│   ├── vqa_generator.py
│   ├── parallel_vlm_evaluator.py
│   └── unified_analyzer.py  ← builds summary_report_unified.json
│
├── system_prompts/           ← System prompts used in Phase A and Phase B
│   ├── phase_a_llm_judge.txt       ← Phase A: suitability + difficulty scorer
│   ├── phase_a_llm_enhance.txt     ← Phase A: prompt enhancer
│   ├── phase_b_vqa_generator.txt   ← Phase B: VQA question generator
│   ├── phase_b_vlm_evaluator.txt   ← Phase B: per-dimension frame scorer
│   └── README.md
│
├── scoring/                  ← Bootstrap BT rating + CI
│   └── bootstrap_bt_rating.py
│
├── vbench/                   ← VBench v1 + v2 evaluation
│   ├── run_single_model.py
│   ├── run_vbench_eval.py
│   ├── run_single_model_vbench2.py
│   └── run_vbench2_eval.py
│
├── human_eval/               ← Human evaluation analysis
│   ├── analyze_bt_evals.py   ← Human BT rating (unweighted)
│   ├── calibrate_phas.py     ← PHAS weight calibration (elastic-net)
│   ├── phas_annotator_stability.py  ← LOAO stability analysis (W3)
│   └── diagnose_composition_weight.py
│
├── figures/                  ← Generate all paper figures
│   ├── generate_figures.py         ← Fig 3 heatmap, Fig 5 bar charts
│   ├── generate_bt_comparison.py   ← Fig 4 BT rating comparison
│   ├── generate_vbench_evidence.py ← Fig 6 VBench evidence
│   ├── generate_vbench_comparison.py
│   ├── generate_gemma4_comparison.py  ← A6 figure
│   ├── generate_dragon_appendix.py    ← Appendix case study
│   ├── generate_assassin_appendix.py  ← Appendix case study
│   └── vbench_bump_chart.py
│
└── ablations/
    ├── a1_prompt_enhancement/   ← A1: prompt enhancement ablation
    │   ├── vqa_generator_ablation_a1.py
    │   ├── vlm_evaluator_ablation_a1.py
    │   └── compare_a1.py
    ├── a2_question_count/       ← A2: question count ablation
    │   └── question_count_ablation.py
    ├── a4_cross_vlm/            ← A4: cross-VLM auditor validation
    │   ├── vlm_evaluator_claude.py
    │   └── analyze_claude_vs_gemini.py
    ├── a5_run_reliability/      ← A5: run reliability analysis
    │   └── compare_runs.py
    └── a6_open_source_vlm/      ← A6: open-source VLM (requires 8× H200)
        ├── download_model.py
        ├── vlm_evaluator_gemma4.py
        └── analyze_gemma4_results.py
```

---

## Data Layout (after `download_data.py`)

```
data/
├── prompts/
│   ├── prompts_50.jsonl                      ← 50 enhanced prompts (paper)
│   ├── prompts_unenhanced_50.jsonl           ← Unenhanced versions
│   ├── prompts_judged_full.jsonl             ← Full scored corpus (~3,754 prompts)
│   ├── prompts_enhanced_full.jsonl           ← Full enhanced corpus
│   ├── prompts_ablation_a1_validation20.jsonl ← 20 unenhanced prompts for A1
│   ├── vqa_questions_50prompts.jsonl         ← VQA questions for all 50 prompts
│   ├── vqa_questions_ablation_a1.jsonl       ← VQA questions for A1 (unenhanced)
│   └── suitability_comparison.csv            ← Enhanced vs unenhanced score diff (A1)
│
├── results/
│   ├── gemini_vlm/        ← 300 JSON files (50 prompts × 6 models, Gemini 3 Flash)
│   ├── gemini_vlm_run2/   ← Run 2 for A5 reliability analysis
│   ├── claude_vlm/        ← 120 JSON files for A4 (Claude Sonnet 4.6)
│   ├── ablation_a1/       ← 120 JSON files for A1 (unenhanced prompts)
│   ├── gemma4_vlm/        ← 300 JSON files for A6 (Gemma 4 31B)
│   └── summaries/
│       ├── summary_report_unified.json               ← BT rating, PHAS, per-dim averages
│       ├── bootstrap_bt_rating_results.json          ← BT rating with 95% CI
│       ├── a1_enhancement_ablation_results.json      ← A1 prompt enhancement summary
│       ├── question_count_ablation_results_unified.json ← A2 question count summary
│       ├── semantic_adherence_results_50.json        ← Gemini embedding cosine similarity
│       ├── a4_ablation_results.json                  ← A4 cross-VLM agreement summary
│       ├── a5_ablation_results.json                  ← A5 run reliability summary
│       └── summary_report_gemma4.json                ← A6 Gemma 4 evaluation summary
│
├── human_eval/
│   ├── anonymized_human_evals.csv    ← 2,696 pairwise comparisons (7 annotators)
│   ├── human_eval_analysis.json      ← PHAS weights, BT ratings, Krippendorff α
│   └── report_bt_anon.txt            ← Full BT analysis report
│
├── vbench/
│   ├── vbench_summary.json           ← Combined VBench v1 + v2 quality scores
│   └── vbench2_evaluation_results/   ← Raw VBench-2.0 per-model JSON files
│
└── videos/                           ← (optional, ~3 GB)
    ├── fal-ai_veo3.1_fast/
    ├── fal-ai_kling-video_v2.6_pro_text-to-video/
    ├── fal-ai_wan_v2.2-a14b_text-to-video/
    ├── fal-ai_ltx-2_text-to-video/
    ├── fal-ai_hunyuan-video-v1.5_text-to-video/
    └── wan2.1-1.3b/
```

> Videos are required only for VLM eval, VBench, Gemma 4 ablation, and the appendix case-study figures.

---

## Reproduction Pipeline

Run `bash reproduce.sh` to execute the full pipeline. Each step checks for its output and skips if already present. Use `--force` to regenerate everything.

| Script | Output (under `data/`) | Paper section |
|---|---|---|
| `download_data.py` | `data/` populated | — |
| `vlm_eval/parallel_vlm_evaluator.py` | `results/gemini_vlm/*.json` | §6 |
| `vlm_eval/unified_analyzer.py` | `results/summaries/summary_report_unified.json` | §6 |
| `scoring/bootstrap_bt_rating.py` | `results/summaries/bootstrap_bt_rating_results.json` | §6 |
| `human_eval/analyze_bt_evals.py` | `human_eval/report_bt_anon.txt` | §5, §6.3 |
| `semantic_adherence/compute_semantic_adherence.py` | `results/summaries/semantic_adherence_results_50.json` | §6.4 |
| `human_eval/calibrate_phas.py` | `human_eval/human_eval_analysis.json` | §7 |
| `ablations/a4_cross_vlm/vlm_evaluator_claude.py` + `analyze_claude_vs_gemini.py` | `results/summaries/a4_ablation_results.json` | §8 |
| `ablations/a5_run_reliability/compare_runs.py` | `results/summaries/a5_ablation_results.json` | §8 |
| `ablations/a6_open_source_vlm/vlm_evaluator_gemma4.py` + `analyze_gemma4_results.py` | `results/summaries/summary_report_gemma4.json` | §8 |
| `vbench/run_vbench_eval.py` | `vbench/vbench_summary.json` | §9 |
| `ablations/a1_prompt_enhancement/vlm_evaluator_ablation_a1.py` + `compare_a1.py` | `results/summaries/a1_enhancement_ablation_results.json` | Appendix |
| `ablations/a2_question_count/question_count_ablation.py` | `results/summaries/question_count_ablation_results_unified.json` | Appendix |
| `figures/generate_figures.py` + others | `figures/*.png` | all |

---

## VBench Setup

VBench is not pip-installable. Install it from source:

```bash
git clone https://github.com/Vchitect/VBench
cd VBench && pip install -r requirements.txt
export VBENCH_DIR=/path/to/VBench   # add to .env
```

For VBench-2.0 (Human Anatomy dimension), the submodule is inside the same repo at `VBench/VBench-2.0`.

---

## Gemma 4 Open-Source VLM Ablation (A6)

Requires 8× NVIDIA H200 GPUs and ~63 GB of disk for model weights.

```bash
# Download model weights once (~63 GB) → saved to data/models/gemma-4-31b-it
python ablations/a6_open_source_vlm/download_model.py

# Tell the evaluator where the weights are
export GEMMA4_MODEL_PATH="${WORLDJEN_DATA_ROOT:-$(pwd)/data}/models/gemma-4-31b-it"  # or add to .env

# Run evaluation
python ablations/a6_open_source_vlm/vlm_evaluator_gemma4.py

# Analyze results
python ablations/a6_open_source_vlm/analyze_gemma4_results.py
```

---

## Model Index

| Paper name | API / model ID |
|---|---|
| Gemini 3 Flash | `gemini-3-flash-preview` |
| Gemini 3.1 Flash-Lite | `gemini-3.1-flash-lite-preview` |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` |
| Gemma 4 31B (A6) | `gemma-4-31b-it` |
| Gemini 2.5 Pro (prompt pipeline) | `gemini-2.5-pro` |

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@misc{inbasekar2026worldjen,
  title         = {WorldJen: An End-to-End Multi-Dimensional Benchmark for Generative Video Models},
  author        = {Karthik Inbasekar and Guy Rom and Omer Shlomovits},
  year          = {2026},
  eprint        = {2605.03475},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2605.03475},
}
```

---

## License

Code: MIT License. See [LICENSE](LICENSE).
Dataset: CC BY 4.0.
