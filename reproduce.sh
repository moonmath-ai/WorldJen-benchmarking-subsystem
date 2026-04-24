#!/usr/bin/env bash
# reproduce.sh — WorldJen Benchmarking System: end-to-end reproduction script
#
# Reproduces all paper results from a downloaded dataset.
# Each step checks whether its primary output exists and skips if so.
# Re-run with --force to regenerate everything from scratch.
#
# Steps follow the paper section order:
#   §3  Prompt pipeline (Phase A curation)
#   §5  Human preference study
#   §6  VLM evaluation
#   §7  PHAS calibration
#   §8  Ablation studies (A1–A6)
#   §9  VBench comparison
#
# Usage:
#   export WORLDJEN_DATA_ROOT=/path/to/data   # where HF dataset was downloaded
#   bash reproduce.sh                          # run all steps
#   bash reproduce.sh --step vlm_eval         # run only one step by name
#   bash reproduce.sh --force                 # skip nothing; regenerate all
#
# Prerequisites:
#   pip install -r requirements.txt
#   cp .env.example .env && nano .env          # fill in your API keys
#   python download_data.py                    # download HF dataset → DATA_ROOT

set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
FORCE=0
ONLY_STEP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --step) ONLY_STEP="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Environment ───────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${WORLDJEN_DATA_ROOT:-${REPO_DIR}/data}"
export WORLDJEN_DATA_ROOT="${DATA_ROOT}"

# Load .env if present
if [[ -f "${REPO_DIR}/.env" ]]; then
    set -a; source "${REPO_DIR}/.env"; set +a
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

skip_or_run() {
    local name="$1"; local sentinel="$2"; local cmd="${@:3}"
    [[ -n "${ONLY_STEP}" && "${ONLY_STEP}" != "${name}" ]] && return 0
    if [[ "${FORCE}" -eq 0 && -e "${DATA_ROOT}/${sentinel}" ]]; then
        log "${name}: SKIP (${sentinel} exists). Use --force to regenerate."
        return 0
    fi
    log "${name}: running..."
    eval "${cmd}"
    log "${name}: done."
}

# ── Download dataset ──────────────────────────────────────────────────────────
skip_or_run download "prompts/prompts_50.jsonl" \
    "cd '${REPO_DIR}' && python download_data.py"

# ── §3 Prompt pipeline (optional — prompts already in dataset) ────────────────
# Regenerate enhanced prompts from raw seeds.
# Skipped by default since prompts/prompts_50.jsonl is already in the dataset.
# skip_or_run prompt_pipeline "prompts/prompts_judged_full.jsonl" \
#     "cd '${REPO_DIR}/prompt_pipeline' && bash run_pipeline.sh"

# ── §6 VLM evaluation (Gemini 3 Flash) ───────────────────────────────────────
# Calls Gemini API; runs in parallel across prompts.
# Output: DATA_ROOT/results/gemini_vlm/<model>_<prompt>.json (300 files)
skip_or_run vlm_eval "results/gemini_vlm" \
    "cd '${REPO_DIR}/vlm_eval' && python parallel_vlm_evaluator.py"

# ── §6 Build unified summary ──────────────────────────────────────────────────
skip_or_run vlm_summary "results/summaries/summary_report_unified.json" \
    "cd '${REPO_DIR}/vlm_eval' && python unified_analyzer.py"

# ── §6 Bootstrap BT rating CI ─────────────────────────────────────────────────
skip_or_run bootstrap_bt "results/summaries/bootstrap_bt_rating_results.json" \
    "cd '${REPO_DIR}/scoring' && python bootstrap_bt_rating.py"

# ── §5+§6 Human BT ratings + VLM vs human rank correlation ───────────────────
# Requires VLM summary (for rank correlation in §6.3); run after vlm_summary.
skip_or_run human_eval_bt "human_eval/report_bt_anon.txt" \
    "cd '${REPO_DIR}/human_eval' && python analyze_bt_evals.py --out '${DATA_ROOT}/human_eval/report_bt_anon.txt'"

# ── §6.4 Semantic adherence cross-method validation ───────────────────────────
skip_or_run semantic_adherence "results/summaries/semantic_adherence_results_50.json" \
    "cd '${REPO_DIR}/semantic_adherence' && python compute_semantic_adherence.py"

# ── §7 PHAS weight calibration ────────────────────────────────────────────────
skip_or_run human_eval_phas "human_eval/human_eval_analysis.json" \
    "cd '${REPO_DIR}/human_eval' && python calibrate_phas.py"

# ── §8 Ablation A4: cross-VLM auditor (Claude) ────────────────────────────────
skip_or_run ablation_a4 "results/summaries/a4_ablation_results.json" \
    "cd '${REPO_DIR}/ablations/a4_cross_vlm' && \
     python vlm_evaluator_claude.py && python analyze_claude_vs_gemini.py"

# ── §8 Ablation A5: run reliability ───────────────────────────────────────────
skip_or_run ablation_a5 "results/summaries/a5_ablation_results.json" \
    "cd '${REPO_DIR}/ablations/a5_run_reliability' && python compare_runs.py"

# ── §8 Ablation A6: Gemma 4 open-source VLM ──────────────────────────────────
if [[ -n "${GEMMA4_MODEL_PATH:-}" ]]; then
    skip_or_run ablation_a6 "results/gemma4_vlm/summary_report_gemma4.json" \
        "cd '${REPO_DIR}/ablations/a6_open_source_vlm' && \
         python vlm_evaluator_gemma4.py && python analyze_gemma4_results.py"
else
    log "ablation_a6: SKIP (GEMMA4_MODEL_PATH not set; pre-computed results are in the dataset)."
fi

# ── §9 VBench comparison ──────────────────────────────────────────────────────
# Requires VBench installed and VBENCH_DIR set.
# For each model: python vbench/run_single_model.py <model_key>
# Then aggregate: python vbench/run_vbench_eval.py
if [[ -n "${VBENCH_DIR:-}" ]]; then
    skip_or_run vbench "vbench/vbench_summary.json" \
        "cd '${REPO_DIR}' && for m in \
            fal-ai_veo3.1_fast \
            fal-ai_kling-video_v2.6_pro_text-to-video \
            fal-ai_wan_v2.2-a14b_text-to-video \
            fal-ai_ltx-2_text-to-video \
            fal-ai_hunyuan-video-v1.5_text-to-video \
            wan2.1-1.3b; do
              python vbench/run_single_model.py \$m 0
              python vbench/run_single_model_vbench2.py \$m 0
            done && python vbench/run_vbench_eval.py && python vbench/run_vbench2_eval.py"
else
    log "vbench: SKIP (VBENCH_DIR not set). Pre-computed vbench_summary.json is in the dataset."
fi

# ── Appendix ablations ────────────────────────────────────────────────────────
skip_or_run ablation_a1 "results/summaries/a1_enhancement_ablation_results.json" \
    "cd '${REPO_DIR}/ablations/a1_prompt_enhancement' && \
     python vlm_evaluator_ablation_a1.py && python compare_a1.py"

skip_or_run ablation_a2 "results/summaries/question_count_ablation_results_unified.json" \
    "cd '${REPO_DIR}/ablations/a2_question_count' && python question_count_ablation.py"

# ── Generate all paper figures ────────────────────────────────────────────────
log "figures: generating..."
cd "${REPO_DIR}/figures"
python generate_figures.py
python generate_bt_comparison.py
python generate_vbench_evidence.py
python generate_vbench_comparison.py
python generate_gemma4_comparison.py
python generate_dragon_appendix.py
python generate_assassin_appendix.py
log "figures: done. Written to figures/"

log ""
log "All done. Results are in ${DATA_ROOT}/results/ and figures/ ."
