#!/usr/bin/env bash
# Run the full prompt pipeline: judge → enhance → pass2 → rescore
# Reads unenhanced prompts from data/ and writes outputs there.
#
# Prerequisites:
#   pip install -r requirements.txt
#   export GEMINI_API_KEY=your_key   (or add to ../.env)
#   python download_data.py          (to get data/prompts/prompts_unenhanced_50.jsonl)
#
# Usage:
#   bash prompt_pipeline/run_pipeline.sh [INPUT] [OUTPUT_DIR]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

INPUT="${1:-$REPO_DIR/data/prompts/prompts_unenhanced_50.jsonl}"
OUTPUT_DIR="${2:-$REPO_DIR/data/prompts/pipeline_out}"
MAX_PROMPTS="${MAX_PROMPTS:-}"

mkdir -p "$OUTPUT_DIR"
JUDGED="$OUTPUT_DIR/judged.jsonl"
ENHANCED="$OUTPUT_DIR/enhanced.jsonl"
PASS2="$OUTPUT_DIR/enhanced_pass2.jsonl"
RESCORED="$OUTPUT_DIR/rescored.jsonl"

echo "Input:      $INPUT"
echo "Output dir: $OUTPUT_DIR"
echo ""

echo "Step 1: Judge (all dimensions + aesthetics)"
JUDGE_ARGS="--input $INPUT --output $JUDGED --resume"
[[ -n "$MAX_PROMPTS" ]] && JUDGE_ARGS="$JUDGE_ARGS --max-prompts $MAX_PROMPTS"
python prompt_pipeline/scripts/gemini_judge.py $JUDGE_ARGS

echo ""
echo "Step 2: Fix flagged prompts"
python prompt_pipeline/scripts/gemini_fix_review_prompts.py \
  --input "$JUDGED" --output "$OUTPUT_DIR/judged_review_fixed.jsonl"

echo ""
echo "Step 3: Enhance"
ENH_ARGS="--input $OUTPUT_DIR/judged_review_fixed.jsonl --output $ENHANCED --resume"
[[ -n "$MAX_PROMPTS" ]] && ENH_ARGS="$ENH_ARGS --max-prompts $MAX_PROMPTS"
python prompt_pipeline/scripts/gemini_enhance.py $ENH_ARGS

echo ""
echo "Step 3b: Remove dimension jargon (pass 2)"
python prompt_pipeline/scripts/gemini_enhance_pass2.py \
  --input "$ENHANCED" --output "$PASS2"

echo ""
echo "Step 4: Rescore enhanced prompts"
RESCORE_ARGS="--input $PASS2 --output $RESCORED --prompt-field enhanced_prompt --resume"
[[ -n "$MAX_PROMPTS" ]] && RESCORE_ARGS="$RESCORE_ARGS --max-prompts $MAX_PROMPTS"
python prompt_pipeline/scripts/gemini_judge_enhanced.py $RESCORE_ARGS

echo ""
echo "Pipeline done.  Final output: $RESCORED"
