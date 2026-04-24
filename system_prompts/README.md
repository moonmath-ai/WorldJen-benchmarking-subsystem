# System Prompts

This directory contains the verbatim system prompts used in the WorldJen benchmarking pipeline.

## Phase A — Prompt Curation

Model: `gemini-3.1-flash-lite-preview` (Google)

| File | Stage | Purpose |
|------|-------|---------|
| `phase_a_llm_judge.txt` | Phase A, Step 1 | LLM Judge — scores each candidate prompt on suitability (does it stress-test this dimension?) and difficulty (1–10) across 16 dimensions in 4 groups (Motion, Logic, Adherence, Aesthetic). Used by `prompt_pipeline/scripts/gemini_judge.py`. |
| `phase_a_llm_enhance.txt` | Phase A, Step 2 | LLM Enhancer — rewrites a prompt to strengthen its weakest dimensions while preserving its core theme. Used by `prompt_pipeline/scripts/gemini_enhance.py` and `gemini_enhance_pass2.py`. |

## Phase B — VLM Evaluation Engine

Model: `gemini-3-flash-preview` (Google)

| File | Stage | Purpose |
|------|-------|---------|
| `phase_b_vqa_generator.txt` | Phase B, Step 1 | VQA Question Generator — given an enhanced prompt and a set of dimensions, generates 10 probing VQA questions per dimension with a 1–5 scoring rubric. The call is split by group (A–D); dimensions with `null` suitability are omitted. Used by `vlm_eval/vqa_generator.py`. |
| `phase_b_vlm_evaluator.txt` | Phase B, Step 2 | VLM Evaluator — given a video's extracted frames and the 10 VQA questions for one dimension, returns a score (1–5) and justification per question. Runs once per (video, dimension) pair. Used by `vlm_eval/parallel_vlm_evaluator.py`. |

## Notes

- The prompts are used verbatim — curly-brace placeholders (e.g. `{dimension_name}`) are filled in by the calling script at runtime.
- The paper refers to these prompts at §3.1 (Phase A) and §3.2 (Phase B) with the note *"available in the code release"*.
- For full context on how each prompt fits into the pipeline, see the [data flow diagram in the main README](../README.md#data-flow).
