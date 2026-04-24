"""
vqa_generator_ablation_a1.py

Generates 10 VQA questions per applicable dimension for the 20 UNENHANCED prompts
used in the A1 ablation study.

Source prompts: data/prompts/prompts_ablation_a1_validation20.jsonl
    (uses 'prompt' field, not 'enhanced_prompt'; suitability scores taken directly
    from the same file — judged_reviewed_final scores of the original pre-enhancement prompt)
Null-suitability dimensions are skipped, exactly as in the main study.

Output: data/prompts/vqa_questions_ablation_a1.jsonl

Usage:
    python vqa_generator_ablation_a1.py
"""

import json
import os
import time
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))
ENV_PATH       = REPO_ROOT / ".env"
PROMPTS_FILE   = DATA_ROOT / "prompts/prompts_ablation_a1_validation20.jsonl"
OUTPUT_VQA     = DATA_ROOT / "prompts/vqa_questions_ablation_a1.jsonl"

# ── Dimension definitions (identical to main study) ───────────────────────────
DIMENSIONS = {
    "motion_stability":      ["subject_consistency", "scene_consistency",
                               "motion_smoothness", "temporal_flickering",
                               "inertial_consistency"],
    "logic_physics":         ["physical_mechanics", "object_permanence",
                               "human_fidelity", "dynamic_degree"],
    "instruction_adherence": ["semantic_adherence", "spatial_relationship",
                               "semantic_drift"],
    "aesthetic_quality":     ["composition_framing", "lighting_volumetric",
                               "color_harmony", "structural_gestalt"],
}

DIMENSION_DEFINITIONS = {
    "subject_consistency":  "Does the main character/object change shape, color, or identity during the video?",
    "scene_consistency":    "Does the environment stay stable or 'warp'/'melt' as the camera moves?",
    "motion_smoothness":    "Does the video have 'stuttering,' 'jitter,' or frames that look like they're skipping?",
    "temporal_flickering":  "Are there flashes of light or sudden brightness changes (unwanted flickering/artifacts)?",
    "inertial_consistency": "Do objects follow the laws of momentum — speeding up and slowing down naturally?",
    "physical_mechanics":   "Do gravity, friction, and collisions look realistic?",
    "object_permanence":    "If an object goes out of view, does it look exactly the same when it reappears?",
    "human_fidelity":       "Are humans rendered without 'alien' artifacts like extra fingers, distorted faces, etc.?",
    "dynamic_degree":       "Is there actual movement, or is it just a still image with zoom?",
    "semantic_adherence":   "Does the video contain exactly what was asked for in the prompt?",
    "spatial_relationship": "Are objects in the right place relative to each other?",
    "semantic_drift":       "Does the AI start following the prompt but 'forget' it halfway through?",
    "composition_framing":  "Is the shot well-balanced, or does it feel like a random crop?",
    "lighting_volumetric":  "Is the lighting realistic with depth, or does it look flat and 'CGI-like'?",
    "color_harmony":        "Are the colors pleasing and consistent, or is there 'digital bleeding'?",
    "structural_gestalt":   "Do the elements look like they belong in the same world, or like stickers pasted on?",
}

GROUP_FIELD = {
    "subject_consistency":  "motion_stability",
    "scene_consistency":    "motion_stability",
    "motion_smoothness":    "motion_stability",
    "temporal_flickering":  "motion_stability",
    "inertial_consistency": "motion_stability",
    "physical_mechanics":   "logic_physics",
    "object_permanence":    "logic_physics",
    "human_fidelity":       "logic_physics",
    "dynamic_degree":       "logic_physics",
    "semantic_adherence":   "instruction_adherence",
    "spatial_relationship": "instruction_adherence",
    "semantic_drift":       "instruction_adherence",
    "composition_framing":  "aesthetic_quality",
    "lighting_volumetric":  "aesthetic_quality",
    "color_harmony":        "aesthetic_quality",
    "structural_gestalt":   "aesthetic_quality",
}

SYSTEM_PROMPT = """
You are a video generation benchmarking expert. Your task is to generate 10 unique, probing VQA (Video Question Answering) questions for EACH of the dimensions listed below, specifically for a video generated from the provided PROMPT.

For each DIMENSION of the PROMPT:
1. Generate 10 questions that specifically probe that dimension as it relates to this prompt.
2. Questions should cover:
   - Expected events and details mentioned in the prompt.
   - Potential failure modes (e.g., "Does the character's face distort when they turn?").
   - Success modes (e.g., "Is the reflection on the water consistent with the light source?").
   - Adversarial probing (checking for subtle inconsistencies).
3. For each question, define a 1-5 scoring rubric:
   - 1: Major failure / Completely incorrect.
   - 2: Notable artifacts / Significant issues.
   - 3: Mediocre / passable but flawed.
   - 4: Good / minor imperfections only.
   - 5: Perfect / Flawless execution.

Return ONLY a JSON object where keys are the dimension names and values are lists of 10 question objects. Each question object must have "question" and "rubric_description".
"""


def get_applicable_dims(record):
    """Return (applicable, skipped) based on suitability scores in record."""
    applicable, skipped = [], []
    for dim, grp in GROUP_FIELD.items():
        suit = (record.get(grp) or {}).get(f"{dim}_suitability")
        if suit is None:
            skipped.append(dim)
        else:
            applicable.append(dim)
    return applicable, skipped


def generate_vqa_for_prompt(model, record):
    prompt_text = record.get("prompt", "")   # unenhanced prompt
    prompt_id   = record.get("prompt_id", "unknown")

    applicable, skipped = get_applicable_dims(record)
    if skipped:
        print(f"  [{prompt_id}] Skipping dims (null suitability): {skipped}")

    results = {}
    for group, dims in DIMENSIONS.items():
        active_dims = [d for d in dims if d in applicable]
        if not active_dims:
            print(f"  [{prompt_id}] group {group}: all dims skipped")
            continue

        print(f"  [{prompt_id}] group: {group}  dims: {active_dims}")
        dim_context = "\n".join([f"- {d}: {DIMENSION_DEFINITIONS[d]}" for d in active_dims])
        user_msg = (
            f"PROMPT: {prompt_text}\n\n"
            f"DIMENSIONS TO ANALYZE:\n{dim_context}\n\n"
            "Generate 10 questions per dimension."
        )

        retries = 5
        while retries > 0:
            try:
                response = model.generate_content([SYSTEM_PROMPT, user_msg])
                text = response.text
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                group_data = json.loads(text)
                for d in active_dims:
                    if d in group_data:
                        results[d] = group_data[d]
                break
            except Exception as e:
                print(f"    Error ({group}): {e}. Retrying in 10s...")
                retries -= 1
                time.sleep(10)

        if retries == 0:
            print(f"    FAILED to generate VQA for group {group} — skipping")

    return results, applicable, skipped


def main():
    load_dotenv(ENV_PATH)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("models/gemini-3.1-flash-lite-preview")

    # Load all 20 prompts
    prompts = []
    with open(PROMPTS_FILE) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} unenhanced prompts from {PROMPTS_FILE}")

    # Resume support
    done_ids = set()
    if OUTPUT_VQA.exists():
        with open(OUTPUT_VQA) as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["prompt_id"])
    remaining = [p for p in prompts if p["prompt_id"] not in done_ids]
    print(f"Already done: {len(done_ids)}  |  Remaining: {len(remaining)}\n")

    with open(OUTPUT_VQA, "a") as out:
        for p in remaining:
            pid = p["prompt_id"]
            print(f"\nGenerating VQA for {pid}  ({p['prompt'][:80]}...)")
            vqa_data, applicable, skipped = generate_vqa_for_prompt(model, p)

            if not vqa_data:
                print(f"  WARNING: no VQA data generated for {pid}")
            else:
                print(f"  OK: {len(vqa_data)} dims, {sum(len(v) for v in vqa_data.values())} questions  |  skipped: {skipped}")

            output_rec = {
                "prompt_id":       pid,
                "prompt":          p["prompt"],   # unenhanced
                "applicable_dims": applicable,
                "skipped_dims":    skipped,
                "vqa":             vqa_data,
            }
            out.write(json.dumps(output_rec) + "\n")
            out.flush()
            time.sleep(2)

    print(f"\nDone. Output → {OUTPUT_VQA}")


if __name__ == "__main__":
    main()
