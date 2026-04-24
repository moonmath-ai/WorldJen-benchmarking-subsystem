import json
import os
import time
import datetime
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

GEMINI_MODEL = "models/gemini-3-flash-preview"

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))
ENV_PATH = REPO_ROOT / ".env"
INPUT_PROMPTS = DATA_ROOT / "prompts/prompts_50.jsonl"
OUTPUT_DIR = DATA_ROOT / "prompts"
OUTPUT_VQA = OUTPUT_DIR / "vqa_questions_50prompts.jsonl"

load_dotenv(ENV_PATH)
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

DIMENSIONS = {
    "motion_stability": ["subject_consistency", "scene_consistency", "motion_smoothness", "temporal_flickering", "inertial_consistency"],
    "logic_physics": ["physical_mechanics", "object_permanence", "human_fidelity", "dynamic_degree"],
    "instruction_adherence": ["semantic_adherence", "spatial_relationship", "semantic_drift"],
    "aesthetic_quality": ["composition_framing", "lighting_volumetric", "color_harmony", "structural_gestalt"]
}

DIMENSION_DEFINITIONS = {
    "subject_consistency": "Does the main character/object change shape, color, or identity during the video?",
    "scene_consistency": "Does the environment (trees, buildings, background) stay stable or 'warp'/'melt' as the camera moves?",
    "motion_smoothness": "Does the video have 'stuttering,' 'jitter,' or frames that look like they're skipping?",
    "temporal_flickering": "Are there flashes of light or sudden brightness changes (unwanted flickering/artifacts)?",
    "inertial_consistency": "Do objects follow the laws of momentum - speeding up and slowing down naturally?",
    "physical_mechanics": "Do gravity, friction, and collisions look realistic?",
    "object_permanence": "If an object goes out of view or behind a wall, does it look exactly the same when it reappears?",
    "human_fidelity": "Are humans rendered without 'alien' artifacts like extra fingers, distorted faces, etc.?",
    "dynamic_degree": "Is there actual movement, or is it just a still image with zoom?",
    "semantic_adherence": "Does the video contain exactly what was asked for in the prompt?",
    "spatial_relationship": "Are objects in the right place relative to each other?",
    "semantic_drift": "Does the AI start following the prompt but 'forget' it and change the scene halfway through?",
    "composition_framing": "Is the shot well-balanced, or does it feel like a random crop?",
    "lighting_volumetric": "Is the lighting realistic with depth, or does it look flat and 'CGI-like'?",
    "color_harmony": "Are the colors pleasing and consistent, or is there 'digital bleeding'?",
    "structural_gestalt": "Do the elements look like they belong in the same world, or like stickers pasted on?"
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

def generate_vqa_for_prompt(model, record):
    prompt_text = record.get("enhanced_prompt", record.get("prompt", ""))
    prompt_id = record.get("prompt_id", "unknown")
    
    results = {}
    
    for group, dims in DIMENSIONS.items():
        print(f"  Processing group: {group} for {prompt_id}")
        dim_context = "\n".join([f"- {d}: {DIMENSION_DEFINITIONS[d]}" for d in dims])
        user_msg = f"PROMPT: {prompt_text}\n\nDIMENSIONS TO ANALYZE:\n{dim_context}\n\nGenerate 10 questions per dimension."
        
        retries = 5
        while retries > 0:
            try:
                # Use a more specific model version
                response = model.generate_content([SYSTEM_PROMPT, user_msg])
                text = response.text
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                
                group_data = json.loads(text)
                results.update(group_data)
                break
            except Exception as e:
                print(f"    Error generating {group}: {e}. Retrying...")
                retries -= 1
                time.sleep(5) # Increase sleep to handle rate limits
        
        if retries == 0:
            print(f"    Failed to generate VQA for group {group}")
            
    return results

def main():
    model = genai.GenerativeModel(GEMINI_MODEL)
    gen_timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    print(f"Initialized model: {GEMINI_MODEL} at {gen_timestamp}", flush=True)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(INPUT_PROMPTS):
        print(f"Error: Input file {INPUT_PROMPTS} not found.")
        return

    with open(INPUT_PROMPTS, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
    
    existing_ids = set()
    if OUTPUT_VQA.exists():
        with open(OUTPUT_VQA, "r") as f:
            for line in f:
                if line.strip():
                    existing_ids.add(json.loads(line)["prompt_id"])
    
    with open(OUTPUT_VQA, "a") as out:
        for rec in records:
            pid = rec["prompt_id"]
            if pid in existing_ids:
                print(f"Skipping {pid} (already exists)")
                continue
            
            print(f"Generating VQA for {pid}...")
            vqa_data = generate_vqa_for_prompt(model, rec)
            
            if not vqa_data:
                print(f"  Warning: No VQA data generated for {pid}")
            else:
                print(f"  Success: Generated VQA for {len(vqa_data)} dimensions for {pid}")
            
            output_rec = {
                "prompt_id": pid,
                "prompt": rec.get("enhanced_prompt", rec.get("prompt", "")),
                "gemini_model": GEMINI_MODEL,
                "gen_timestamp": gen_timestamp,
                "vqa": vqa_data
            }
            out.write(json.dumps(output_rec) + "\n")
            out.flush()
            print(f"  Wrote results for {pid} to {OUTPUT_VQA}")
            time.sleep(2) # Throttle a bit

if __name__ == "__main__":
    main()
