import json
import os
import time
import asyncio
import datetime
import cv2
import PIL.Image
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

GEMINI_MODEL = "models/gemini-3-flash-preview"

# Request / retry tuning
REQUEST_TIMEOUT_S  = 240   # per-call HTTP timeout (seconds)
ASYNCIO_TIMEOUT_S  = 270   # asyncio-level safety net (slightly above HTTP timeout)
MAX_RETRIES        = 5
BATCH_SIZE         = 2     # dimensions evaluated in parallel per video
VIDEO_COOLDOWN_S   = 5     # sleep between videos

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))
ENV_PATH     = REPO_ROOT / ".env"
PROMPTS_FILE = DATA_ROOT / "prompts/prompts_50.jsonl"
VQA_QUESTIONS = DATA_ROOT / "prompts/vqa_questions_50prompts.jsonl"
VIDEO_ROOT   = DATA_ROOT / "videos"
OUTPUT_DIR   = DATA_ROOT / "results/gemini_vlm"

load_dotenv(ENV_PATH)
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

MODELS_PATHS = [
    "fal-ai_kling-video_v2.6_pro_text-to-video",
    "fal-ai_ltx-2_text-to-video",
    "fal-ai_hunyuan-video-v1.5_text-to-video",
    "fal-ai_veo3.1_fast",
    "fal-ai_wan_v2.2-a14b_text-to-video",
    "wan2.1-1.3b"
]

DIMENSION_MODES = {
    "semantic_adherence": "holistic", "composition_framing": "holistic", "aesthetic_quality": "holistic",
    "lighting_volumetric": "holistic", "color_harmony": "holistic", "structural_gestalt": "holistic",
    "scene_consistency": "sampled", "object_permanence": "sampled", "subject_consistency": "sampled",
    "motion_smoothness": "micro", "temporal_flickering": "micro", "inertial_consistency": "micro",
    "physical_mechanics": "micro", "human_fidelity": "micro", "dynamic_degree": "holistic",
    "semantic_drift": "holistic"
}

def extract_frames(video_path, mode="sampled", num_frames=16):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if mode == "sampled":
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    elif mode == "micro":
        indices = list(range(0, min(total_frames, 60), 5))
    else:
        indices = [int(i * total_frames / 32) for i in range(min(32, total_frames))]
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(PIL.Image.fromarray(frame_rgb))
    cap.release()
    return frames

async def evaluate_dimension(model, video_path, questions, dimension, prompt_id, model_name):
    mode = DIMENSION_MODES.get(dimension, "holistic")
    print(f"  [{model_name}][{prompt_id}] Evaluating {dimension}...", flush=True)
    
    # Run CPU-bound frame extraction in a thread
    frames = await asyncio.to_thread(extract_frames, video_path, mode)
    
    prompt = f"You are evaluating a video generated from a prompt. Dimension: {dimension}\n\nQuestions to answer:\n"
    for i, q in enumerate(questions):
        prompt += f"{i+1}. {q['question']}\n   Rubric: {q['rubric_description']}\n"
    prompt += "\nAnswer each question with a score (1-5) and a short justification. Return ONLY a JSON list of objects: [{'score': X, 'justification': '...'}]"

    for attempt in range(MAX_RETRIES):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    model.generate_content,
                    [prompt] + frames,
                    request_options={"timeout": REQUEST_TIMEOUT_S}
                ),
                timeout=ASYNCIO_TIMEOUT_S
            )
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            return dimension, json.loads(text)
        except asyncio.TimeoutError:
            wait_time = (2 ** attempt) * 15
            print(f"  [{model_name}][{prompt_id}] TIMEOUT on {dimension} (attempt {attempt+1}). Backing off {wait_time}s...", flush=True)
            await asyncio.sleep(wait_time)
        except Exception as e:
            if "429" in str(e):
                wait_time = (2 ** attempt) * 10
                print(f"  [{model_name}][{prompt_id}] Rate limit on {dimension}. Backing off {wait_time}s...", flush=True)
                await asyncio.sleep(wait_time)
            elif "timeout" in str(e).lower() or "deadline" in str(e).lower():
                wait_time = (2 ** attempt) * 15
                print(f"  [{model_name}][{prompt_id}] API timeout on {dimension} (attempt {attempt+1}). Backing off {wait_time}s...", flush=True)
                await asyncio.sleep(wait_time)
            else:
                print(f"  [{model_name}][{prompt_id}] Error on {dimension}: {e}. Retrying in 5s...", flush=True)
                await asyncio.sleep(5)
    print(f"  [{model_name}][{prompt_id}] FAILED {dimension} after {MAX_RETRIES} attempts.", flush=True)
    return dimension, None

async def process_video(model, video_path, vqa_set, prompt_id, model_name, prompt_set="unknown"):
    output_file = OUTPUT_DIR / f"{model_name}_{prompt_id}.json"
    if output_file.exists():
        return

    print(f"Processing Video: {model_name}/{prompt_id} [{prompt_set}]", flush=True)
    tasks = []
    for dimension, questions in vqa_set.items():
        tasks.append(evaluate_dimension(model, video_path, questions, dimension, prompt_id, model_name))
    
    results = {}
    for i in range(0, len(tasks), BATCH_SIZE):
        batch_results = await asyncio.gather(*tasks[i:i+BATCH_SIZE])
        for dim, res in batch_results:
            results[dim] = res
            
    with open(output_file, "w") as f:
        json.dump({
            "model": model_name,
            "prompt_id": prompt_id,
            "prompt_set": prompt_set,
            "gemini_model": GEMINI_MODEL,
            "eval_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "results": results
        }, f, indent=2)
    print(f"DONE Video: {model_name}/{prompt_id} -> {output_file}", flush=True)

async def main():
    model = genai.GenerativeModel(GEMINI_MODEL)
    eval_timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    print(f"Initialized model: {GEMINI_MODEL} at {eval_timestamp}", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build prompt_set lookup from prompts_50.jsonl (has "prompt_set" field)
    prompt_set_map = {}
    with open(PROMPTS_FILE) as f:
        for line in f:
            entry = json.loads(line)
            prompt_set_map[entry["prompt_id"]] = entry["prompt_set"]

    questions_map = {}
    with open(VQA_QUESTIONS, "r") as f:
        for line in f:
            data = json.loads(line)
            questions_map[data["prompt_id"]] = data["vqa"]

    video_tasks = []
    for rel_path in MODELS_PATHS:
        model_name = rel_path.split("/")[-1]
        path = Path(VIDEO_ROOT) / rel_path
        if not path.exists(): continue
        
        for video_file in path.glob("*.mp4"):
            pid = video_file.stem
            if pid in questions_map:
                video_tasks.append(process_video(model, video_file, questions_map[pid], pid, model_name, prompt_set_map.get(pid, "unknown")))

    # Process videos sequentially to keep overall rate limits manageable, 
    # but process dimensions within each video in parallel
    for task in video_tasks:
        await task
        await asyncio.sleep(VIDEO_COOLDOWN_S)

if __name__ == "__main__":
    asyncio.run(main())
