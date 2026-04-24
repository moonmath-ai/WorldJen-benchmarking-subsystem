#!/usr/bin/env python3
"""
Judge prompts via Gemini API: score all dimension groups (A–D) including aesthetics.
Input: JSONL with "prompt" (and optional prompt_id). Output: same + dimension scores.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

# Allow importing config from project root when run as scripts/gemini_judge.py
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import google.generativeai as genai

from config import get_api_key
from rate_limits import RATE_LIMIT_SLEEP, is_rate_limit, get_safe_delay, DEFAULT_DELAY

# Optional: context cache for JUDGE_SYSTEM (reduces TPM and helps rate limits)
_CACHE_TTL_HOURS = 1

# Full judge prompt: Groups A, B, C (from phase2_judge) + D aesthetics (from phase2_aesthetic_dimensions)
JUDGE_SYSTEM = """You are a video generation benchmarking expert. Analyze the prompt and provide scores on multiple dimensions.

A prompt can test MULTIPLE groups. Determine which groups apply and score all relevant dimensions.

**Group A: Motion & Stability** (Does this prompt test video stability or motion artifacts?)
Applicable if: prompt involves camera movement, moving subjects, or temporal consistency challenges

For each dimension, score BOTH suitability and difficulty (1-10 each):

1. **subject_consistency**: Does the main character/object change shape, color, or identity during the video?
   - suitability: Does this prompt create conditions where subject inconsistency would be exposed? (high=complex subject with changes/transformations/motion that would reveal identity shifts; low=static/simple subject unlikely to change)
   - difficulty: How hard for a video model to keep this subject's identity/shape/color consistent throughout? (1=easy, simple static subject; 10=very hard, complex moving/transforming subject)

2. **scene_consistency**: Does the environment (trees, buildings, background) stay stable or "warp"/"melt" as the camera moves?
   - suitability: Does this prompt create conditions where scene warping/melting would be exposed? (high=camera motion through complex environment that might warp; low=static camera or simple background)
   - difficulty: How hard for a video model to keep this scene stable during camera motion? (1=easy, static/simple scene; 10=very hard, complex environment with camera movement)

3. **motion_smoothness**: Does the video have "stuttering," "jitter," or frames that look like they're skipping?
   - suitability: Does this prompt create conditions where motion jitter/stuttering would be exposed? (high=fast/complex motion that would reveal frame skips; low=slow/simple motion where stuttering is less noticeable)
   - difficulty: How hard for a video model to render this motion smoothly? (1=easy, slow/simple motion; 10=very hard, fast/complex motion)
   - NOTE: This is about rendering quality (frame skips), NOT physics. Focus on speed and complexity of motion.

4. **temporal_flickering**: Are there flashes of light or sudden brightness changes that shouldn't be there?
   - suitability: Does this prompt create conditions where unwanted flickering/artifacts would be a problem? Score high only if the prompt involves complex textures like water, hair, fire, smoke, or fine patterns which typically trigger neural flickering. (high=complex textures/lighting that might cause artifacts; low=uniform textures unlikely to flicker). Note: intentional lighting changes (sunrise, lights turning on) are NOT flickering.
   - difficulty: How hard for a video model to avoid unwanted flickering? (1=easy, uniform/simple lighting and textures; 10=very hard, complex lighting/fine textures)

5. **inertial_consistency**: Do objects follow the laws of momentum - speeding up and slowing down naturally like real objects?
   - suitability: Does this prompt create conditions where inertia violations would be exposed? Focus on whether the prompt implies a CHANGE in velocity (falling, stopping, throwing, catching, sliding to a stop). (high=objects with velocity changes; low=constant-velocity or static objects)
   - difficulty: How hard for a video model to render physically accurate inertia? (1=easy, no acceleration changes; 10=very hard, complex acceleration/deceleration)
   - NOTE: This is about physics (velocity changes), NOT rendering smoothness. Focus on acceleration/deceleration.

**Group B: Logic & Physics** (Does this prompt test physical realism or logic?)
Applicable if: prompt involves physical interactions, gravity, collisions, humans/animals, or object persistence

For each dimension, score BOTH suitability and difficulty (1-10 each):

1. **physical_mechanics**: Do gravity, friction, and collisions look realistic?
   - suitability: Does this prompt create conditions where physics violations would be exposed? (high=objects falling/bouncing/colliding that must obey physics; low=no physical interactions)
   - difficulty: How hard for a video model to render physically accurate mechanics? (1=easy, no physics interactions; 10=very hard, complex physics like fluid/cloth/collisions)

2. **object_permanence**: If an object goes out of view or behind a wall, does it look exactly the same when it reappears?
   - suitability: Does this prompt create conditions where object permanence failures would be exposed? (high=objects going behind/under things that must reappear unchanged; low=no occlusion)
   - difficulty: How hard for a video model to maintain object identity after occlusion? (1=easy, no occlusion; 10=very hard, complex occlusion with object changes)

3. **human_fidelity**: Are humans rendered with "alien" artifacts like extra fingers, distorted faces, or impossible body twisting?
   - suitability: Does this prompt create conditions where human rendering artifacts would be exposed? (high=close-ups of hands/faces, complex poses; low=no humans or distant humans; null if no humans)
   - difficulty: How hard for a video model to render humans without artifacts? (1=easy, simple pose/distant; 10=very hard, close-up hands/complex poses; null if no humans)

4. **dynamic_degree**: Is there actual movement, or is it just a still image with zoom?
   - suitability: Does this prompt require substantial object/character movement (not just camera pan/zoom)? (high=requires state transformation/motion; low=static scene acceptable)
   - difficulty: How much genuine movement must the video model generate? (1=static/camera-only acceptable, 10=high action/state transformation required)

**Group C: Instruction Adherence** (Does this prompt test following specific instructions?)
Applicable if: prompt has specific objects, colors, spatial relationships, or precise requirements

For each dimension, score BOTH suitability and difficulty (1-10 each):

1. **semantic_adherence**: Does the video contain exactly what was asked for?
   - suitability: Does this prompt create conditions where semantic violations would be exposed? (high=specific colors/attributes/objects that must match exactly; low=vague/general description)
   - difficulty: How hard for a video model to match the exact semantic requirements? (1=easy, vague/general; 10=very hard, many specific attributes to match precisely)

2. **spatial_relationship**: Are objects in the right place relative to each other?
   - suitability: Does this prompt create conditions where spatial errors would be exposed? (high=specific spatial relationships stated (on/under/holding/inside); low=no spatial relationships specified)
   - difficulty: How hard for a video model to get the spatial relationships right? (1=easy, no spatial constraints; 10=very hard, complex spatial positioning required)

3. **semantic_drift**: Does the AI start following the prompt but "forget" it and change the scene halfway through?
   - suitability: Does this prompt create conditions where semantic drift would be exposed? (high=multi-stage or sustained action; low=single brief action)
   - difficulty: How hard for a video model to maintain the concept throughout the video? (1=easy, brief single action; 10=very hard, multi-stage sequence that must persist)

**Group D: Aesthetic Quality** (Does this prompt test artistic rendering and visual polish?)
Applicable if: prompt involves specific artistic styles, high-detail environments, or cinematic descriptions.

For each dimension, score BOTH suitability and difficulty (1-10 each):

1. **composition_framing**: Is the shot well-balanced, or does it feel like a random crop?
   - suitability: Does the prompt specify a shot type (Close-up, Wide, POV) or a specific composition? (high=explicit framing instructions; low=vague description)
   - difficulty: How hard is it to maintain this specific framing during the requested motion? (1=easy, static wide; 10=very hard, complex tracking or macro shots)

2. **lighting_volumetric**: Is the lighting realistic with depth, or does it look flat and "CGI-like"?
   - suitability: Does the prompt involve complex light sources (Neon, Sunset, Volumetric fog, multiple lights)? (high=lighting-heavy prompt; low=flat daylight)
   - difficulty: How hard for a model to calculate realistic light/shadow for this scene? (1=easy, single light; 10=very hard, translucent objects or ray-traced reflections)

3. **color_harmony**: Are the colors pleasing and consistent, or is there "digital bleeding"?
   - suitability: Does the prompt ask for a specific color palette (Monochromatic, Vibrant, Pastel)? (high=specific color requirements; low=standard natural colors)
   - difficulty: How hard to maintain this color grade across the entire sequence? (1=easy, natural colors; 10=very hard, stylized or shifting color palettes)

4. **structural_gestalt**: Do the elements look like they belong in the same world, or like stickers pasted on?
   - suitability: Does the prompt involve complex compositing (e.g., a "dragon in a modern kitchen")? (high=interacting textures that must blend; low=simple landscapes)
   - difficulty: How hard to blend these specific textures and objects seamlessly? (1=easy, uniform textures; 10=very hard, complex material interactions)

**Categories**: Assign ALL categories that fit the prompt. Use any of these when they apply: Human, Animal, Nature, Tech, Abstract, Interaction. You may also add any other category labels you find appropriate (e.g. Cinematic, Fantasy, Product, Sports). Return a list of strings; use only categories that genuinely apply.

**Confidence** (1-10): How confident are you in the group assignments and scores?

**Flag for review**: If the prompt is inappropriate by your judgment (e.g. harmful, policy-violating, unsafe) or raises copyright/sensitivity concerns (e.g. specific copyrighted characters, brands, or real people in a way that may be problematic), set "needs_review" to true and set "review_reason" to a short explanation (e.g. "copyright: references Disney character", "inappropriate: violence"). Otherwise set "needs_review" to false and "review_reason" to null.

Scoring guidelines:
- Suitability: 1=poor test for this dimension, 5=decent test, 10=excellent/ideal test
- Difficulty: 1=easy for model, 5=moderate challenge, 10=extremely hard for model
- Only include a group in applicable_groups if relevant to the prompt. Set scores to null for non-applicable dimensions.

Return ONLY valid JSON (no extra text, no markdown). Use this exact structure (all four dimension groups in the same format; aesthetic_quality alongside the others). Include "needs_review" and "review_reason":
{"applicable_groups": ["motion_stability", "logic_physics", "instruction_adherence", "aesthetic_quality"], "motion_stability": {"subject_consistency_suitability": 7, "subject_consistency_difficulty": 6, "scene_consistency_suitability": 8, "scene_consistency_difficulty": 7, "motion_smoothness_suitability": 8, "motion_smoothness_difficulty": 7, "temporal_flickering_suitability": 7, "temporal_flickering_difficulty": 6, "inertial_consistency_suitability": 9, "inertial_consistency_difficulty": 8}, "logic_physics": {"physical_mechanics_suitability": 6, "physical_mechanics_difficulty": 5, "object_permanence_suitability": 3, "object_permanence_difficulty": 2, "human_fidelity_suitability": 8, "human_fidelity_difficulty": 7, "dynamic_degree_suitability": 9, "dynamic_degree_difficulty": 7}, "instruction_adherence": {"semantic_adherence_suitability": 8, "semantic_adherence_difficulty": 6, "spatial_relationship_suitability": 7, "spatial_relationship_difficulty": 5, "semantic_drift_suitability": 6, "semantic_drift_difficulty": 4}, "aesthetic_quality": {"composition_framing_suitability": 8, "composition_framing_difficulty": 7, "lighting_volumetric_suitability": 9, "lighting_volumetric_difficulty": 8, "color_harmony_suitability": 6, "color_harmony_difficulty": 5, "structural_gestalt_suitability": 7, "structural_gestalt_difficulty": 6}, "categories": ["Human", "Nature", "Tech", "Abstract", "Interaction"], "confidence": 9, "needs_review": false, "review_reason": null}
"""

# For batched mode: instruction to return a JSON array
JUDGE_BATCH_SUFFIX = """

If you are analyzing MULTIPLE prompts below, return a JSON ARRAY of results: one object per prompt, in the same order as the prompts. Example: [ {"applicable_groups": [...], "motion_stability": {...}, ...}, {"applicable_groups": [...], ...} ]
"""


# Output key order: ids/prompt first, then dimension groups together, then categories/confidence
OUTPUT_KEY_ORDER = [
    "prompt_id", "prompt",
    "applicable_groups",
    "motion_stability", "logic_physics", "instruction_adherence", "aesthetic_quality",
    "categories", "confidence",
    "needs_review", "review_reason",
]

def _ordered_record(rec: dict) -> dict:
    """Build a dict with dimension groups (including aesthetic_quality) in a consistent block."""
    out = {}
    for k in OUTPUT_KEY_ORDER:
        if k in rec:
            out[k] = rec[k]
    for k, v in rec.items():
        if k not in out:
            out[k] = v
    return out


def _parse_judge_response(text: str) -> dict:
    """Extract JSON from model response and normalize."""
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.lower().startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") and p.endswith("}"):
                text = p
                break
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return {"applicable_groups": [], "confidence": 1, "error": "No JSON in response", "raw": text[:500]}
    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError as e:
        return {"applicable_groups": [], "confidence": 1, "error": str(e), "raw": text[start:start+1000]}
    # Ensure all group keys exist
    for key in ("motion_stability", "logic_physics", "instruction_adherence", "aesthetic_quality"):
        data.setdefault(key, {})
    data.setdefault("applicable_groups", [])
    data.setdefault("categories", [])
    data["confidence"] = max(1, min(10, int(data.get("confidence", 7))))
    data.setdefault("needs_review", False)
    data.setdefault("review_reason", None)
    if isinstance(data.get("needs_review"), str):
        data["needs_review"] = data["needs_review"].lower() in ("true", "1", "yes")
    return data


def _parse_judge_response_array(text: str, expected_len: int) -> list[dict]:
    """Parse response as JSON array of judge results. On failure return list of empty dicts."""
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.lower().startswith("json"):
                p = p[4:].strip()
            if p.startswith("["):
                text = p
                break
    start = text.find("[")
    end = text.rfind("]") + 1
    if start < 0 or end <= start:
        return [{"applicable_groups": [], "confidence": 1, "error": "No JSON array in response"} for _ in range(expected_len)]
    try:
        arr = json.loads(text[start:end])
    except json.JSONDecodeError:
        return [{"applicable_groups": [], "confidence": 1, "error": "JSON array parse failed"} for _ in range(expected_len)]
    if not isinstance(arr, list):
        return [{"applicable_groups": [], "confidence": 1, "error": "Response not an array"} for _ in range(expected_len)]
    out = []
    for i, item in enumerate(arr):
        if not isinstance(item, dict):
            out.append({"applicable_groups": [], "confidence": 1, "error": f"Item {i} not a dict"})
            continue
        for key in ("motion_stability", "logic_physics", "instruction_adherence", "aesthetic_quality"):
            item.setdefault(key, {})
        item.setdefault("applicable_groups", [])
        item.setdefault("categories", [])
        item["confidence"] = max(1, min(10, int(item.get("confidence", 7))))
        item.setdefault("needs_review", False)
        item.setdefault("review_reason", None)
        if isinstance(item.get("needs_review"), str):
            item["needs_review"] = item["needs_review"].lower() in ("true", "1", "yes")
        out.append(item)
    # Pad or trim to expected_len
    while len(out) < expected_len:
        out.append({"applicable_groups": [], "confidence": 1, "error": "Missing in batch response"})
    return out[:expected_len]


def judge_one(model, prompt_text: str, delay_seconds: float = 0.0, max_retries: int = 5, system_and_suffix: str = None) -> dict:
    # When using cached model, system_and_suffix is "" so we send only user content
    user_content = f"Prompt to analyze:\n\n{prompt_text}\n\nReturn ONLY the JSON object."
    if system_and_suffix is not None and system_and_suffix == "":
        content = user_content
    else:
        content = (system_and_suffix if system_and_suffix is not None else JUDGE_SYSTEM) + "\n\n" + user_content
    attempt = 0
    consecutive_429 = 0
    while attempt < max_retries:
        try:
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            response = model.generate_content(content)
            consecutive_429 = 0
            text = response.text if response.text else ""
            return _parse_judge_response(text)
        except Exception as e:
            if is_rate_limit(e):
                consecutive_429 += 1
                sleep_secs = int(RATE_LIMIT_SLEEP * (1.5 ** min(consecutive_429 - 1, 2)))
                print(f"  Rate limit reached. Sleeping {sleep_secs}s to reset...")
                time.sleep(sleep_secs)
                continue
            if attempt == max_retries - 1:
                return {"applicable_groups": [], "confidence": 1, "error": str(e)}
            time.sleep(2 ** attempt)
            attempt += 1
    return {"applicable_groups": [], "confidence": 1}


def judge_batch(
    model,
    prompt_texts: list[str],
    delay_seconds: float = 0.0,
    max_retries: int = 5,
    system_and_suffix: str = None,
) -> list[dict]:
    """Call API once with N prompts; return list of N parsed results. Falls back to per-item on parse failure."""
    if not prompt_texts:
        return []
    system = (system_and_suffix if system_and_suffix is not None else JUDGE_SYSTEM) + JUDGE_BATCH_SUFFIX
    if len(prompt_texts) == 1:
        return [judge_one(model, prompt_texts[0], delay_seconds=delay_seconds, max_retries=max_retries, system_and_suffix=system)]
    numbered = "\n\n".join(f"--- Prompt {i+1} ---\n{p}" for i, p in enumerate(prompt_texts))
    user_content = f"Analyze the following {len(prompt_texts)} prompts. Return a JSON ARRAY of {len(prompt_texts)} objects (same structure as above), one per prompt, in order.\n\n{numbered}\n\nReturn ONLY the JSON array, no other text."
    # When using cached model, system is in cache; we send batch suffix + user content
    if system_and_suffix is not None and system_and_suffix == "":
        batch_content = JUDGE_BATCH_SUFFIX.strip() + "\n\n" + user_content
    else:
        batch_content = system + "\n\n" + user_content
    attempt = 0
    consecutive_429 = 0
    while attempt < max_retries:
        try:
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            response = model.generate_content(batch_content)
            consecutive_429 = 0
            text = response.text if response.text else ""
            results = _parse_judge_response_array(text, len(prompt_texts))
            if any("error" in r for r in results) and attempt < max_retries - 1:
                raise ValueError("Batch parse had errors, retrying")
            return results
        except Exception as e:
            if is_rate_limit(e):
                consecutive_429 += 1
                sleep_secs = int(RATE_LIMIT_SLEEP * (1.5 ** min(consecutive_429 - 1, 2)))
                print(f"  Rate limit reached. Sleeping {sleep_secs}s to reset...")
                time.sleep(sleep_secs)
                continue
            if attempt == max_retries - 1:
                return [
                    {"applicable_groups": [], "confidence": 1, "error": str(e)}
                    for _ in prompt_texts
                ]
            time.sleep(2 ** attempt)
            attempt += 1
    return [{"applicable_groups": [], "confidence": 1} for _ in prompt_texts]


def main():
    ap = argparse.ArgumentParser(description="Judge prompts with Gemini (all dimensions + aesthetics)")
    ap.add_argument("--input", required=True, help="Input JSONL path (must have 'prompt' field)")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--prompt-field", default="prompt", help="Field name for prompt text (default: prompt)")
    ap.add_argument("--max-prompts", type=int, default=None, help="Max prompts to process (default: all)")
    ap.add_argument("--batch-size", type=int, default=5, help="Prompts per API call (5 recommended for 3.5k runs; 1=no batching)")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"Seconds between API calls (default {DEFAULT_DELAY} from rate_limits; set GEMINI_RPM_LIMIT / GEMINI_BUFFER to tune)")
    ap.add_argument("--model", default="gemini-3.1-flash-lite-preview", help="Gemini model")
    ap.add_argument("--resume", action="store_true", help="Skip lines already in output file")
    ap.add_argument("--use-cache", action="store_true", help="Use context cache for system prompt (reduces TPM, helps rate limits)")
    ap.add_argument("--no-cache", action="store_true", help="Disable context cache (default: no cache unless --use-cache)")
    args = ap.parse_args()

    genai.configure(api_key=get_api_key())
    use_cache = args.use_cache and not args.no_cache
    model = None
    cached_system_suffix = None  # None = use JUDGE_SYSTEM; "" = system already in cache

    if use_cache:
        try:
            import datetime
            caching = getattr(genai, "caching", None)
            if caching is not None and hasattr(caching, "CachedContent"):
                model_name = args.model if args.model.startswith("models/") else f"models/{args.model}"
                system_cache = caching.CachedContent.create(
                    model=model_name,
                    display_name="prompt_judge_instructions",
                    system_instruction=JUDGE_SYSTEM,
                    ttl=datetime.timedelta(hours=_CACHE_TTL_HOURS),
                )
                model = genai.GenerativeModel.from_cached_content(cached_content=system_cache)
                cached_system_suffix = ""
                print("Using context cache for judge system prompt (reduces TPM).")
        except Exception as e:
            print(f"Context cache not available ({e}). Using standard model.")
            use_cache = False

    if model is None:
        model = genai.GenerativeModel(args.model)
        cached_system_suffix = None

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if args.max_prompts is not None:
        records = records[: args.max_prompts]
    total = len(records)
    print(f"Input: {input_path} -> {total} records", flush=True)

    start_idx = 0
    if args.resume and output_path.exists() and output_path.stat().st_size > 0:
        with open(output_path, "r", encoding="utf-8") as f:
            # Count only non-empty lines so trailing newline or one bad line doesn't break resume
            start_idx = sum(1 for line in f if line.strip())
        if start_idx > total:
            print(f"Warning: output has {start_idx} lines but input has {total}. Check that --input is the same file used for the original run.", flush=True)
        if start_idx > 0:
            print(f"Resuming: skipping first {start_idx} already in output (-> {len(records)} left to judge)")
            records = records[start_idx:]
            # Progress denominator = expected final count (lines already in output + remaining)
            total = start_idx + len(records)

    if not records:
        print("Nothing to process.")
        return

    def apply_result(rec: dict, result: dict) -> None:
        rec["applicable_groups"] = result.get("applicable_groups", [])
        rec["motion_stability"] = result.get("motion_stability", {})
        rec["logic_physics"] = result.get("logic_physics", {})
        rec["instruction_adherence"] = result.get("instruction_adherence", {})
        rec["aesthetic_quality"] = result.get("aesthetic_quality", {})
        rec["categories"] = result.get("categories", [])
        rec["confidence"] = result.get("confidence", 7)
        rec["needs_review"] = result.get("needs_review", False)
        rec["review_reason"] = result.get("review_reason")
        if "error" in result:
            rec["judge_error"] = result["error"]

    prompt_field = args.prompt_field
    batch_size = max(1, args.batch_size)
    mode = "a" if start_idx > 0 else "w"
    with open(output_path, mode, encoding="utf-8") as out:
        if batch_size <= 1:
            for i, rec in enumerate(records):
                prompt_text = rec.get(prompt_field, rec.get("prompt", ""))
                if not prompt_text:
                    rec["motion_stability"] = {}
                    rec["logic_physics"] = {}
                    rec["instruction_adherence"] = {}
                    rec["aesthetic_quality"] = {}
                    rec["applicable_groups"] = []
                    rec["confidence"] = 1
                    rec["needs_review"] = False
                    rec["review_reason"] = None
                    out.write(json.dumps(_ordered_record(rec), ensure_ascii=False) + "\n")
                    out.flush()
                    continue
                result = judge_one(model, prompt_text, delay_seconds=args.delay, system_and_suffix=cached_system_suffix)
                apply_result(rec, result)
                out.write(json.dumps(_ordered_record(rec), ensure_ascii=False) + "\n")
                out.flush()
                if (i + 1) % 10 == 0:
                    print(f"  {start_idx + i + 1}/{total} judged")
        else:
            idx = 0
            while idx < len(records):
                chunk = records[idx : idx + batch_size]
                prompts_in_chunk = []
                prompt_indices = []
                for j, rec in enumerate(chunk):
                    pt = rec.get(prompt_field, rec.get("prompt", ""))
                    if not pt:
                        rec["motion_stability"] = {}
                        rec["logic_physics"] = {}
                        rec["instruction_adherence"] = {}
                        rec["aesthetic_quality"] = {}
                        rec["applicable_groups"] = []
                        rec["confidence"] = 1
                        rec["needs_review"] = False
                        rec["review_reason"] = None
                    else:
                        prompts_in_chunk.append(pt)
                        prompt_indices.append(j)
                if prompts_in_chunk:
                    results = judge_batch(
                        model,
                        prompts_in_chunk,
                        delay_seconds=args.delay,
                        system_and_suffix=cached_system_suffix,
                    )
                    for k, j in enumerate(prompt_indices):
                        if k < len(results):
                            apply_result(chunk[j], results[k])
                for rec in chunk:
                    out.write(json.dumps(_ordered_record(rec), ensure_ascii=False) + "\n")
                    out.flush()
                idx += len(chunk)
                done = start_idx + idx
                if done % (batch_size * 5) == 0 or done == total:
                    print(f"  {done}/{total} judged")

    print(f"Done. Output: {output_path} ({start_idx + len(records)} records)")


if __name__ == "__main__":
    main()
