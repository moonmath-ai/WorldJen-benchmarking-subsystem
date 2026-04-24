#!/usr/bin/env python3
"""
Enhance prompts via Gemini API: dimension-aware improvement (same logic as phase4, via API).
Input: Judged JSONL (with dimension scores). Output: same + enhanced_prompt.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import google.generativeai as genai

from config import get_api_key
from rate_limits import RATE_LIMIT_SLEEP, is_rate_limit, DEFAULT_DELAY

THRESHOLD = 7  # dimensions with suitability < this are "weak"


def get_weaknesses(rec: dict) -> dict:
    """Return dict group -> list of dimension names with suitability < THRESHOLD (non-null only)."""
    weaknesses = {}
    for group in ("motion_stability", "logic_physics", "instruction_adherence", "aesthetic_quality"):
        data = rec.get(group) or {}
        weak = []
        for key, val in data.items():
            if not key.endswith("_suitability"):
                continue
            if val is None:
                continue
            try:
                if int(val) < THRESHOLD:
                    weak.append(key.replace("_suitability", ""))
            except (TypeError, ValueError):
                pass
        if weak:
            weaknesses[group] = weak
    return weaknesses


# Static system instruction for enhancement (used with context cache or concatenated with user content)
ENHANCE_SYSTEM = """You are a video generation prompt expert. Enhance the given video prompt by:

1. **Fixing language**: Correct grammar, spelling, improve coherence.
2. **Addressing weak dimensions**: Add specific elements to boost weak dimensions (listed in the user message).
3. **Preserving core theme**: Keep the main subject and concept EXACTLY as intended.

**Guidelines for weak dimensions (use specific, stress-testing details):**
- motion_smoothness: Add fast/complex motion (running, spinning, fast-moving objects).
- temporal_flickering: Add complex textures (water, fire, hair, reflective surfaces). Specify high-frequency details like "individual water droplets", "strands of silk hair", or "fine mesh textures" which are prone to flickering.
- inertial_consistency: Add velocity changes (falling, stopping, throwing/catching, sliding). Describe the specific transition, such as "initial drag", "sudden deceleration", or "natural pendulous swing".
- physical_mechanics: Add falling, bouncing, colliding, or fluid interactions.
- object_permanence: Add occlusion (objects going behind/under things).
- human_fidelity: Add close-ups of hands/faces, complex poses (only if humans already in prompt). Focus on anatomical stress points: "knuckle articulation during gripping", "subtle facial muscle movements", or "realistic foot-to-ground contact".
- dynamic_degree: Add state transformations (melting, dancing, running) not just camera movement.
- semantic_adherence: Add specific colors, attributes, objects that must match.
- spatial_relationship: Specify precise positions (on top of, inside, holding, under).
- semantic_drift: Add multi-stage or sustained activities.
- composition_framing: Specify shot type or composition (close-up, wide shot, POV, rule of thirds, framing).
- lighting_volumetric: Add lighting detail (neon, sunset, volumetric fog, multiple lights, shadows).
- color_harmony: Add color palette (monochromatic, vibrant, pastel, specific color grading).
- structural_gestalt: Add detail so elements feel unified (blending of textures, consistent style, coherent world).

**Constraints:**
- DO NOT add new characters/entities not in the original.
- Only enhance elements already present or naturally implied.
- Keep it concise (3–5 sentences, under 1000 characters).

Return a JSON object with exactly one key: "enhanced_prompt" (string). Example: {"enhanced_prompt": "Your enhanced prompt text here."}"""


def _build_enhance_user_content(original_prompt: str, weaknesses: dict, categories: list) -> str:
    """Variable part of the enhance request (original prompt, categories, weak dimensions list)."""
    if not weaknesses:
        intro = "The prompt is already well-balanced. Focus on grammar/spelling and minor clarity."
    else:
        lines = ["**Weak dimensions to address:**"]
        for group, dims in weaknesses.items():
            name = group.replace("_", " ").title()
            dim_str = ", ".join(d.replace("_", " ").title() for d in dims)
            lines.append(f"  • {name}: {dim_str}")
        intro = "\n".join(lines)
    return f"""**Original prompt:**
{original_prompt}

**Categories:** {', '.join(categories or [])}

{intro}

Apply the guidelines above and return ONLY a JSON object with key "enhanced_prompt"."""


def build_enhance_prompt(original_prompt: str, weaknesses: dict, categories: list) -> str:
    """Full prompt when not using cache (system + user content)."""
    return ENHANCE_SYSTEM + "\n\n" + _build_enhance_user_content(original_prompt, weaknesses, categories)


def _is_unchanged(original: str, enhanced: str) -> bool:
    """True if enhanced is substantially the same as original (soft refusal or no-op)."""
    if not original and not enhanced:
        return True
    a = " ".join(original.strip().lower().split())
    b = " ".join(enhanced.strip().lower().split())
    if a == b:
        return True
    # Allow tiny edits (e.g. punctuation) but flag when almost identical
    if len(a) < 20 or len(b) < 20:
        return a == b
    max_len = max(len(a), len(b))
    if max_len == 0:
        return True
    # Levenshtein-style: if edit distance is small relative to length, treat as unchanged
    # Simple check: one string is a prefix of the other and lengths are close
    if abs(len(a) - len(b)) <= 10 and (a in b or b in a):
        return True
    return False


def enhance_one(
    model,
    rec: dict,
    delay_seconds: float = 0.0,
    max_retries: int = 5,
    use_json_mode: bool = True,
    use_cached_system: bool = False,
    temperature: float = 0.8,
) -> tuple[str, bool]:
    """Returns (enhanced_prompt_text, enhancement_unchanged). enhancement_unchanged=True means soft refusal or no-op."""
    prompt_text = rec.get("prompt", "")
    weaknesses = get_weaknesses(rec)
    categories = rec.get("categories", [])
    user_content = _build_enhance_user_content(prompt_text, weaknesses, categories)
    if use_cached_system:
        content = user_content
    else:
        content = ENHANCE_SYSTEM + "\n\n" + user_content
    generation_config = {"response_mime_type": "application/json", "temperature": temperature} if use_json_mode else {"temperature": temperature}
    attempt = 0
    consecutive_429 = 0
    while attempt < max_retries:
        try:
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            response = model.generate_content(
                content,
                generation_config=generation_config,
            )
            consecutive_429 = 0  # success: reset so next 429 gets base sleep
            text = (response.text or "").strip()
            if use_json_mode and text:
                data = json.loads(text)
                out = (data.get("enhanced_prompt") or "").strip()[:2000] or ""
            else:
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1].replace('\\"', '"')
                out = text[:2000]
            unchanged = _is_unchanged(prompt_text, out)
            if unchanged and out:
                print("  [Soft refusal or unchanged enhancement]")
            return (out, unchanged)
        except (json.JSONDecodeError, TypeError, KeyError):
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                attempt += 1
                continue
            return ("", True)
        except Exception as e:
            if is_rate_limit(e):
                consecutive_429 += 1
                # Exponential backoff: 120s, then 180s, then 270s so the bucket can fully reset
                sleep_secs = int(RATE_LIMIT_SLEEP * (1.5 ** min(consecutive_429 - 1, 2)))
                print(f"  Rate limit reached. Sleeping {sleep_secs}s to reset...", flush=True)
                if consecutive_429 == 1:
                    print("  Tip: If you hit 429 on the first request, wait ~1 min and re-run with --start-delay 65", flush=True)
                time.sleep(sleep_secs)
                continue  # retry without incrementing attempt
            if attempt == max_retries - 1:
                return ("", True)
            time.sleep(2 ** attempt)
            attempt += 1
    return ("", True)


def main():
    ap = argparse.ArgumentParser(description="Enhance prompts with Gemini (dimension-aware)")
    ap.add_argument("--input", required=True, help="Judged JSONL (with dimension scores)")
    ap.add_argument("--output", required=True, help="Output JSONL (adds enhanced_prompt)")
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Seconds between API calls (default from rate_limits)")
    ap.add_argument("--model", default="gemini-3.1-flash-lite-preview", help="Gemini model")
    ap.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature for more diverse enhancements (default 0.8)")
    ap.add_argument("--use-cache", action="store_true", help="Use context cache for system instructions (reduces TPM)")
    ap.add_argument("--resume", action="store_true", help="Skip records that already have enhanced_prompt")
    ap.add_argument("--start-delay", type=float, default=0, help="Seconds to wait before first API call (e.g. 65 if you hit 429 immediately after another run)")
    args = ap.parse_args()

    genai.configure(api_key=get_api_key())
    model = None
    use_cached_system = False
    if args.use_cache:
        try:
            import datetime
            caching = getattr(genai, "caching", None)
            if caching is not None and hasattr(caching, "CachedContent"):
                model_name = args.model if args.model.startswith("models/") else f"models/{args.model}"
                instruction_cache = caching.CachedContent.create(
                    model=model_name,
                    display_name="enhancement_instructions",
                    system_instruction=ENHANCE_SYSTEM,
                    ttl=datetime.timedelta(hours=1),
                )
                model = genai.GenerativeModel.from_cached_content(cached_content=instruction_cache)
                use_cached_system = True
                print("Using context cache for enhancement instructions (reduces TPM).")
        except Exception as e:
            print(f"Context cache not available ({e}). Using standard model.")
    if model is None:
        model = genai.GenerativeModel(args.model)

    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.is_dir() or (output_path.exists() and output_path.is_dir()):
        output_path = output_path / "enhanced.jsonl"
    elif not output_path.suffix and not output_path.exists():
        # path looks like a dir (e.g. "out" or "scripts/out")
        output_path = output_path / "enhanced.jsonl"
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

    start_idx = 0
    if args.resume and output_path.exists() and output_path.stat().st_size > 0:
        with open(output_path, "r", encoding="utf-8") as f:
            start_idx = sum(1 for _ in f)
        if start_idx > 0:
            print(f"Resuming: skipping first {start_idx}")
            records = records[start_idx:]

    if not records:
        print("Nothing to process.")
        return

    n_to_do = sum(1 for r in records if not r.get("enhanced_prompt"))
    print(f"Total records: {total} (resumed from {start_idx}). To enhance: {n_to_do}")
    if args.delay > 0:
        eta_min = (n_to_do * args.delay) / 60.0
        print(f"At {args.delay}s delay: ~{eta_min:.0f} min for this run (excluding rate-limit waits)")

    if args.start_delay > 0:
        print(f"Waiting {args.start_delay:.0f}s before first request (--start-delay)...", flush=True)
        time.sleep(args.start_delay)

    mode = "a" if start_idx > 0 else "w"
    with open(output_path, mode, encoding="utf-8") as out:
        for i, rec in enumerate(records):
            current = start_idx + i + 1
            if rec.get("enhanced_prompt"):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                continue
            print(f"  [{current}/{start_idx + total}] Enhancing...", flush=True)
            enhanced, unchanged = enhance_one(
                model,
                rec,
                delay_seconds=args.delay,
                use_cached_system=use_cached_system,
                temperature=args.temperature,
            )
            rec["enhanced_prompt"] = enhanced or rec.get("prompt", "")
            rec["enhancement_unchanged"] = unchanged
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  [{current}/{start_idx + total}] Done.", flush=True)

    n_unchanged = sum(1 for r in records if r.get("enhancement_unchanged"))
    if n_unchanged:
        print(f"  Enhancement unchanged (soft refusal/no-op): {n_unchanged}")
    print(f"Done. Output: {output_path} ({start_idx + len(records)} records)")


if __name__ == "__main__":
    main()
