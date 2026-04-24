#!/usr/bin/env python3
"""
Pass 2: Replace exact dimension terms in enhanced prompts with natural descriptive language.

Scans enhanced.jsonl offline to find prompts that contain any of the exact dimension
phrases (e.g. "motion smoothness", "object permanence"). Only those records are sent
to the Gemini API to rewrite the enhanced_prompt, replacing the technical term with
prompt-specific natural language. All other records are copied unchanged to the output.

Input:  data/prompts/prompts_enhanced_full.jsonl
Output: same format; enhanced_prompt updated only for fixed records.
"""

import os
import argparse
import json
import sys
import time
import warnings
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import google.generativeai as genai

from config import get_api_key
from rate_limits import RATE_LIMIT_SLEEP, is_rate_limit, DEFAULT_DELAY


def _response_text_safe(response) -> str:
    """Get text from Gemini response without assuming a single candidate (avoids response.parts ValueError)."""
    try:
        if hasattr(response, "text") and response.text:
            return response.text.strip()
    except (ValueError, AttributeError):
        pass
    if not getattr(response, "candidates", None) or len(response.candidates) == 0:
        return ""
    parts = []
    for c in response.candidates:
        if not getattr(c, "content", None) or not getattr(c.content, "parts", None):
            continue
        for p in c.content.parts:
            if hasattr(p, "text") and p.text:
                parts.append(p.text)
    return "".join(parts).strip() if parts else ""


# Exact phrases to detect in enhanced_prompt (case-insensitive)
DIMENSION_PHRASES = [
    "motion smoothness",
    "temporal flickering",
    "inertial consistency",
    "physical mechanics",
    "object permanence",
    "human fidelity",
    "dynamic degree",
    "semantic adherence",
    "spatial relationship",
    "semantic drift",
    "composition framing",
    "volumetric lighting",
    "color harmony",
    "structural gestalt",
]

# Definitions used in enhancements (from gemini_enhance.py) — for natural-language replacement
DIMENSION_DEFINITIONS = """
- motion smoothness: Fast/complex motion (running, spinning, fast-moving objects); fluid, non-jerky movement.
- temporal flickering: Complex textures prone to flickering (water, fire, hair, reflective surfaces); high-frequency details like individual water droplets, strands of hair, fine mesh textures.
- inertial consistency: Velocity changes (falling, stopping, throwing/catching, sliding); specific transitions like initial drag, sudden deceleration, or natural pendulous swing.
- physical mechanics: Falling, bouncing, colliding, or fluid interactions; believable physics.
- object permanence: Occlusion (objects going behind/under things); objects remaining visible or correctly hidden when passing behind others.
- human fidelity: Close-ups of hands/faces, complex poses; anatomical detail like knuckle articulation, subtle facial muscle movements, realistic foot-to-ground contact.
- dynamic degree: State transformations (melting, dancing, running), not just camera movement; things changing state or moving in substantive ways.
- semantic adherence: Specific colors, attributes, objects that must match the description; faithful to the requested details.
- spatial relationship: Precise positions (on top of, inside, holding, under); clear layout of elements.
- semantic drift: Multi-stage or sustained activities; consistency of meaning over time.
- composition framing: Shot type or composition (close-up, wide shot, POV, rule of thirds, framing).
- volumetric lighting: Lighting detail (neon, sunset, volumetric fog, multiple lights, shadows); visible light beams or atmospheric glow.
- color harmony: Color palette (monochromatic, vibrant, pastel, specific color grading); cohesive colors.
- structural gestalt: Elements feel unified (blending of textures, consistent style, coherent world); overall visual coherence.
"""

PASS2_SYSTEM = """You are a video generation prompt editor. Your task is to replace technical dimension terms with natural, descriptive language.

**CRITICAL:** The user will list exact phrases that MUST NOT appear in your output. You must replace every occurrence of those phrases with plain, scene-specific descriptive language that conveys the same meaning (use the dimension definitions to understand each term). Your final enhanced_prompt must NOT contain any of the listed phrases—no exceptions.

**Rules:**
1. Identify every occurrence of the listed phrases in the given prompt.
2. Replace each with natural language a director or artist would use (e.g. instead of "object permanence" write something like "the object stays visible or correctly hidden when passing behind others"; instead of "structural gestalt" write "unified look" or "coherent, consistent style" or similar—tailored to the scene).
3. Do not add new scenes or change the theme. Only reword the dimension terms.
4. Output a single JSON object with one key: "enhanced_prompt" (string). No other keys.
"""


def find_dimension_phrases_in_text(text: str) -> list[str]:
    """Return list of dimension phrases that appear in text (case-insensitive)."""
    if not text:
        return []
    lower = text.lower()
    return [p for p in DIMENSION_PHRASES if p in lower]


def build_pass2_user_content(enhanced_prompt: str, found_phrases: list[str], retry_still_contained: list[str] | None = None) -> str:
    forbidden = " / ".join(f'"{p}"' for p in found_phrases)
    retry_note = ""
    if retry_still_contained:
        retry_note = f"\n**RETRY:** Your previous output still contained these phrases (forbidden): {', '.join(retry_still_contained)}. Remove them entirely and use only natural descriptive wording.\n"
    return f"""**Dimension definitions (for reference):**
{DIMENSION_DEFINITIONS}
{retry_note}
**Enhanced prompt (rewrite so it no longer contains the forbidden phrases):**
{enhanced_prompt}

**FORBIDDEN—these exact phrases must NOT appear in your output. Replace each with natural language:** {forbidden}

Return only a JSON object with key "enhanced_prompt" containing the revised text."""


def fix_one(
    model,
    rec: dict,
    delay_seconds: float = 0.0,
    max_retries: int = 5,
) -> tuple[str, bool, str]:
    """
    Call Gemini to rewrite rec["enhanced_prompt"] so dimension phrases are replaced with natural language.
    Returns (fixed_prompt, success, reason). reason is one of:
      "model" - API returned text with no dimension phrases (success)
      "failed: <cause>" - not fixed; enhanced_prompt left unchanged. cause = empty_response | json_error | api_error:... | max_retries | model_still_contained_phrases | no_enhanced_prompt
    """
    enhanced = rec.get("enhanced_prompt", "")
    if not enhanced:
        return (rec.get("prompt", ""), False, "failed: no_enhanced_prompt")
    found = find_dimension_phrases_in_text(enhanced)
    if not found:
        return (enhanced, True, "model")
    generation_config = {"response_mime_type": "application/json", "temperature": 0.2}
    retry_still_contained = None
    last_error: str | None = None
    for pass_attempt in range(2):  # normal pass + one retry if output still has phrases
        content = PASS2_SYSTEM + "\n\n" + build_pass2_user_content(
            enhanced, found, retry_still_contained=retry_still_contained
        )
        attempt = 0
        consecutive_429 = 0
        while attempt < max_retries:
            try:
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
                response = model.generate_content(content, generation_config=generation_config)
                consecutive_429 = 0
                text = _response_text_safe(response)
                if not text:
                    reason = "failed: empty_response"
                    print(f"  [{reason}]", flush=True)
                    return (enhanced, False, reason)
                data = json.loads(text)
                out = (data.get("enhanced_prompt") or "").strip()[:2000] or enhanced
                still_contained = find_dimension_phrases_in_text(out)
                if not still_contained:
                    return (out, True, "model")
                if pass_attempt == 0:
                    retry_still_contained = still_contained
                    if retry_still_contained:
                        print(f"  Retry: output still had {retry_still_contained}", flush=True)
                    break  # retry with stricter prompt
                # second pass still had phrases; not fixed, keep original
                reason = "failed: model_still_contained_phrases"
                print(f"  [{reason}]", flush=True)
                return (enhanced, False, reason)
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                last_error = f"failed: json_error_{type(e).__name__}"
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    attempt += 1
                    continue
                print(f"  [{last_error}]", flush=True)
                return (enhanced, False, last_error)
            except Exception as e:
                last_error = f"failed: api_error_{type(e).__name__}: {str(e)[:80]}"
                if is_rate_limit(e):
                    consecutive_429 += 1
                    sleep_secs = int(RATE_LIMIT_SLEEP * (1.5 ** min(consecutive_429 - 1, 2)))
                    print(f"  Rate limit. Sleeping {sleep_secs}s...", flush=True)
                    time.sleep(sleep_secs)
                    continue
                if attempt == max_retries - 1:
                    print(f"  [{last_error}]", flush=True)
                    return (enhanced, False, last_error)
                time.sleep(2**attempt)
                attempt += 1
    reason = "failed: max_retries"
    print(f"  [{reason}]", flush=True)
    return (enhanced, False, reason)


def main():
    ap = argparse.ArgumentParser(
        description="Pass 2: Replace dimension terms in enhanced prompts with natural language (Gemini only for records that need it)"
    )
    ap.add_argument(
        "--input",
        default=str(DATA_ROOT / "prompts/prompts_enhanced_full.jsonl"),
        help="Input enhanced JSONL",
    )
    ap.add_argument(
        "--output",
        default=str(DATA_ROOT / "prompts/prompts_enhanced_pass2.jsonl"),
        help="Output JSONL (same format as input)",
    )
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Seconds between API calls")
    ap.add_argument("--model", default="gemini-3.1-flash-lite-preview", help="Gemini model (same as judge/fix)")
    ap.add_argument("--dry-run", action="store_true", help="Only scan and report which records need fixing; do not call API or write output")
    ap.add_argument("--resume", action="store_true", help="Skip records already written to output (append remaining)")
    ap.add_argument("--start-delay", type=float, default=0, help="Seconds to wait before first API call")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    # Offline scan: which records need fixing?
    need_fix = []
    for i, rec in enumerate(records):
        ep = rec.get("enhanced_prompt", "")
        phrases = find_dimension_phrases_in_text(ep)
        if phrases:
            need_fix.append((i, rec, phrases))

    print(f"Total records: {len(records)}")
    print(f"Records with dimension phrases (need fix): {len(need_fix)}")
    if need_fix:
        for i, rec, phrases in need_fix[:5]:
            print(f"  e.g. {rec.get('prompt_id', i)}: {phrases}")
        if len(need_fix) > 5:
            print(f"  ... and {len(need_fix) - 5} more")

    if args.dry_run:
        print("Dry run: no API calls, no output written.")
        return

    if not need_fix:
        print("No records to fix. Copying input to output as-is.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out:
            for rec in records:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Output: {output_path}")
        return

    # Resume: skip records already written to output
    start_idx = 0
    if args.resume and output_path.exists() and output_path.stat().st_size > 0:
        with open(output_path, "r", encoding="utf-8") as f:
            start_idx = sum(1 for _ in f)
        if start_idx >= len(records):
            print(f"Output already has {start_idx} records (>= {len(records)}). Nothing to do.")
            return
        if start_idx > 0:
            print(f"Resuming: skipping first {start_idx} records (already in output).")

    genai.configure(api_key=get_api_key())
    model = genai.GenerativeModel(args.model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.start_delay > 0:
        print(f"Waiting {args.start_delay:.0f}s before first request...", flush=True)
        time.sleep(args.start_delay)

    # Write output incrementally (append when resuming)
    fixed_count = 0
    reason_counts: dict[str, int] = {}
    need_fix_idx = {idx for idx, _, _ in need_fix}
    mode = "a" if start_idx > 0 else "w"
    with open(output_path, mode, encoding="utf-8") as out:
        for i in range(start_idx, len(records)):
            rec = records[i]
            if i not in need_fix_idx:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                continue
            prompt_id = rec.get("prompt_id", i)
            print(f"  [{i + 1}/{len(records)}] Fixing [{prompt_id}] ...", flush=True)
            fixed_text, ok, reason = fix_one(model, rec, delay_seconds=args.delay)
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            # pass2_source: model (fixed) | failed (not fixed; enhanced_prompt unchanged)
            if reason == "model":
                pass2_source = "model"
                pass2_failure_reason = None
            else:
                pass2_source = "failed"
                pass2_failure_reason = reason  # e.g. "failed: empty_response"
            new_rec = {
                **rec,
                "enhanced_prompt": fixed_text,
                "pass2_fixed": ok,
                "pass2_source": pass2_source,
                "pass2_failure_reason": pass2_failure_reason,
            }
            if ok:
                fixed_count += 1
            out.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  [{i + 1}/{len(records)}] Done [{prompt_id}] ({reason})", flush=True)

    print(f"Done. Fixed {fixed_count} in this run. Output: {output_path} ({len(records)} records total)")
    if reason_counts:
        print("Pass2 outcome breakdown:", " | ".join(f"{r}: {c}" for r, c in sorted(reason_counts.items())))


if __name__ == "__main__":
    main()
