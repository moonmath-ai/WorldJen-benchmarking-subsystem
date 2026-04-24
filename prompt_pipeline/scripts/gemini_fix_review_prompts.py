#!/usr/bin/env python3
"""
Fix prompts flagged for review (needs_review=true) using Gemini API.

Rules applied based on review_reason:
- Famous person / real person → change to generic "a man" / "a woman" (or "a person").
- Objectionable content (violence, inappropriate) → rewrite to non-objectionable while keeping theme.
- Copyright (e.g. Disney, brand) → replace with generic style/description.
Do not change the general theme of the prompt.

Output: Same JSONL schema as input. Fixed prompts get prompt updated, needs_review=false,
review_fix_applied=true. If fix unsuccessful or model declines, needs_manual_review=true,
review_fix_note=reason.

Usage:
  python scripts/gemini_fix_review_prompts.py --input out/judged.jsonl --output out/judged_review_fixed.jsonl
  # If interrupted, re-run with same output as input to resume: --input out/judged_review_fixed.jsonl --output out/judged_review_fixed.jsonl --resume
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
from rate_limits import RATE_LIMIT_SLEEP, is_rate_limit, get_safe_delay, DEFAULT_DELAY

SYSTEM = """You are a video prompt editor. Your job is to fix a prompt that was flagged for review so it stays usable and non-problematic while keeping the same general theme and intent.

**Rules:**
1. **Famous or real person**: Replace the specific person with a generic description (e.g. "a man", "a woman", "a person", "a character with ..."). Do not use real names or identifiable real people.
2. **Objectionable content** (violence, inappropriate, harmful): Rewrite only the objectionable part to something non-objectionable. Keep the same scene type and mood (e.g. tension without graphic violence, drama without harm).
3. **Copyright / brand** (e.g. Disney, specific IP, brand names): Replace with a generic style or description (e.g. "animated style", "fantasy style", "vibrant cartoon style"). Do not mention the brand or copyrighted character names.
4. **Keep**: Same general theme, setting, action, and mood. Same length and level of detail. Only change what is necessary to address the flag.

Return a JSON object with:
- "fixed_prompt" (string): The revised prompt text. If you cannot fix it safely, return the original prompt and set "fix_applied" to false.
- "fix_applied" (boolean): true if you made a revision that addresses the issue, false if you left the prompt unchanged (flag for manual review).
- "reason_if_not" (string, optional): If fix_applied is false, briefly why (e.g. "ambiguous objectionable content").
"""

# When processing multiple prompts, return a JSON array of such objects in the same order.
FIX_BATCH_SUFFIX = "\n\nFor multiple prompts below, return a JSON ARRAY of objects (same keys: fixed_prompt, fix_applied, reason_if_not), one per prompt, in order."


def _parse_fix_result(text: str, fallback_prompt: str) -> dict:
    """Parse single fix JSON. Returns dict with fixed_prompt, fix_applied, reason_if_not."""
    data = json.loads(text)
    fixed = (data.get("fixed_prompt") or fallback_prompt).strip()
    applied = bool(data.get("fix_applied", False))
    reason = data.get("reason_if_not")
    return {"fixed_prompt": fixed or fallback_prompt, "fix_applied": applied, "reason_if_not": reason}


def fix_one(model, prompt_text: str, review_reason: str, delay_seconds: float = 5.0, max_retries: int = 5) -> dict:
    """Call Gemini to fix the prompt. Returns {"fixed_prompt": str, "fix_applied": bool, "reason_if_not": str or None}."""
    user = f"""**Flagged prompt:**
{prompt_text}

**Reason for review:** {review_reason or "Unspecified"}

Apply the rules above and return a JSON object with keys: "fixed_prompt", "fix_applied", and optionally "reason_if_not".
Return ONLY valid JSON, no other text."""

    generation_config = {"response_mime_type": "application/json"}
    attempt = 0
    consecutive_429 = 0
    while attempt < max_retries:
        try:
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            response = model.generate_content(
                SYSTEM + "\n\n" + user,
                generation_config=generation_config,
            )
            consecutive_429 = 0
            text = (response.text or "").strip()
            if not text:
                return {"fixed_prompt": prompt_text, "fix_applied": False, "reason_if_not": "empty_response"}
            return _parse_fix_result(text, prompt_text)
        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                return {"fixed_prompt": prompt_text, "fix_applied": False, "reason_if_not": f"parse_error: {e}"}
            time.sleep(2 ** attempt)
            attempt += 1
        except Exception as e:
            if is_rate_limit(e):
                consecutive_429 += 1
                sleep_secs = int(RATE_LIMIT_SLEEP * (1.5 ** min(consecutive_429 - 1, 2)))
                print(f"  Rate limit reached. Sleeping {sleep_secs}s to reset...")
                time.sleep(sleep_secs)
                continue
            if attempt == max_retries - 1:
                return {"fixed_prompt": prompt_text, "fix_applied": False, "reason_if_not": str(e)}
            time.sleep(2 ** attempt)
            attempt += 1
    return {"fixed_prompt": prompt_text, "fix_applied": False, "reason_if_not": "max_retries"}


def _parse_fix_batch(text: str, fallback_prompts: list[str]) -> list[dict]:
    """Parse JSON array of fix results. Pads or trims to len(fallback_prompts)."""
    raw = json.loads(text)
    if not isinstance(raw, list):
        raw = [raw]
    out = []
    for i, p in enumerate(fallback_prompts):
        if i < len(raw) and isinstance(raw[i], dict):
            d = raw[i]
            out.append({
                "fixed_prompt": (d.get("fixed_prompt") or p).strip() or p,
                "fix_applied": bool(d.get("fix_applied", False)),
                "reason_if_not": d.get("reason_if_not"),
            })
        else:
            out.append({"fixed_prompt": p, "fix_applied": False, "reason_if_not": "missing_in_batch"})
    return out


def fix_batch(
    model,
    items: list[tuple[str, str]],
    delay_seconds: float = 5.0,
    max_retries: int = 5,
) -> list[dict]:
    """Call Gemini once for N (prompt, review_reason) pairs. Returns list of N result dicts."""
    if not items:
        return []
    if len(items) == 1:
        return [fix_one(model, items[0][0], items[0][1], delay_seconds=delay_seconds, max_retries=max_retries)]
    numbered = "\n\n".join(
        f"--- Item {i+1} ---\n**Prompt:** {p}\n**Reason for review:** {r or 'Unspecified'}"
        for i, (p, r) in enumerate(items)
    )
    user = f"""Fix the following {len(items)} flagged prompts. Apply the same rules to each.

{numbered}

Return a JSON ARRAY of {len(items)} objects, one per item, in order. Each object: "fixed_prompt", "fix_applied", "reason_if_not".
Return ONLY the JSON array, no other text."""

    generation_config = {"response_mime_type": "application/json"}
    fallback_prompts = [it[0] for it in items]
    attempt = 0
    consecutive_429 = 0
    while attempt < max_retries:
        try:
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            response = model.generate_content(
                SYSTEM + FIX_BATCH_SUFFIX + "\n\n" + user,
                generation_config=generation_config,
            )
            consecutive_429 = 0
            text = (response.text or "").strip()
            if not text:
                return [
                    {"fixed_prompt": p, "fix_applied": False, "reason_if_not": "empty_response"}
                    for p in fallback_prompts
                ]
            return _parse_fix_batch(text, fallback_prompts)
        except Exception as e:
            if is_rate_limit(e):
                consecutive_429 += 1
                sleep_secs = int(RATE_LIMIT_SLEEP * (1.5 ** min(consecutive_429 - 1, 2)))
                print(f"  Rate limit reached. Sleeping {sleep_secs}s to reset...")
                time.sleep(sleep_secs)
                continue
            if attempt == max_retries - 1:
                return [
                    {"fixed_prompt": p, "fix_applied": False, "reason_if_not": str(e)}
                    for p in fallback_prompts
                ]
            time.sleep(2 ** attempt)
            attempt += 1
    return [{"fixed_prompt": p, "fix_applied": False, "reason_if_not": "max_retries"} for p in fallback_prompts]


def main():
    ap = argparse.ArgumentParser(description="Fix prompts flagged for review (needs_review=true) via Gemini")
    ap.add_argument("--input", default=str(_root / "out" / "judged.jsonl"), help="Input judged JSONL")
    ap.add_argument("--output", default=str(_root / "out" / "judged_review_fixed.jsonl"), help="Output JSONL (same schema)")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Seconds between API calls (default from rate_limits)")
    ap.add_argument("--batch-size", type=int, default=1, help="Flagged prompts per API call (5 with delay 4.5 ≈ 13 RPM when under 15 RPM cap)")
    ap.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    ap.add_argument("--resume", action="store_true", help="Skip records already fixed (review_fix_applied or needs_manual_review)")
    args = ap.parse_args()

    genai.configure(api_key=get_api_key())
    model = genai.GenerativeModel(args.model)

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

    # Indices that need processing: needs_review true. If resume and input is previous output, skip already fixed.
    to_fix = []
    for i, rec in enumerate(records):
        if not rec.get("needs_review"):
            continue
        if args.resume and (rec.get("review_fix_applied") or rec.get("needs_manual_review")):
            continue
        to_fix.append(i)

    print(f"Total records: {len(records)}. Flagged for review (to process): {len(to_fix)}")

    # Process in chunks of batch_size; apply results to records.
    batch_size = max(1, args.batch_size)
    for chunk_start in range(0, len(to_fix), batch_size):
        chunk_indices = to_fix[chunk_start : chunk_start + batch_size]
        chunk_items = []
        for idx in chunk_indices:
            rec = records[idx]
            prompt_text = rec.get("prompt", "")
            review_reason = rec.get("review_reason") or "Unspecified"
            if not prompt_text:
                rec["needs_manual_review"] = True
                rec["review_fix_note"] = "empty prompt"
                continue
            chunk_items.append((idx, prompt_text, review_reason))

        if not chunk_items:
            continue

        indices = [x[0] for x in chunk_items]
        items = [(x[1], x[2]) for x in chunk_items]
        results = fix_batch(model, items, delay_seconds=args.delay)

        for idx, result in zip(indices, results):
            rec = records[idx]
            if result["fix_applied"]:
                rec["prompt"] = result["fixed_prompt"]
                rec["needs_review"] = False
                rec["review_reason"] = rec.get("review_reason")
                rec["review_fix_applied"] = True
            else:
                rec["needs_manual_review"] = True
                rec["review_fix_applied"] = False
                rec["review_fix_note"] = result.get("reason_if_not") or "fix not applied"

        k = chunk_start + len(chunk_items)
        if k % 10 == 0 or k == len(to_fix):
            print(f"  Processed {k}/{len(to_fix)} flagged prompts")

    # Write full output (same order as input, same schema)
    with open(output_path, "w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_fixed = sum(1 for r in records if r.get("review_fix_applied"))
    n_manual = sum(1 for r in records if r.get("needs_manual_review"))
    print(f"Done. Output: {output_path}")
    print(f"  Auto-fixed: {n_fixed}. Flagged for manual review: {n_manual}")


if __name__ == "__main__":
    main()
