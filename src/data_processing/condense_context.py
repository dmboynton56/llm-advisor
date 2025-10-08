# src/data_processing/condense_context.py
#!/usr/bin/env python3

import os
import sys
import json
import re
import time
from datetime import datetime
import pytz
from dotenv import load_dotenv
import google.generativeai as genai

# allow imports relative to project root if needed (prompts/, config/, etc.)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ET_ZONE = pytz.timezone("US/Eastern")

# --- Add these helpers near the top (after imports) ---

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _parse_json_loose(text: str):
    # best-effort: strip fences, normalize quotes, remove trailing commas
    s = _strip_code_fences(text)
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r",\s*([\}\]])", r"\1", s)
    # extract outermost object if necessary
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]
    return json.loads(s)

def _validate_and_coerce(obj: dict, day_dash: str) -> dict:
    # Ensure required top-level keys exist; fill defaults
    from datetime import timezone
    if not isinstance(obj, dict):
        obj = {}
    if "ts" not in obj or not isinstance(obj["ts"], str):
        obj["ts"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if "day" not in obj or not isinstance(obj["day"], str):
        obj["day"] = day_dash
    obj.setdefault("macro", {})
    if not isinstance(obj["macro"], dict):
        obj["macro"] = {}

    # consensus: list of {s, bias, conf}
    raw_cons = obj.get("consensus", [])
    if not isinstance(raw_cons, list):
        raw_cons = []
    cons = []
    for it in raw_cons:
        # accept either tuple-like or object-like
        if isinstance(it, (list, tuple)) and len(it) >= 3:
            s, bias, conf = it[0], it[1], it[2]
        elif isinstance(it, dict):
            s, bias, conf = it.get("s"), it.get("bias"), it.get("conf")
        else:
            continue
        try:
            s = str(s)
            bias = int(bias)
            if bias not in (-1, 0, 1):
                continue
            conf = float(conf)
            # clamp 0..1
            conf = 0.0 if conf < 0 else (1.0 if conf > 1 else conf)
            cons.append({"s": s, "bias": bias, "conf": conf})
        except Exception:
            continue
    obj["consensus"] = cons

    # alerts: list of {s, bias, conf, why[]}
    raw_alerts = obj.get("alerts", [])
    if not isinstance(raw_alerts, list):
        raw_alerts = []
    alerts = []
    for it in raw_alerts:
        if not isinstance(it, dict):
            continue
        try:
            s = str(it.get("s"))
            bias = int(it.get("bias"))
            conf = float(it.get("conf"))
            why = it.get("why") or []
            if bias not in (-1, 0, 1):
                continue
            if not isinstance(why, list):
                why = [str(why)]
            why = [str(w) for w in why if w is not None]
            conf = 0.0 if conf < 0 else (1.0 if conf > 1 else conf)
            if s:
                alerts.append({"s": s, "bias": bias, "conf": conf, "why": why})
        except Exception:
            continue
    obj["alerts"] = alerts

    return obj


def get_file_paths():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    today_str = datetime.now(ET_ZONE).strftime('%Y-%m-%d')
    daily_processed = os.path.join(project_root, 'data', 'daily_news', today_str, 'processed')
    os.makedirs(daily_processed, exist_ok=True)
    return {
        "synth_input": os.path.join(daily_processed, 'processed_briefing.json'),
        "prompt_template": os.path.join(project_root, 'prompts', 'prompts.json'),
        "out_path": os.path.join(daily_processed, 'session_context.json'),
        "day_dash": today_str,
    }

def construct_prompt(paths):
    try:
        with open(paths['synth_input'], 'r', encoding='utf-8') as f:
            synth = json.load(f)
    except FileNotFoundError:
        print(f"Error: synthesized file not found at {paths['synth_input']}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding synthesized JSON: {e}")
        return None

    tmpl = None
    try:
        with open(paths['prompt_template'], 'r', encoding='utf-8') as f:
            prompts = json.load(f)
            tmpl = (prompts.get('premarket_condense_prompt') or {}).get('content')
    except Exception:
        tmpl = None

    if not tmpl:
        # NOTE: consensus is now an array of objects, not 3-tuples.
        tmpl = (
            "You will receive a large premarket synthesized-bias JSON (multiple models). "
            "Condense it into a tiny JSON context for intraday use. "
            "Bias mapping: bearish=-1, choppy=0, bullish=1. Confidence in [0,1]. "
            "Return STRICT JSON only with:\n"
            "{"
            "\"ts\":\"<UTC ISO>\","
            "\"day\":\"YYYY-MM-DD\","
            "\"macro\":{\"vix\":num,\"dxy\":num,\"ust\":num,\"breadth\":int,\"adr\":num,\"smt\":0|1}  // omit keys you don't know"
            ",\"consensus\":[{\"s\":\"TICK\",\"bias\":-1|0|1,\"conf\":0-1}],"
            "\"alerts\":[{\"s\":\"TICK\",\"bias\":-1|0|1,\"conf\":0-1,\"why\":[\"tag\",\"tag\"]}]"
            "}"
        )

    payload = json.dumps(synth, separators=(",", ":"), ensure_ascii=False)
    prompt = f"{tmpl}\n\nINPUT_JSON:\n```json\n{payload}\n```"
    return prompt

# ---- Structured-output schema (Gemini-compatible subset) ----
def _response_schema():
    # Avoid tuple-style items and 'additionalProperties'. Keep types simple.
    return {
        "type": "object",
        "properties": {
            "ts":  {"type": "string"},          # ISO-UTC
            "day": {"type": "string"},          # YYYY-MM-DD
            "macro": {
                "type": "object",
                "properties": {
                    # Make all macro fields optional (omit if unknown).
                    "vix":     {"type": "number"},
                    "dxy":     {"type": "number"},
                    "ust":     {"type": "number"},
                    "breadth": {"type": "integer"},
                    "adr":     {"type": "number"},
                    "smt":     {"type": "integer", "enum": [0, 1]},
                }
            },
            "consensus": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "s":    {"type": "string"},
                        "bias": {"type": "integer", "enum": [-1, 0, 1]},
                        "conf": {"type": "number"}
                    },
                    "required": ["s", "bias", "conf"]
                }
            },
            "alerts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "s":    {"type": "string"},
                        "bias": {"type": "integer", "enum": [-1, 0, 1]},
                        "conf": {"type": "number"},
                        "why":  {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["s", "bias", "conf", "why"]
                }
            }
        },
        "required": ["ts", "day", "macro", "consensus", "alerts"]
    }

# --- tolerant JSON loader (strips code fences if the model ever slips) ---
def _loads_json_or_strip(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        s = raw.strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        return json.loads(s)

# --- Replace your ask_gemini(...) with this version ---

def ask_gemini(prompt, day_dash):
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        "gemini-2.5-flash-lite",
        system_instruction=(
            "You MUST return valid JSON only—no prose. "
            "Your entire response must be a single JSON object. "
            "If 'day' is missing, use the provided fallback."
        )
    )

    gen_cfg = {
        "temperature": 1,
        "max_output_tokens": 1000,
        "response_mime_type": "application/json",  # keep JSON-only
        # NOTE: intentionally omitting response_schema to avoid SDK bug
    }

    merged_prompt = (
        prompt +
        f"\n\nREMINDER:\n- Output must be VALID JSON (no code fences).\n"
        f"- If 'day' is missing, use {day_dash}.\n"
        "- If you don't know a macro field (vix/dxy/ust/breadth/adr/smt), omit it.\n"
        "- consensus should be an array of objects: [{\"s\":\"TICK\",\"bias\":-1|0|1,\"conf\":0..1}].\n"
        "- alerts should be an array of objects with fields {s,bias,conf,why[]}.\n"
    )

    resp = model.generate_content(merged_prompt, generation_config=gen_cfg, request_options={"timeout": 25})
    raw = getattr(resp, "text", "") or ""

    # Parse & validate
    data = _parse_json_loose(raw)
    data = _validate_and_coerce(data, day_dash)
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def main():
    print(f"--- Pipeline started at {datetime.now(ET_ZONE).strftime('%Y-%m-%d %H:%M:%S')} ET ---")
    start_time = time.time()

    paths = get_file_paths()
    prompt = construct_prompt(paths)
    if not prompt:
        sys.exit(1)

    try:
        condensed_text = ask_gemini(prompt, paths["day_dash"])
    except Exception as e:
        print(f"Gemini request failed: {e}")
        sys.exit(1)

    # Sanity check: fail fast if malformed JSON sneaks through
    try:
        _ = _loads_json_or_strip(condensed_text)
    except Exception as e:
        print(f"Model returned non-JSON (after strip): {e}")
        sys.exit(1)

    try:
        with open(paths["out_path"], "w", encoding="utf-8") as f:
            f.write(condensed_text)
        print(f"Saved: {paths['out_path']}")
    except Exception as e:
        print(f"Error writing output: {e}")
        sys.exit(1)

    print(f"--- Pipeline finished in {time.time() - start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()
