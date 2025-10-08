# src/data_processing/condense_context.py
#!/usr/bin/env python3

import os
import sys
import json
from datetime import datetime
import pytz
from dotenv import load_dotenv
from openai import OpenAI
import time

# allow imports relative to project root if needed (prompts/, config/, etc.)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

load_dotenv()

ET_TZ = "US/Eastern"

def get_file_paths():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    today_str = datetime.now(pytz.timezone(ET_TZ)).strftime('%Y-%m-%d')
    daily_processed = os.path.join(project_root, 'data', 'daily_news', today_str, 'processed')
    os.makedirs(daily_processed, exist_ok=True)
    return {
        # synthesized/combined JSON you created earlier in processed/
        "synth_input": os.path.join(daily_processed, 'processed_briefing.json'),
        # optional prompt template (we'll fallback if missing)
        "prompt_template": os.path.join(project_root, 'prompts', 'prompts.json'),
        # condensed output
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

    # try to load a template; fallback to a tiny inline instruction
    tmpl = None
    try:
        with open(paths['prompt_template'], 'r', encoding='utf-8') as f:
            prompts = json.load(f)
            tmpl = (prompts.get('premarket_condense_prompt') or {}).get('content')
    except Exception:
        tmpl = None

    if not tmpl:
        tmpl = (
            "You will receive a large premarket synthesized-bias JSON (multiple models). "
            "Condense it into a tiny JSON context for intraday use. "
            "Bias mapping: bearish=-1, choppy=0, bullish=1. Confidence in [0,1]. "
            "Return STRICT JSON only with:\n"
            "{"
            "\"ts\":\"<UTC ISO>\","
            "\"day\":\"YYYY-MM-DD\","
            "\"macro\":{\"vix\":num|null,\"dxy\":num|null,\"ust\":num|null,"
            "\"breadth\":int|null,\"adr\":num|null,\"smt\":0|1|null},"
            "\"consensus\":[[\"TICK\",-1|0|1,0-1],...],"
            "\"alerts\":[{\"s\":\"TICK\",\"bias\":-1|0|1,\"conf\":0-1,\"why\":[\"tag\",\"tag\"]}]"
            "}"
        )

    payload = json.dumps(synth, separators=(",", ":"), ensure_ascii=False)
    prompt = f"{tmpl}\n\nINPUT_JSON:\n```json\n{payload}\n```"
    return prompt

def ask_openai(prompt, day_dash):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        temperature=1,
        max_completion_tokens=1000,
        messages=[
            {"role": "system", "content": "Return STRICT JSON per schema. No prose."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"If 'day' is missing, use {day_dash}."}
        ],
    )
    return resp.choices[0].message.content

def main():
    paths = get_file_paths()
    prompt = construct_prompt(paths)
    if not prompt:
        sys.exit(1)

    try:
        condensed = ask_openai(prompt, paths["day_dash"])
    except Exception as e:
        print(f"OpenAI request failed: {e}")
        sys.exit(1)

    try:
        with open(paths["out_path"], "w", encoding="utf-8") as f:
            f.write(condensed)
        print(f"Saved: {paths['out_path']}")
    except Exception as e:
        print(f"Error writing output: {e}")
        sys.exit(1)

if __name__ == "__main__":
    t0 = time.monotonic()
    print(f"--- Pipeline started at {datetime.now(ET_TZ).strftime('%Y-%m-%d %H:%M:%S')} ET ---")
    main()
    print(f"--- Pipeline finished in {time.monotonic() - t0:.2f} seconds ---")
