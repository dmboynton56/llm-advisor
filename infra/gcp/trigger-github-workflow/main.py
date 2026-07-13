"""HTTP Cloud Function: trigger llm-advisor GitHub Actions via workflow_dispatch."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request

import functions_framework

logger = logging.getLogger(__name__)

MAX_ERROR_BODY_LENGTH = 4096

REPO = os.environ.get("GITHUB_REPO", "dmboynton56/llm-advisor")
SYMBOLS = os.environ.get("GITHUB_SYMBOLS", "SPY,QQQ,IWM")

WORKFLOWS = {
    "premarket": "premarket.yml",
    "live": "live_loop.yml",
}


def _dispatch(workflow_key: str) -> tuple[int, str]:
    if workflow_key not in WORKFLOWS:
        return 400, f"Unknown workflow: {workflow_key!r}. Use: {', '.join(WORKFLOWS)}"

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        return 500, "GITHUB_TOKEN is not configured."

    body = json.dumps(
        {"ref": "main", "inputs": {"symbols": SYMBOLS}},
        separators=(",", ":"),
    ).encode("utf-8")
    url = (
        f"https://api.github.com/repos/{REPO}/actions/workflows/"
        f"{WORKFLOWS[workflow_key]}/dispatches"
    )
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "llm-advisor-gcp-scheduler",
        },
        method="POST",
    )
    started_at = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            latency_ms = (time.monotonic() - started_at) * 1000
            logger.info(
                "GitHub dispatch workflow=%s status=%s latency_ms=%.1f",
                workflow_key,
                resp.status,
                latency_ms,
            )
            return 204, f"Triggered {workflow_key} ({WORKFLOWS[workflow_key]})"
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        detail = detail[:MAX_ERROR_BODY_LENGTH]
        latency_ms = (time.monotonic() - started_at) * 1000
        logger.error(
            "GitHub dispatch workflow=%s status=%s latency_ms=%.1f body=%s",
            workflow_key,
            exc.code,
            latency_ms,
            detail,
        )
        status = 502 if exc.code >= 500 else exc.code
        return status, f"GitHub API error: {detail}"
    except Exception:
        latency_ms = (time.monotonic() - started_at) * 1000
        logger.exception(
            "GitHub dispatch workflow=%s status=unavailable latency_ms=%.1f",
            workflow_key,
            latency_ms,
        )
        return 502, "GitHub API request failed."


@functions_framework.http
def trigger(request):
    workflow = request.args.get("workflow", "").strip().lower()
    status, message = _dispatch(workflow)
    return (message, status)
