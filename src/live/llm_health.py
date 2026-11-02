"""Keep dependency outages visible without interrupting position management."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.analysis.llm_client import LLMClient, LLMResponse


class MonitoredLLMClient(LLMClient):
    """Track this segment's actual calls, including market and shadow reviews."""

    def __init__(self, client: LLMClient):
        self.client = client
        self.calls = 0
        self.errors = 0
        self.first_error: Optional[str] = None

    def call_structured(
        self, prompt: str, schema: Dict[str, Any], timeout: Optional[float] = None,
    ) -> LLMResponse:
        self.calls += 1
        try:
            response = self.client.call_structured(prompt, schema, timeout=timeout)
            if isinstance(response.content, dict) and response.content.get("error"):
                self._record_error(str(response.content["error"]))
            return response
        except Exception as exc:
            self._record_error(str(exc))
            raise

    def _record_error(self, error: str) -> None:
        self.errors += 1
        if self.first_error is None:
            self.first_error = error[:500]
            logging.getLogger(__name__).error(
                "LLM dependency unavailable; entries still require valid approval. "
                "Position management will continue; segment will finish as failed: %s",
                self.first_error,
            )

    def finalize(self, summary_path: Path) -> None:
        """Run AFTER shutdown telemetry, then fail the segment if calls failed.

        The existing workflow uploads artifacts on failure and runs subsequent
        segments in manage_only mode. Never exit early and abandon open positions.
        """
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["llm_health"] = {
            "status": "error" if self.errors else "ok" if self.calls else "not_checked",
            "calls": self.calls,
            "errors": self.errors,
            "first_error": self.first_error,
        }
        if self.errors:
            summary["summary_status"] = "degraded"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if self.errors:
            raise SystemExit(
                f"LLM dependency failed on {self.errors}/{self.calls} calls; "
                "see session_summary.json llm_health. Restore provider access before normal entries."
            )
