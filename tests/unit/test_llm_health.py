import json
from types import SimpleNamespace

import pytest

from src.live.llm_health import MonitoredLLMClient


def _summary(tmp_path):
    path = tmp_path / "session_summary.json"
    path.write_text(json.dumps({
        "session_end_reason": "segment_handoff", "summary_status": "ok", "total_trades": 0,
    }), encoding="utf-8")
    return path


def test_quota_outage_defers_failure_until_after_position_management(tmp_path):
    def unavailable(*args, **kwargs):
        raise RuntimeError("429 credit_balance_exhausted: You have no credits remaining")

    client = MonitoredLLMClient(SimpleNamespace(call_structured=unavailable))
    with pytest.raises(RuntimeError, match="credit_balance_exhausted"):
        client.call_structured("trade", {})
    # The caller can reject the entry and keep managing positions until cutoff.
    summary = _summary(tmp_path)
    with pytest.raises(SystemExit, match="1/1 calls"):
        client.finalize(summary)
    payload = json.loads(summary.read_text())
    assert payload["session_end_reason"] == "segment_handoff"
    assert payload["summary_status"] == "degraded"
    assert payload["llm_health"]["errors"] == 1


def test_returned_provider_errors_also_fail_segment(tmp_path):
    response = SimpleNamespace(content={"error": "provider unavailable"})
    client = MonitoredLLMClient(SimpleNamespace(call_structured=lambda *a, **k: response))
    assert client.call_structured("market review", {}) is response
    with pytest.raises(SystemExit):
        client.finalize(_summary(tmp_path))


@pytest.mark.parametrize("calls", [0, 1])
def test_healthy_rejections_and_no_signal_sessions_succeed(tmp_path, calls):
    response = SimpleNamespace(content={"should_execute": False})
    client = MonitoredLLMClient(SimpleNamespace(call_structured=lambda *a, **k: response))
    for _ in range(calls):
        client.call_structured("trade", {})
    summary = _summary(tmp_path)
    client.finalize(summary)
    payload = json.loads(summary.read_text())
    assert payload["summary_status"] == "ok"
    assert payload["llm_health"]["status"] == ("ok" if calls else "not_checked")


def test_later_success_does_not_hide_a_dependency_failure(tmp_path):
    responses = iter([RuntimeError("temporary outage"), SimpleNamespace(content={"ok": True})])

    def call(*args, **kwargs):
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    client = MonitoredLLMClient(SimpleNamespace(call_structured=call))
    with pytest.raises(RuntimeError):
        client.call_structured("trade", {})
    client.call_structured("trade", {})
    with pytest.raises(SystemExit, match="1/2 calls"):
        client.finalize(_summary(tmp_path))
