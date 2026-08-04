from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.analysis.llm_client import OpenAILLMClient, normalize_structured_content
from src.core.config import Settings


def test_settings_default_to_openai_gpt_5_4_nano(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    settings = Settings.load()

    assert settings.llm.provider == "openai"
    assert settings.llm.model == "gpt-5.4-nano"


def test_openai_client_uses_gpt_5_reasoning_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                usage=SimpleNamespace(total_tokens=7),
            )

    class FakeOpenAI:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(completions=FakeCompletions())
            assert api_key == "test-key"

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    client = OpenAILLMClient(model="gpt-5.4-nano", api_key="test-key")
    response = client.call_structured("Return JSON.", {})

    assert response.content == {"ok": True}
    assert response.tokens_used == 7
    assert captured["model"] == "gpt-5.4-nano"
    assert captured["reasoning_effort"] == "low"
    assert captured["response_format"] == {"type": "json_object"}


def test_normalize_structured_content_passes_dict_through() -> None:
    payload = {"should_execute": False, "confidence": 80}
    assert normalize_structured_content(payload) is payload


def test_normalize_structured_content_unwraps_single_object_list() -> None:
    payload = [{"should_execute": True, "confidence": 72}]
    assert normalize_structured_content(payload) == payload[0]


def test_normalize_structured_content_unwraps_wrapped_object_list() -> None:
    payload = [{"symbols": {"SPY": {"final_bias": "bullish"}}}]
    assert normalize_structured_content(payload) == payload[0]


def test_normalize_structured_content_rejects_empty_list() -> None:
    with pytest.raises(TypeError, match="no object elements"):
        normalize_structured_content([])


def test_normalize_structured_content_rejects_non_object_list() -> None:
    with pytest.raises(TypeError, match="no object elements"):
        normalize_structured_content(["not-a-dict"])


def test_normalize_structured_content_rejects_scalar() -> None:
    with pytest.raises(TypeError, match="Expected dict from LLM, got str"):
        normalize_structured_content("nope")
