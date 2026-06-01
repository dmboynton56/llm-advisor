from __future__ import annotations

import pytest

from src.analysis.llm_client import normalize_structured_content


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
