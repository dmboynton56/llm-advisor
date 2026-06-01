from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from src.analysis.trade_validator import validate_trade_with_llm
from src.live.threshold_evaluator import SignalEvent


def test_llm_validation_parse_failure_rejects_trade() -> None:
    signal = SignalEvent(
        symbol="IWM",
        setup_type="MR",
        side="short",
        entry_price=284.18,
        z_score=0.58,
        thresholds_used={},
        timestamp=datetime.now(timezone.utc),
    )
    state = SimpleNamespace(
        trade=SimpleNamespace(
            entry_price=284.18,
            sl_price=284.79,
            tp_price=283.26,
        ),
        last_z=0.58,
        atr_percentile=45.0,
        htf_bias="bullish",
        status="mr_triggered",
    )
    premarket_context = SimpleNamespace(symbols={})
    llm_client = SimpleNamespace(
        call_structured=lambda prompt, schema: SimpleNamespace(content=[])
    )

    result = validate_trade_with_llm(
        signal=signal,
        state=state,
        premarket_context=premarket_context,
        llm_client=llm_client,
    )

    assert result.should_execute is False
    assert result.confidence == 0
    assert result.risk_assessment == "validation_error"


def test_llm_validation_unwraps_gemini_style_list_response() -> None:
    signal = SignalEvent(
        symbol="QQQ",
        setup_type="TC",
        side="long",
        entry_price=737.77,
        z_score=2.5,
        thresholds_used={},
        timestamp=datetime.now(timezone.utc),
    )
    state = SimpleNamespace(
        trade=SimpleNamespace(
            entry_price=737.77,
            sl_price=737.14,
            tp_price=738.715,
        ),
        last_z=2.5,
        atr_percentile=81.7,
        htf_bias="bullish",
        status="tc_triggered",
    )
    premarket_context = SimpleNamespace(symbols={})
    llm_client = SimpleNamespace(
        call_structured=lambda prompt, schema: SimpleNamespace(
            content=[
                {
                    "should_execute": True,
                    "confidence": 65,
                    "reasoning": "Breakout holds above PDH.",
                    "risk_assessment": "medium",
                }
            ]
        )
    )

    result = validate_trade_with_llm(
        signal=signal,
        state=state,
        premarket_context=premarket_context,
        llm_client=llm_client,
    )

    assert result.should_execute is True
    assert result.confidence == 65
    assert result.risk_assessment == "medium"
