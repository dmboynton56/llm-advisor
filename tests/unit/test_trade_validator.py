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


def test_hard_rr_gate_rejects_before_llm_call() -> None:
    signal = SignalEvent(
        symbol="SPY",
        setup_type="MR",
        side="long",
        entry_price=500.0,
        z_score=-1.2,
        thresholds_used={},
        timestamp=datetime.now(timezone.utc),
        signal_uid="signal-rr-1",
    )
    state = SimpleNamespace(
        trade=SimpleNamespace(entry_price=500.0, sl_price=499.0, tp_price=500.5),
        last_z=-1.2,
        atr_percentile=40.0,
        htf_bias="bullish",
        status="mr_triggered",
    )
    llm_client = SimpleNamespace(
        call_structured=lambda prompt, schema: (_ for _ in ()).throw(
            AssertionError("hard gate should run before the LLM")
        )
    )

    result = validate_trade_with_llm(
        signal=signal,
        state=state,
        premarket_context=SimpleNamespace(symbols={}),
        llm_client=llm_client,
    )

    assert result.should_execute is False
    assert result.risk_assessment == "hard_veto"
    assert "underlying_risk_reward" in result.veto_flags


def test_hard_rr_gate_accepts_float_dust_at_min_ratio() -> None:
    """1.5R plans with float residue just under 1.5 must not hard-veto."""
    from src.execution.risk_calculator import calculate_risk_reward_ratio

    entry = 500.0
    stop = 499.0
    # Slightly under exact 1.5R (fails bare `>= 1.5`, passes epsilon gate).
    target = entry + (1.5 * (entry - stop)) * (1.0 - 1e-12)
    assert calculate_risk_reward_ratio(entry, stop, target) < 1.5

    signal = SignalEvent(
        symbol="SPY",
        setup_type="MR",
        side="long",
        entry_price=entry,
        z_score=-1.2,
        thresholds_used={},
        timestamp=datetime.now(timezone.utc),
        signal_uid="signal-rr-float-1",
    )
    state = SimpleNamespace(
        trade=SimpleNamespace(entry_price=entry, sl_price=stop, tp_price=target),
        last_z=-1.2,
        atr_percentile=40.0,
        htf_bias="bullish",
        status="mr_triggered",
    )
    llm_client = SimpleNamespace(
        call_structured=lambda prompt, schema: SimpleNamespace(
            content={
                "should_execute": True,
                "confidence": 55,
                "reasoning": "RR geometry is valid.",
                "risk_assessment": "medium",
            }
        )
    )

    result = validate_trade_with_llm(
        signal=signal,
        state=state,
        premarket_context=SimpleNamespace(symbols={}),
        llm_client=llm_client,
    )

    assert result.should_execute is True
    assert "underlying_risk_reward" not in result.veto_flags
    rr_gate = next(g for g in result.gate_results if g["code"] == "underlying_risk_reward")
    assert rr_gate["status"] == "pass"
