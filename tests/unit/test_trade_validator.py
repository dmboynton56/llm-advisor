from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import pytest

from src.analysis.trade_validator import validate_trade_with_llm
from src.live.threshold_evaluator import SignalEvent


def _context(symbol: str):
    return SimpleNamespace(symbols={symbol: SimpleNamespace(
        bias_available=True, bias_error=None, model_output={},
        daily_bias="choppy", confidence=0, news_summary="No material news",
    )})


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
    premarket_context = _context("IWM")
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
    premarket_context = _context("QQQ")
    llm_client = SimpleNamespace(
        call_structured=lambda prompt, schema: SimpleNamespace(
            content=[
                {
                    "should_execute": True,
                    "confidence": 65,
                    "reasoning": "Breakout holds above PDH.",
                    "risk_assessment": "medium",
                    "veto_flags": [],
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
        premarket_context=_context("SPY"),
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
                "veto_flags": [],
            }
        )
    )

    result = validate_trade_with_llm(
        signal=signal,
        state=state,
        premarket_context=_context("SPY"),
        llm_client=llm_client,
    )

    assert result.should_execute is True
    assert "underlying_risk_reward" not in result.veto_flags
    rr_gate = next(g for g in result.gate_results if g["code"] == "underlying_risk_reward")
    assert rr_gate["status"] == "pass"


@pytest.mark.parametrize("require_ml_bias,missing_record,approved", [
    (True, False, False), (False, False, True), (False, True, False),
])
def test_missing_model_is_explicit_and_only_optional_for_stock_experiments(
    require_ml_bias: bool, missing_record: bool, approved: bool,
) -> None:
    signal = SimpleNamespace(symbol="TSLA", setup_type="TC", side="long", signal_uid="stock-trial")
    state = SimpleNamespace(
        trade=SimpleNamespace(entry_price=100, sl_price=99, tp_price=102),
        last_z=2.5, atr_percentile=80, htf_bias="bullish", status="tc_triggered",
    )
    context = _context("TSLA")
    if missing_record:
        context.symbols = {}
    else:
        bias = context.symbols["TSLA"]
        bias.bias_available = False
        bias.bias_error = "model_load_failed"
        bias.model_output = {"error": "model_load_failed"}
    prompts = []

    def call(prompt, schema):
        prompts.append(prompt)
        return SimpleNamespace(content={
            "should_execute": True, "confidence": 65, "reasoning": "Valid technical setup",
            "risk_assessment": "medium", "veto_flags": [],
        })

    result = validate_trade_with_llm(
        signal, state, context, SimpleNamespace(call_structured=call), require_ml_bias,
    )
    assert result.should_execute is approved
    gate = next(g for g in result.gate_results if g["code"] == "premarket_data_quality")
    assert gate["status"] == ("warn" if approved else "fail")
    assert bool(prompts) is approved
    if approved:
        assert "ML daily bias unavailable" in prompts[0]
        assert "ML Model Prediction" not in prompts[0]


@pytest.mark.parametrize("value", ["false", "true", 1, None])
def test_non_boolean_llm_decision_is_never_an_approval(value) -> None:
    signal = SimpleNamespace(symbol="SPY", setup_type="MR", side="long", signal_uid="parse-trial")
    state = SimpleNamespace(
        trade=SimpleNamespace(entry_price=100, sl_price=99, tp_price=102),
        last_z=-1.5, atr_percentile=40, htf_bias="bullish", status="mr_triggered",
    )
    result = validate_trade_with_llm(signal, state, _context("SPY"), SimpleNamespace(
        call_structured=lambda *args: SimpleNamespace(content={"should_execute": value}),
    ))
    assert result.should_execute is False
    assert result.risk_assessment == "validation_error"


@pytest.mark.parametrize("patch", [
    {"confidence": True}, {"confidence": 101}, {"reasoning": None},
    {"veto_flags": "liquidity_risk"}, {"veto_flags": ["unknown"]},
])
def test_malformed_required_decision_fields_fail_closed(patch) -> None:
    signal = SimpleNamespace(symbol="SPY", setup_type="MR", side="long", signal_uid="parse-fields")
    state = SimpleNamespace(
        trade=SimpleNamespace(entry_price=100, sl_price=99, tp_price=102),
        last_z=-1.5, atr_percentile=40, htf_bias="bullish", status="mr_triggered",
    )
    content = {
        "should_execute": True, "confidence": 65, "reasoning": "Valid plan",
        "risk_assessment": "medium", "veto_flags": [], **patch,
    }
    result = validate_trade_with_llm(signal, state, _context("SPY"), SimpleNamespace(
        call_structured=lambda *args: SimpleNamespace(content=content),
    ))
    assert not result.should_execute
    assert result.risk_assessment == "validation_error"
