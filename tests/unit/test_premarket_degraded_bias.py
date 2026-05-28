from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

from src.analysis.market_analyzer import MarketAnalyzer
from src.premarket.bias_gatherer import (
    PremarketBias,
    PremarketContext,
    gather_premarket_bias,
)
from src.premarket.bias_validator import validate_biases_with_llm


def test_gather_premarket_bias_keeps_symbol_errors_degraded(
    tmp_path,
    monkeypatch,
) -> None:
    trading_date = date(2026, 5, 28)
    output_dir = tmp_path / "data" / "daily_news" / "2026-05-28" / "processed"
    raw_dir = output_dir.parent / "raw"

    def fake_run(cmd, cwd, env, check, capture_output, text):
        script = str(cmd[1])
        raw_dir.mkdir(parents=True, exist_ok=True)
        if script.endswith("news_scraper.py"):
            (raw_dir / "news.json").write_text(
                json.dumps(
                    {
                        "symbols": {
                            "IWM": [{"headline": "Small caps steady before open"}],
                            "SPY": [{"headline": "Index futures edge higher"}],
                        },
                        "macro": [{"headline": "Macro calendar quiet"}],
                    }
                ),
                encoding="utf-8",
            )
        elif script.endswith("daily_bias_computing.py"):
            (raw_dir / "daily_bias.json").write_text(
                json.dumps(
                    {
                        "generated_for_date_et": "2026-05-28",
                        "data_feed": "iex",
                        "symbols": {
                            "SPY": {
                                "bias": "bullish",
                                "confidence": 0.72,
                                "premarket_price": 621.40,
                                "features_snapshot": {"overnight_gap_pct": 0.1},
                            },
                            "IWM": {"error": {"error": "no_open_or_premarket"}},
                        },
                    }
                ),
                encoding="utf-8",
            )
        return SimpleNamespace(stdout="ok", stderr="")

    monkeypatch.setattr("src.premarket.bias_gatherer.subprocess.run", fake_run)

    context = gather_premarket_bias(
        trading_date=trading_date,
        symbols=["SPY", "IWM"],
        output_dir=output_dir,
        enable_llm_validation=False,
    )

    assert set(context.symbols) == {"SPY", "IWM"}
    assert context.symbols["SPY"].bias_available is True
    assert context.symbols["SPY"].needs_bias is True
    assert context.symbols["SPY"].daily_bias == "bullish"
    assert context.symbols["SPY"].confidence == 72

    iwm = context.symbols["IWM"]
    assert iwm.bias_available is False
    assert iwm.needs_bias is False
    assert iwm.bias_error == "no_open_or_premarket"
    assert iwm.daily_bias == "choppy"
    assert iwm.confidence == 0
    assert iwm.model_output["degraded_mode"] is True
    assert "Small caps steady" in iwm.news_summary


def test_llm_bias_validation_skips_unavailable_ml_bias() -> None:
    context = PremarketContext(
        date="2026-05-28",
        symbols={
            "IWM": PremarketBias(
                symbol="IWM",
                daily_bias="choppy",
                confidence=0,
                model_output={"error": "no_open_or_premarket"},
                news_summary="",
                premarket_price=0.0,
                needs_bias=False,
                bias_available=False,
                bias_error="no_open_or_premarket",
            )
        },
        market_context={},
    )
    llm_client = SimpleNamespace(
        call_structured=lambda prompt, schema: (_ for _ in ()).throw(
            AssertionError("LLM should not be called without available ML bias")
        )
    )

    assert validate_biases_with_llm(context, llm_client=llm_client) == {}


def test_market_analysis_prompt_marks_unavailable_bias_as_degraded() -> None:
    context = PremarketContext(
        date="2026-05-28",
        symbols={
            "IWM": PremarketBias(
                symbol="IWM",
                daily_bias="choppy",
                confidence=0,
                model_output={"error": "no_open_or_premarket"},
                news_summary="",
                premarket_price=0.0,
                needs_bias=False,
                bias_available=False,
                bias_error="no_open_or_premarket",
            )
        },
        market_context={},
    )
    analyzer = MarketAnalyzer(llm_client=SimpleNamespace())

    prompt = analyzer._build_market_analysis_prompt(
        states={},
        premarket_context=context,
        recent_price_action={},
    )

    assert "IWM: ML bias unavailable (no_open_or_premarket)" in prompt
    assert "IWM: choppy (0% confidence)" not in prompt
