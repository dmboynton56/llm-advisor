"""Unit tests for src/analytics/ops_metrics.py against fixture trade sets."""
from __future__ import annotations

import pytest

from src.analytics.ops_metrics import (
    biggest_losers,
    compute_ops_metrics,
    dte_bucket,
    execution_funnel,
    max_drawdown,
    signal_level_funnel,
    summarize_trades,
    trades_per_day,
)


def _trade(**overrides):
    base = {
        "trade_uid": "2026-07-01:abc",
        "run_date": "2026-07-01",
        "order_id": "abc",
        "symbol": "SPY260710C00600000",
        "underlying_symbol": "SPY",
        "asset_class": "option",
        "side": "buy",
        "setup_type": "MR",
        "option_dte": 9,
        "qty": 1,
        "entry_price": 2.0,
        "stop_loss": None,
        "take_profit": None,
        "exit_price": 2.5,
        "pnl": 50.0,
        "status": "closed",
        "exit_reason": "option_profit_target",
    }
    base.update(overrides)
    return base


FIXTURE_TRADES = [
    _trade(trade_uid="t1", order_id="o1", pnl=50.0),
    _trade(trade_uid="t2", order_id="o2", pnl=-30.0, exit_reason="option_stop_loss"),
    _trade(
        trade_uid="t3",
        order_id="o3",
        underlying_symbol="QQQ",
        symbol="QQQ260703P00500000",
        side="sell",
        setup_type="TC",
        option_dte=2,
        pnl=-80.0,
        run_date="2026-07-02",
        exit_reason="option_time_stop",
    ),
    _trade(trade_uid="t4", order_id="o4", pnl=100.0, run_date="2026-07-02"),
    # Open trade: counted in totals but not in closed stats.
    _trade(trade_uid="t5", order_id="o5", pnl=None, status="open", exit_price=None),
]


class TestDteBucket:
    @pytest.mark.parametrize(
        "dte,expected",
        [(0, "0"), (1, "1-3"), (3, "1-3"), (4, "4-7"), (7, "4-7"), (8, "8-14"), (14, "8-14"), (15, "15+"), (45, "15+")],
    )
    def test_buckets(self, dte, expected):
        assert dte_bucket(dte) == expected

    def test_invalid(self):
        assert dte_bucket(None) is None
        assert dte_bucket("nope") is None


class TestSummarizeTrades:
    def test_overall_stats(self):
        stats = summarize_trades(FIXTURE_TRADES)
        assert stats["trades"] == 5
        assert stats["closed_trades"] == 4
        assert stats["winning_trades"] == 2
        assert stats["losing_trades"] == 2
        assert stats["total_pnl"] == pytest.approx(40.0)
        assert stats["win_rate"] == pytest.approx(0.5)
        assert stats["average_win"] == pytest.approx(75.0)
        assert stats["average_loss"] == pytest.approx(-55.0)
        assert stats["avg_realized_rr"] == pytest.approx(75.0 / 55.0, rel=1e-3)
        assert stats["profit_factor"] == pytest.approx(150.0 / 110.0, rel=1e-3)

    def test_empty(self):
        stats = summarize_trades([])
        assert stats["trades"] == 0
        assert stats["win_rate"] is None
        assert stats["profit_factor"] is None

    def test_planned_rr_uses_sl_tp(self):
        trades = [
            _trade(trade_uid="rr1", entry_price=100.0, stop_loss=99.0, take_profit=102.0),
        ]
        stats = summarize_trades(trades)
        assert stats["avg_planned_rr"] == pytest.approx(2.0)


class TestEquityAndCadence:
    def test_max_drawdown(self):
        assert max_drawdown([100.0, 110.0, 90.0, 120.0, 100.0]) == pytest.approx(20.0)

    def test_max_drawdown_short_series(self):
        assert max_drawdown([100.0]) is None

    def test_trades_per_day(self):
        assert trades_per_day(FIXTURE_TRADES) == pytest.approx(2.5)
        assert trades_per_day([]) is None


class TestBiggestLosers:
    def test_orders_by_pnl_and_joins_reasoning(self):
        events = [
            {
                "event_type": "validation_approved",
                "order_id": "o3",
                "details": {"reasoning": "LLM liked the setup", "confidence": 70},
            }
        ]
        losers = biggest_losers(FIXTURE_TRADES, events, n=2)
        assert [l["trade_uid"] for l in losers] == ["t3", "t2"]
        assert losers[0]["validation_reasoning"] == "LLM liked the setup"
        assert losers[1]["validation_reasoning"] is None
        assert losers[0]["side"] == "short"

    def test_joins_pre_order_validation_through_execution_correlation(self):
        events = [
            {
                "run_date": "2026-07-02",
                "event_type": "validation_approved",
                "symbol": "QQQ",
                "loop_count": 17,
                "order_id": None,
                "details": {"reasoning": "Approved before the broker order existed"},
            },
            {
                "run_date": "2026-07-02",
                "event_type": "execution_succeeded",
                "symbol": "QQQ",
                "loop_count": 17,
                "order_id": "o3",
                "details": {"order": {"order_id": "o3"}},
            },
        ]

        losers = biggest_losers(FIXTURE_TRADES, events, n=1)

        assert (
            losers[0]["validation_reasoning"]
            == "Approved before the broker order existed"
        )

    def test_correlation_does_not_cross_symbols_or_loop_iterations(self):
        events = [
            {
                "run_date": "2026-07-02",
                "event_type": "validation_approved",
                "symbol": "SPY",
                "loop_count": 17,
                "order_id": None,
                "details": {"reasoning": "Wrong symbol"},
            },
            {
                "run_date": "2026-07-02",
                "event_type": "validation_approved",
                "symbol": "QQQ",
                "loop_count": 18,
                "order_id": None,
                "details": {"reasoning": "Wrong loop"},
            },
            {
                "run_date": "2026-07-02",
                "event_type": "execution_succeeded",
                "symbol": "QQQ",
                "loop_count": 17,
                "order_id": "o3",
                "details": {"order": {"order_id": "o3"}},
            },
        ]

        losers = biggest_losers(FIXTURE_TRADES, events, n=1)

        assert losers[0]["validation_reasoning"] is None


class TestExecutionFunnel:
    def test_stages_and_rejections(self):
        events = [
            {"event_type": "signal_detected"},
            {"event_type": "signal_detected"},
            {"event_type": "signal_detected"},
            {"event_type": "validation_approved"},
            {"event_type": "validation_approved"},
            {"event_type": "validation_rejected", "details": {"reasoning": "chop"}},
            {"event_type": "execution_attempt"},
            {"event_type": "execution_attempt"},
            {"event_type": "execution_succeeded"},
            {"event_type": "execution_failed", "details": {"reason": "no_contract"}},
        ]
        funnel = execution_funnel(events)
        assert funnel["stages"] == {
            "signals": 3,
            "validation_approved": 2,
            "execution_attempted": 2,
            "executed": 1,
        }
        assert funnel["llm_approval_rate"] == pytest.approx(2 / 3)
        assert funnel["rejection_reasons"]["execution_failed:no_contract"] == 1
        assert "validation_rejected" in funnel["rejection_reasons"]

    def test_empty(self):
        funnel = execution_funnel([])
        assert funnel["llm_approval_rate"] is None
        assert funnel["stages"]["signals"] == 0

    def test_signal_level_funnel_deduplicates_retries_and_tracks_session_periods(self):
        events = [
            {
                "event_type": "signal_detected",
                "ts": "2026-08-20T18:45:00Z",
                "details": {"signal_uid": "sig-cap"},
            },
            {
                "event_type": "validation_approved",
                "details": {"signal_uid": "sig-cap"},
            },
            {
                "event_type": "max_concurrent_skipped",
                "details": {"signal_uid": "sig-cap"},
            },
            {
                "event_type": "max_concurrent_skipped",
                "details": {"signal_uid": "sig-cap"},
            },
            {
                "event_type": "execution_timeout",
                "details": {"signal_uid": "sig-cap", "phase": "capacity"},
            },
            {
                "event_type": "signal_outcome",
                "details": {
                    "signal_uid": "sig-cap",
                    "outcome": "capacity_expired",
                    "capacity_skip_count": 2,
                },
            },
            {
                "event_type": "signal_outcome",
                "details": {
                    "signal_uid": "sig-cap",
                    "outcome": "capacity_expired",
                    "capacity_skip_count": 2,
                },
            },
            {
                "event_type": "signal_detected",
                "ts": "2026-08-20T19:10:00Z",
                "details": {"signal_uid": "sig-fill"},
            },
            {
                "event_type": "validation_approved",
                "details": {"signal_uid": "sig-fill"},
            },
            {
                "event_type": "execution_attempt",
                "details": {"signal_uid": "sig-fill"},
            },
            {
                "event_type": "execution_succeeded",
                "details": {"signal_uid": "sig-fill"},
            },
            {
                "event_type": "signal_outcome",
                "details": {
                    "signal_uid": "sig-fill",
                    "outcome": "execution_succeeded",
                },
            },
        ]

        funnel = signal_level_funnel(events)

        assert funnel["detected"] == 2
        assert funnel["approved"] == 2
        assert funnel["capacity_blocked"] == 1
        assert funnel["capacity_expired"] == 1
        assert funnel["attempted"] == 1
        assert funnel["execution_succeeded"] == 1
        assert funnel["approved_no_attempt"] == 1
        assert funnel["session_periods"]["13_plus"]["detected"] == 2
        assert funnel["terminal_outcomes"]["capacity_expired"] == 1


class TestComputeOpsMetrics:
    def test_full_payload_shape(self):
        snapshots = [
            {"captured_at": "2026-07-01T13:00:00Z", "equity": 100000.0},
            {"captured_at": "2026-07-01T20:00:00Z", "equity": 100050.0},
            {"captured_at": "2026-07-02T20:00:00Z", "equity": 99990.0},
        ]
        payload = compute_ops_metrics(
            trades=FIXTURE_TRADES,
            order_events=[],
            account_snapshots=snapshots,
            start_date="2026-07-01",
            end_date="2026-07-02",
        )
        assert payload["overall"]["max_drawdown"] == pytest.approx(60.0)
        assert payload["equity"]["latest"] == pytest.approx(99990.0)

        by_underlying = payload["breakdowns"]["by_underlying"]
        assert set(by_underlying) == {"SPY", "QQQ"}
        assert by_underlying["SPY"]["trades"] == 4
        assert by_underlying["QQQ"]["total_pnl"] == pytest.approx(-80.0)

        by_side = payload["breakdowns"]["by_side"]
        assert by_side["long"]["trades"] == 4
        assert by_side["short"]["trades"] == 1

        by_setup = payload["breakdowns"]["by_setup_type"]
        assert by_setup["MR"]["trades"] == 4
        assert by_setup["TC"]["trades"] == 1

        by_dte = payload["breakdowns"]["by_dte_bucket"]
        assert by_dte["8-14"]["trades"] == 4
        assert by_dte["1-3"]["trades"] == 1

    def test_empty_inputs(self):
        payload = compute_ops_metrics([], [], [])
        assert payload["overall"]["trades"] == 0
        assert payload["overall"]["max_drawdown"] is None
        assert payload["biggest_losers"] == []
        assert payload["equity"]["latest"] is None
