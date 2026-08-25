import json

from scripts.report_trade_funnel import build_report, collect_local, render_markdown


def test_collect_local_counts_funnel_failures_exits_and_pnl(tmp_path) -> None:
    processed = tmp_path / "2026-05-22" / "processed"
    processed.mkdir(parents=True)
    events = [
        {"event_type": "signal_detected"},
        {"event_type": "validation_approved"},
        {"event_type": "execution_attempt"},
        {"event_type": "execution_failed", "details": {"reason": "no_contract"}},
        {"event_type": "execution_failed", "details": {"reason": "no_contract"}},
        {"event_type": "validation_rejected", "details": {"reason": "chop"}},
    ]
    (processed / "order_events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
    )
    (processed / "session_summary.json").write_text(
        json.dumps(
            {
                "trades": [
                    {
                        "asset_class": "option",
                        "qty": 2,
                        "entry_price": 8.0,
                        "exit_reason": "stop_loss",
                        "pnl": -120.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_report(collect_local(tmp_path))
    day = report["days"][0]

    assert day["signal_detected"] == 1
    assert day["validation_approved"] == 1
    assert day["validation_rejected"] == 1
    assert day["execution_attempt"] == 1
    assert day["failure_reasons"] == {"no_contract": 2}
    assert day["exit_reasons"] == {"stop_loss": 1}
    assert day["total_pnl"] == -120.0
    assert day["average_position_size"] == 1600.0


def test_report_uses_signal_outcomes_and_late_session_split(tmp_path) -> None:
    processed = tmp_path / "2026-05-22" / "processed"
    processed.mkdir(parents=True)
    events = [
        {
            "event_type": "signal_detected",
            "ts": "2026-05-22T16:10:00Z",
            "details": {"signal_uid": "sig-1"},
        },
        {
            "event_type": "validation_approved",
            "details": {"signal_uid": "sig-1"},
        },
        {
            "event_type": "max_concurrent_skipped",
            "details": {"signal_uid": "sig-1"},
        },
        {
            "event_type": "max_concurrent_skipped",
            "details": {"signal_uid": "sig-1"},
        },
        {
            "event_type": "signal_outcome",
            "details": {"signal_uid": "sig-1", "outcome": "capacity_expired"},
        },
    ]
    (processed / "order_events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
    )

    report = build_report(collect_local(tmp_path))
    day = report["days"][0]

    assert day["signal_funnel"]["detected"] == 1
    assert day["signal_funnel"]["capacity_expired"] == 1
    assert day["signal_funnel"]["approved_no_attempt"] == 1
    assert day["signal_funnel"]["session_periods"]["11_to_13"]["detected"] == 1
    rendered = render_markdown(report)
    assert "Approved/no attempt" in rendered
    assert "Late-session signal outcomes" in rendered
