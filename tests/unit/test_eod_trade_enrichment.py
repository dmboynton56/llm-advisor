"""Unit tests for trade enrichment helpers in scripts/run_eod_aggregate.py."""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_eod_aggregate import (  # noqa: E402
    AccountSnapshotRow,
    _underlying_from_occ,
    dedupe_account_snapshots,
    derive_trade_enrichment,
    parse_account_snapshots,
)


class TestUnderlyingFromOcc:
    def test_parses_occ_symbols(self):
        assert _underlying_from_occ("SPY260620C00500000") == "SPY"
        assert _underlying_from_occ("QQQ260703P00500000") == "QQQ"
        assert _underlying_from_occ("IWM251219C00230000") == "IWM"

    def test_rejects_plain_symbols(self):
        assert _underlying_from_occ("SPY") is None
        assert _underlying_from_occ("") is None
        assert _underlying_from_occ(None) is None


class TestDeriveTradeEnrichment:
    def test_option_trade_with_metadata(self):
        result = derive_trade_enrichment(
            symbol="SPY260620C00500000",
            asset_class="option",
            underlying_symbol=None,
            setup_type="mr",
            option_metadata={"dte": 9, "underlying_symbol": "SPY", "delta": 0.42},
        )
        assert result["underlying_symbol"] == "SPY"
        assert result["asset_class"] == "option"
        assert result["setup_type"] == "MR"
        assert result["option_dte"] == 9
        assert result["option_metadata"]["delta"] == 0.42

    def test_occ_fallback_without_metadata(self):
        result = derive_trade_enrichment(
            symbol="QQQ260703P00500000",
            asset_class=None,
            underlying_symbol=None,
            setup_type=None,
            option_metadata=None,
        )
        assert result["asset_class"] == "option"
        assert result["underlying_symbol"] == "QQQ"
        assert result["option_dte"] is None
        assert result["setup_type"] is None

    def test_stock_trade(self):
        result = derive_trade_enrichment(
            symbol="SPY",
            asset_class=None,
            underlying_symbol=None,
            setup_type="TC",
            option_metadata=None,
        )
        assert result["asset_class"] == "stock"
        assert result["underlying_symbol"] == "SPY"

    def test_metadata_as_json_string(self):
        result = derive_trade_enrichment(
            symbol="SPY260620C00500000",
            asset_class="option",
            underlying_symbol="SPY",
            setup_type=None,
            option_metadata='{"dte": "12"}',
        )
        assert result["option_dte"] == 12


class TestAccountSnapshotParsing:
    def test_parse_artifact(self, tmp_path):
        artifact = tmp_path / "account_snapshot.json"
        artifact.write_text(
            """
            {"snapshots": [
              {"captured_at": "2026-07-02T13:31:00+00:00", "equity": 100000.5,
               "last_equity": 99900.0, "buying_power": 200000.0,
               "daily_pnl": 100.5, "daily_pnl_pct": 0.001, "source": "alpaca_paper"},
              {"captured_at": "not a timestamp", "equity": 1.0}
            ]}
            """,
            encoding="utf-8",
        )
        rows = parse_account_snapshots("2026-07-02", artifact)
        assert len(rows) == 1
        row = rows[0]
        assert row.snapshot_date == "2026-07-02"
        assert row.equity == 100000.5
        assert row.daily_pnl == 100.5
        assert row.source == "alpaca_paper"

    def test_missing_file(self, tmp_path):
        assert parse_account_snapshots("2026-07-02", tmp_path / "nope.json") == []

    def test_dedupe(self):
        row = AccountSnapshotRow(
            snapshot_date="2026-07-02",
            captured_at="2026-07-02T13:31:00+00:00",
            equity=1.0,
            last_equity=None,
            buying_power=None,
            daily_pnl=None,
            daily_pnl_pct=None,
            source="alpaca_paper",
        )
        newer = AccountSnapshotRow(
            snapshot_date="2026-07-02",
            captured_at="2026-07-02T13:31:00+00:00",
            equity=2.0,
            last_equity=None,
            buying_power=None,
            daily_pnl=None,
            daily_pnl_pct=None,
            source="alpaca_paper",
        )
        deduped = dedupe_account_snapshots([row, newer])
        assert len(deduped) == 1
        assert deduped[0].equity == 2.0
