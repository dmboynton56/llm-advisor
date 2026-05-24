from __future__ import annotations

from datetime import date, datetime, timezone

from src.data.bigquery_storage import BigQueryStorage


class _QueryResult:
    def result(self):
        return []


class _FakeClient:
    def __init__(self) -> None:
        self.query_text = ""
        self.job_config = None

    def query(self, query_text, job_config=None):
        self.query_text = query_text
        self.job_config = job_config
        return _QueryResult()


def _storage(fake_client: _FakeClient) -> BigQueryStorage:
    storage = object.__new__(BigQueryStorage)
    storage.project_id = "project"
    storage.dataset_id = "dataset"
    storage.client = fake_client
    return storage


def test_get_trades_uses_date_expression_for_date_filters() -> None:
    fake = _FakeClient()
    storage = _storage(fake)

    storage.get_trades(
        start_date=date(2026, 5, 22),
        end_date=date(2026, 5, 22),
        symbol="SPY",
    )

    assert "DATE(entry_time, 'America/New_York') >= @start_date" in fake.query_text
    assert "DATE(entry_time, 'America/New_York') <= @end_date" in fake.query_text
    params = {param.name: param for param in fake.job_config.query_parameters}
    assert params["start_date"].type_ == "DATE"
    assert params["end_date"].type_ == "DATE"


def test_get_trades_uses_timestamp_comparison_for_datetime_filters() -> None:
    fake = _FakeClient()
    storage = _storage(fake)

    storage.get_trades(
        start_date=datetime(2026, 5, 22, 13, 30, tzinfo=timezone.utc),
        end_date=datetime(2026, 5, 22, 16, 0, tzinfo=timezone.utc),
    )

    assert "entry_time >= @start_date" in fake.query_text
    assert "entry_time <= @end_date" in fake.query_text
    assert "DATE(entry_time" not in fake.query_text
    params = {param.name: param for param in fake.job_config.query_parameters}
    assert params["start_date"].type_ == "TIMESTAMP"
    assert params["end_date"].type_ == "TIMESTAMP"
