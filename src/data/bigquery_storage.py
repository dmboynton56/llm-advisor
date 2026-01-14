"""BigQuery storage adapter for production GCP deployment.

Supports all StorageAdapter methods using Google BigQuery.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import json
import os

try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

from .storage import StorageAdapter


class BigQueryStorage(StorageAdapter):
    """BigQuery storage adapter for production."""
    
    def __init__(self, project_id: str, dataset_id: str, credentials_path: Optional[str] = None):
        """Initialize BigQuery storage.
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID (e.g., 'trading_signals')
            credentials_path: Path to service account JSON file (or use GOOGLE_APPLICATION_CREDENTIALS env var)
        """
        if not BIGQUERY_AVAILABLE:
            raise ImportError("google-cloud-bigquery not available. Install with: pip install google-cloud-bigquery google-auth")
        
        self.project_id = project_id
        self.dataset_id = dataset_id
        
        # Initialize BigQuery client
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/bigquery"]
            )
            self.client = bigquery.Client(credentials=credentials, project=project_id)
        else:
            # Use default credentials from environment
            self.client = bigquery.Client(project=project_id)
        
        self.dataset_ref = self.client.dataset(dataset_id)
        self._init_schema()
    
    def _init_schema(self):
        """Initialize BigQuery dataset and tables if they don't exist."""
        # Create dataset if it doesn't exist
        try:
            self.client.get_dataset(self.dataset_ref)
        except Exception:
            dataset = bigquery.Dataset(self.dataset_ref)
            dataset.location = "US"  # Set your preferred location
            self.client.create_dataset(dataset, exists_ok=True)
        
        # Create tables (BigQuery will ignore if they already exist)
        self._create_table_daily_bias()
        self._create_table_premarket_snapshots()
        self._create_table_market_analysis()
        self._create_table_live_loop_logs()
        self._create_table_trade_signals()
        self._create_table_llm_validations()
        self._create_table_trades()
        self._create_table_positions()
    
    def _create_table_daily_bias(self):
        """Create daily_bias table."""
        table_id = f"{self.project_id}.{self.dataset_id}.daily_bias"
        schema = [
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("bias", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("confidence", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("model_output", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("news_summary", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("premarket_price", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        table.clustering_fields = ["date", "symbol"]
        self.client.create_table(table, exists_ok=True)
    
    def _create_table_premarket_snapshots(self):
        """Create premarket_snapshots table."""
        table_id = f"{self.project_id}.{self.dataset_id}.premarket_snapshots"
        schema = [
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("htf_stats", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("bands_5m", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        table.clustering_fields = ["date", "symbol"]
        self.client.create_table(table, exists_ok=True)
    
    def _create_table_market_analysis(self):
        """Create market_analysis table."""
        table_id = f"{self.project_id}.{self.dataset_id}.market_analysis"
        schema = [
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("analysis_text", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("threshold_multipliers", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("confidence", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("llm_model", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        table.clustering_fields = ["timestamp"]
        self.client.create_table(table, exists_ok=True)
    
    def _create_table_live_loop_logs(self):
        """Create live_loop_logs table."""
        table_id = f"{self.project_id}.{self.dataset_id}.live_loop_logs"
        schema = [
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("z_score", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("mu", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("sigma", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("side", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("current_price", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("atr_percentile", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("ema_slope_hourly", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        table.clustering_fields = ["timestamp", "symbol"]
        self.client.create_table(table, exists_ok=True)
    
    def _create_table_trade_signals(self):
        """Create trade_signals table."""
        table_id = f"{self.project_id}.{self.dataset_id}.trade_signals"
        schema = [
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("setup_type", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("side", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("entry_price", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("z_score", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("threshold_multipliers", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        table.clustering_fields = ["timestamp", "symbol"]
        self.client.create_table(table, exists_ok=True)
    
    def _create_table_llm_validations(self):
        """Create llm_validations table."""
        table_id = f"{self.project_id}.{self.dataset_id}.llm_validations"
        schema = [
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("signal_id", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("should_execute", "BOOL", mode="NULLABLE"),
            bigquery.SchemaField("confidence", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("reasoning", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("risk_assessment", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("llm_model", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        table.clustering_fields = ["signal_id", "timestamp"]
        self.client.create_table(table, exists_ok=True)
    
    def _create_table_trades(self):
        """Create trades table."""
        table_id = f"{self.project_id}.{self.dataset_id}.trades"
        schema = [
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("trade_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("side", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("entry_price", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("stop_loss", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("take_profit", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("qty", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("entry_time", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("exit_time", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("exit_price", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("pnl", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("exit_reason", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        table.clustering_fields = ["symbol", "entry_time"]
        self.client.create_table(table, exists_ok=True)
    
    def _create_table_positions(self):
        """Create positions table."""
        table_id = f"{self.project_id}.{self.dataset_id}.positions"
        schema = [
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("trade_id", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("side", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("entry_price", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("current_price", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("stop_loss", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("take_profit", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("qty", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("unrealized_pnl", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("last_updated", "TIMESTAMP", mode="REQUIRED"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        table.clustering_fields = ["symbol", "trade_id"]
        self.client.create_table(table, exists_ok=True)
    
    def _get_next_id(self, table_name: str) -> int:
        """Get next ID for a table (BigQuery doesn't support auto-increment)."""
        query = f"""
        SELECT COALESCE(MAX(id), 0) + 1 as next_id
        FROM `{self.project_id}.{self.dataset_id}.{table_name}`
        """
        result = list(self.client.query(query).result())
        if result:
            return result[0].next_id
        return 1
    
    def _json_dumps(self, data: Any) -> Optional[str]:
        """Convert data to JSON string."""
        return json.dumps(data) if data is not None else None
    
    def _json_loads(self, data: str) -> Any:
        """Parse JSON string."""
        return json.loads(data) if data else None
    
    def save_daily_bias(self, date: date, symbol: str, bias_data: Dict[str, Any]) -> None:
        """Save daily bias prediction."""
        table_id = f"{self.project_id}.{self.dataset_id}.daily_bias"
        
        # Check if record exists
        query = f"""
        SELECT id FROM `{table_id}`
        WHERE date = @date AND symbol = @symbol
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("date", "DATE", date),
                bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
            ]
        )
        result = self.client.query(query, job_config=job_config).result()
        existing = list(result)
        
        if existing:
            # Update existing record
            update_query = f"""
            UPDATE `{table_id}`
            SET bias = @bias,
                confidence = @confidence,
                model_output = @model_output,
                news_summary = @news_summary,
                premarket_price = @premarket_price
            WHERE date = @date AND symbol = @symbol
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("date", "DATE", date),
                    bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
                    bigquery.ScalarQueryParameter("bias", "STRING", bias_data.get("bias", "choppy")),
                    bigquery.ScalarQueryParameter("confidence", "INT64", bias_data.get("confidence", 50)),
                    bigquery.ScalarQueryParameter("model_output", "STRING", self._json_dumps(bias_data.get("model_output"))),
                    bigquery.ScalarQueryParameter("news_summary", "STRING", bias_data.get("news_summary", "")),
                    bigquery.ScalarQueryParameter("premarket_price", "NUMERIC", float(bias_data.get("premarket_price", 0.0))),
                ]
            )
            self.client.query(update_query, job_config=job_config).result()
        else:
            # Insert new record
            next_id = self._get_next_id("daily_bias")
            insert_query = f"""
            INSERT INTO `{table_id}`
            (id, date, symbol, bias, confidence, model_output, news_summary, premarket_price, created_at)
            VALUES (@id, @date, @symbol, @bias, @confidence, @model_output, @news_summary, @premarket_price, CURRENT_TIMESTAMP())
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("id", "INT64", next_id),
                    bigquery.ScalarQueryParameter("date", "DATE", date),
                    bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
                    bigquery.ScalarQueryParameter("bias", "STRING", bias_data.get("bias", "choppy")),
                    bigquery.ScalarQueryParameter("confidence", "INT64", bias_data.get("confidence", 50)),
                    bigquery.ScalarQueryParameter("model_output", "STRING", self._json_dumps(bias_data.get("model_output"))),
                    bigquery.ScalarQueryParameter("news_summary", "STRING", bias_data.get("news_summary", "")),
                    bigquery.ScalarQueryParameter("premarket_price", "NUMERIC", float(bias_data.get("premarket_price", 0.0))),
                ]
            )
            self.client.query(insert_query, job_config=job_config).result()
    
    def get_daily_bias(self, date: date, symbol: str) -> Optional[Dict[str, Any]]:
        """Get daily bias for a symbol."""
        table_id = f"{self.project_id}.{self.dataset_id}.daily_bias"
        query = f"""
        SELECT * FROM `{table_id}`
        WHERE date = @date AND symbol = @symbol
        LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("date", "DATE", date),
                bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
            ]
        )
        result = self.client.query(query, job_config=job_config).result()
        row = next(result, None)
        if not row:
            return None
        
        return {
            "id": row.id,
            "date": row.date,
            "symbol": row.symbol,
            "bias": row.bias,
            "confidence": row.confidence,
            "model_output": self._json_loads(row.model_output) if row.model_output else None,
            "news_summary": row.news_summary,
            "premarket_price": float(row.premarket_price) if row.premarket_price else 0.0,
            "created_at": row.created_at
        }
    
    def save_premarket_snapshot(self, date: date, symbol: str, snapshot: Dict[str, Any]) -> None:
        """Save premarket snapshot."""
        table_id = f"{self.project_id}.{self.dataset_id}.premarket_snapshots"
        
        # Check if record exists
        query = f"""
        SELECT id FROM `{table_id}`
        WHERE date = @date AND symbol = @symbol
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("date", "DATE", date),
                bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
            ]
        )
        result = self.client.query(query, job_config=job_config).result()
        existing = list(result)
        
        if existing:
            # Update existing record
            update_query = f"""
            UPDATE `{table_id}`
            SET htf_stats = @htf_stats,
                bands_5m = @bands_5m
            WHERE date = @date AND symbol = @symbol
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("date", "DATE", date),
                    bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
                    bigquery.ScalarQueryParameter("htf_stats", "STRING", self._json_dumps(snapshot.get("htf"))),
                    bigquery.ScalarQueryParameter("bands_5m", "STRING", self._json_dumps(snapshot.get("bands_5m"))),
                ]
            )
            self.client.query(update_query, job_config=job_config).result()
        else:
            # Insert new record
            next_id = self._get_next_id("premarket_snapshots")
            insert_query = f"""
            INSERT INTO `{table_id}`
            (id, date, symbol, htf_stats, bands_5m, created_at)
            VALUES (@id, @date, @symbol, @htf_stats, @bands_5m, CURRENT_TIMESTAMP())
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("id", "INT64", next_id),
                    bigquery.ScalarQueryParameter("date", "DATE", date),
                    bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
                    bigquery.ScalarQueryParameter("htf_stats", "STRING", self._json_dumps(snapshot.get("htf"))),
                    bigquery.ScalarQueryParameter("bands_5m", "STRING", self._json_dumps(snapshot.get("bands_5m"))),
                ]
            )
            self.client.query(insert_query, job_config=job_config).result()
    
    def get_premarket_snapshot(self, date: date, symbol: str) -> Optional[Dict[str, Any]]:
        """Get premarket snapshot for a symbol."""
        table_id = f"{self.project_id}.{self.dataset_id}.premarket_snapshots"
        query = f"""
        SELECT * FROM `{table_id}`
        WHERE date = @date AND symbol = @symbol
        LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("date", "DATE", date),
                bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
            ]
        )
        result = self.client.query(query, job_config=job_config).result()
        row = next(result, None)
        if not row:
            return None
        
        return {
            "id": row.id,
            "date": row.date,
            "symbol": row.symbol,
            "htf": self._json_loads(row.htf_stats) if row.htf_stats else None,
            "bands_5m": self._json_loads(row.bands_5m) if row.bands_5m else None,
            "created_at": row.created_at
        }
    
    def save_market_analysis(self, analysis: Dict[str, Any]) -> int:
        """Save market analysis. Returns analysis ID."""
        table_id = f"{self.project_id}.{self.dataset_id}.market_analysis"
        next_id = self._get_next_id("market_analysis")
        
        timestamp = analysis.get("timestamp")
        if isinstance(timestamp, datetime):
            timestamp = timestamp
        elif isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            timestamp = datetime.now()
        
        insert_query = f"""
        INSERT INTO `{table_id}`
        (id, timestamp, analysis_text, threshold_multipliers, confidence, llm_model, created_at)
        VALUES (@id, @timestamp, @analysis_text, @threshold_multipliers, @confidence, @llm_model, CURRENT_TIMESTAMP())
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("id", "INT64", next_id),
                bigquery.ScalarQueryParameter("timestamp", "TIMESTAMP", timestamp),
                bigquery.ScalarQueryParameter("analysis_text", "STRING", analysis.get("analysis_text", "")),
                bigquery.ScalarQueryParameter("threshold_multipliers", "STRING", self._json_dumps(analysis.get("threshold_multipliers", {}))),
                bigquery.ScalarQueryParameter("confidence", "INT64", analysis.get("confidence", 0)),
                bigquery.ScalarQueryParameter("llm_model", "STRING", analysis.get("llm_model", "")),
            ]
        )
        self.client.query(insert_query, job_config=job_config).result()
        return next_id
    
    def save_live_loop_log(self, log_entry: Dict[str, Any]) -> None:
        """Save live loop log entry."""
        table_id = f"{self.project_id}.{self.dataset_id}.live_loop_logs"
        next_id = self._get_next_id("live_loop_logs")
        
        timestamp = log_entry.get("timestamp")
        if isinstance(timestamp, datetime):
            timestamp = timestamp
        elif isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            timestamp = datetime.now()
        
        insert_query = f"""
        INSERT INTO `{table_id}`
        (id, timestamp, symbol, z_score, mu, sigma, status, side, current_price, atr_percentile, ema_slope_hourly, created_at)
        VALUES (@id, @timestamp, @symbol, @z_score, @mu, @sigma, @status, @side, @current_price, @atr_percentile, @ema_slope_hourly, CURRENT_TIMESTAMP())
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("id", "INT64", next_id),
                bigquery.ScalarQueryParameter("timestamp", "TIMESTAMP", timestamp),
                bigquery.ScalarQueryParameter("symbol", "STRING", log_entry.get("symbol", "")),
                bigquery.ScalarQueryParameter("z_score", "NUMERIC", float(log_entry.get("z_score")) if log_entry.get("z_score") is not None else None),
                bigquery.ScalarQueryParameter("mu", "NUMERIC", float(log_entry.get("mu")) if log_entry.get("mu") is not None else None),
                bigquery.ScalarQueryParameter("sigma", "NUMERIC", float(log_entry.get("sigma")) if log_entry.get("sigma") is not None else None),
                bigquery.ScalarQueryParameter("status", "STRING", log_entry.get("status", "idle")),
                bigquery.ScalarQueryParameter("side", "STRING", log_entry.get("side")),
                bigquery.ScalarQueryParameter("current_price", "NUMERIC", float(log_entry.get("current_price")) if log_entry.get("current_price") is not None else None),
                bigquery.ScalarQueryParameter("atr_percentile", "NUMERIC", float(log_entry.get("atr_percentile")) if log_entry.get("atr_percentile") is not None else None),
                bigquery.ScalarQueryParameter("ema_slope_hourly", "NUMERIC", float(log_entry.get("ema_slope_hourly")) if log_entry.get("ema_slope_hourly") is not None else None),
            ]
        )
        self.client.query(insert_query, job_config=job_config).result()
    
    def save_trade_signal(self, signal: Dict[str, Any]) -> int:
        """Save trade signal. Returns signal ID."""
        table_id = f"{self.project_id}.{self.dataset_id}.trade_signals"
        next_id = self._get_next_id("trade_signals")
        
        timestamp = signal.get("timestamp")
        if isinstance(timestamp, datetime):
            timestamp = timestamp
        elif isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            timestamp = datetime.now()
        
        insert_query = f"""
        INSERT INTO `{table_id}`
        (id, timestamp, symbol, setup_type, side, entry_price, z_score, threshold_multipliers, created_at)
        VALUES (@id, @timestamp, @symbol, @setup_type, @side, @entry_price, @z_score, @threshold_multipliers, CURRENT_TIMESTAMP())
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("id", "INT64", next_id),
                bigquery.ScalarQueryParameter("timestamp", "TIMESTAMP", timestamp),
                bigquery.ScalarQueryParameter("symbol", "STRING", signal.get("symbol", "")),
                bigquery.ScalarQueryParameter("setup_type", "STRING", signal.get("setup_type", "")),
                bigquery.ScalarQueryParameter("side", "STRING", signal.get("side", "")),
                bigquery.ScalarQueryParameter("entry_price", "NUMERIC", float(signal.get("entry_price")) if signal.get("entry_price") is not None else None),
                bigquery.ScalarQueryParameter("z_score", "NUMERIC", float(signal.get("z_score")) if signal.get("z_score") is not None else None),
                bigquery.ScalarQueryParameter("threshold_multipliers", "STRING", self._json_dumps(signal.get("threshold_multipliers", {}))),
            ]
        )
        self.client.query(insert_query, job_config=job_config).result()
        return next_id
    
    def save_llm_validation(self, validation: Dict[str, Any]) -> int:
        """Save LLM validation. Returns validation ID."""
        table_id = f"{self.project_id}.{self.dataset_id}.llm_validations"
        next_id = self._get_next_id("llm_validations")
        
        timestamp = validation.get("timestamp")
        if isinstance(timestamp, datetime):
            timestamp = timestamp
        elif isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            timestamp = datetime.now()
        
        insert_query = f"""
        INSERT INTO `{table_id}`
        (id, signal_id, timestamp, should_execute, confidence, reasoning, risk_assessment, llm_model, created_at)
        VALUES (@id, @signal_id, @timestamp, @should_execute, @confidence, @reasoning, @risk_assessment, @llm_model, CURRENT_TIMESTAMP())
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("id", "INT64", next_id),
                bigquery.ScalarQueryParameter("signal_id", "INT64", validation.get("signal_id")),
                bigquery.ScalarQueryParameter("timestamp", "TIMESTAMP", timestamp),
                bigquery.ScalarQueryParameter("should_execute", "BOOL", validation.get("should_execute", False)),
                bigquery.ScalarQueryParameter("confidence", "INT64", validation.get("confidence", 0)),
                bigquery.ScalarQueryParameter("reasoning", "STRING", validation.get("reasoning", "")),
                bigquery.ScalarQueryParameter("risk_assessment", "STRING", validation.get("risk_assessment", "")),
                bigquery.ScalarQueryParameter("llm_model", "STRING", validation.get("llm_model", "")),
            ]
        )
        self.client.query(insert_query, job_config=job_config).result()
        return next_id
    
    def save_trade(self, trade: Dict[str, Any]) -> int:
        """Save trade. Returns trade ID."""
        table_id = f"{self.project_id}.{self.dataset_id}.trades"
        
        # Check if trade_id exists
        trade_id_str = trade.get("trade_id")
        if trade_id_str:
            query = f"""
            SELECT id FROM `{table_id}`
            WHERE trade_id = @trade_id
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("trade_id", "STRING", trade_id_str),
                ]
            )
            result = self.client.query(query, job_config=job_config).result()
            existing = list(result)
            
            if existing:
                # Update existing trade
                entry_time = trade.get("entry_time")
                if isinstance(entry_time, datetime):
                    entry_time = entry_time
                elif isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                else:
                    entry_time = None
                
                exit_time = trade.get("exit_time")
                if isinstance(exit_time, datetime):
                    exit_time = exit_time
                elif isinstance(exit_time, str):
                    exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                else:
                    exit_time = None
                
                update_query = f"""
                UPDATE `{table_id}`
                SET status = @status,
                    exit_time = @exit_time,
                    exit_price = @exit_price,
                    pnl = @pnl,
                    exit_reason = @exit_reason
                WHERE trade_id = @trade_id
                """
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("trade_id", "STRING", trade_id_str),
                        bigquery.ScalarQueryParameter("status", "STRING", trade.get("status", "pending")),
                        bigquery.ScalarQueryParameter("exit_time", "TIMESTAMP", exit_time),
                        bigquery.ScalarQueryParameter("exit_price", "NUMERIC", float(trade.get("exit_price")) if trade.get("exit_price") is not None else None),
                        bigquery.ScalarQueryParameter("pnl", "NUMERIC", float(trade.get("pnl")) if trade.get("pnl") is not None else None),
                        bigquery.ScalarQueryParameter("exit_reason", "STRING", trade.get("exit_reason", "")),
                    ]
                )
                self.client.query(update_query, job_config=job_config).result()
                return existing[0].id
        
        # Insert new trade
        next_id = self._get_next_id("trades")
        
        entry_time = trade.get("entry_time")
        if isinstance(entry_time, datetime):
            entry_time = entry_time
        elif isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
        else:
            entry_time = None
        
        exit_time = trade.get("exit_time")
        if isinstance(exit_time, datetime):
            exit_time = exit_time
        elif isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
        else:
            exit_time = None
        
        insert_query = f"""
        INSERT INTO `{table_id}`
        (id, trade_id, symbol, side, entry_price, stop_loss, take_profit, qty, status, entry_time, exit_time, exit_price, pnl, exit_reason, created_at)
        VALUES (@id, @trade_id, @symbol, @side, @entry_price, @stop_loss, @take_profit, @qty, @status, @entry_time, @exit_time, @exit_price, @pnl, @exit_reason, CURRENT_TIMESTAMP())
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("id", "INT64", next_id),
                bigquery.ScalarQueryParameter("trade_id", "STRING", trade_id_str),
                bigquery.ScalarQueryParameter("symbol", "STRING", trade.get("symbol", "")),
                bigquery.ScalarQueryParameter("side", "STRING", trade.get("side", "")),
                bigquery.ScalarQueryParameter("entry_price", "NUMERIC", float(trade.get("entry_price")) if trade.get("entry_price") is not None else None),
                bigquery.ScalarQueryParameter("stop_loss", "NUMERIC", float(trade.get("stop_loss")) if trade.get("stop_loss") is not None else None),
                bigquery.ScalarQueryParameter("take_profit", "NUMERIC", float(trade.get("take_profit")) if trade.get("take_profit") is not None else None),
                bigquery.ScalarQueryParameter("qty", "INT64", trade.get("qty", 0)),
                bigquery.ScalarQueryParameter("status", "STRING", trade.get("status", "pending")),
                bigquery.ScalarQueryParameter("entry_time", "TIMESTAMP", entry_time),
                bigquery.ScalarQueryParameter("exit_time", "TIMESTAMP", exit_time),
                bigquery.ScalarQueryParameter("exit_price", "NUMERIC", float(trade.get("exit_price")) if trade.get("exit_price") is not None else None),
                bigquery.ScalarQueryParameter("pnl", "NUMERIC", float(trade.get("pnl")) if trade.get("pnl") is not None else None),
                bigquery.ScalarQueryParameter("exit_reason", "STRING", trade.get("exit_reason", "")),
            ]
        )
        self.client.query(insert_query, job_config=job_config).result()
        return next_id
    
    def update_position(self, position: Dict[str, Any]) -> None:
        """Update or create position."""
        table_id = f"{self.project_id}.{self.dataset_id}.positions"
        trade_id = position.get("trade_id")
        
        # Check if position exists
        query = f"""
        SELECT id FROM `{table_id}`
        WHERE trade_id = @trade_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("trade_id", "INT64", trade_id),
            ]
        )
        result = self.client.query(query, job_config=job_config).result()
        existing = list(result)
        
        if existing:
            # Update existing position
            update_query = f"""
            UPDATE `{table_id}`
            SET current_price = @current_price,
                unrealized_pnl = @unrealized_pnl,
                last_updated = CURRENT_TIMESTAMP()
            WHERE trade_id = @trade_id
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("trade_id", "INT64", trade_id),
                    bigquery.ScalarQueryParameter("current_price", "NUMERIC", float(position.get("current_price")) if position.get("current_price") is not None else None),
                    bigquery.ScalarQueryParameter("unrealized_pnl", "NUMERIC", float(position.get("unrealized_pnl", 0.0))),
                ]
            )
            self.client.query(update_query, job_config=job_config).result()
        else:
            # Insert new position
            next_id = self._get_next_id("positions")
            insert_query = f"""
            INSERT INTO `{table_id}`
            (id, trade_id, symbol, side, entry_price, current_price, stop_loss, take_profit, qty, unrealized_pnl, last_updated)
            VALUES (@id, @trade_id, @symbol, @side, @entry_price, @current_price, @stop_loss, @take_profit, @qty, @unrealized_pnl, CURRENT_TIMESTAMP())
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("id", "INT64", next_id),
                    bigquery.ScalarQueryParameter("trade_id", "INT64", trade_id),
                    bigquery.ScalarQueryParameter("symbol", "STRING", position.get("symbol", "")),
                    bigquery.ScalarQueryParameter("side", "STRING", position.get("side", "")),
                    bigquery.ScalarQueryParameter("entry_price", "NUMERIC", float(position.get("entry_price")) if position.get("entry_price") is not None else None),
                    bigquery.ScalarQueryParameter("current_price", "NUMERIC", float(position.get("current_price")) if position.get("current_price") is not None else None),
                    bigquery.ScalarQueryParameter("stop_loss", "NUMERIC", float(position.get("stop_loss")) if position.get("stop_loss") is not None else None),
                    bigquery.ScalarQueryParameter("take_profit", "NUMERIC", float(position.get("take_profit")) if position.get("take_profit") is not None else None),
                    bigquery.ScalarQueryParameter("qty", "INT64", position.get("qty", 0)),
                    bigquery.ScalarQueryParameter("unrealized_pnl", "NUMERIC", float(position.get("unrealized_pnl", 0.0))),
                ]
            )
            self.client.query(insert_query, job_config=job_config).result()
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        table_id = f"{self.project_id}.{self.dataset_id}.positions"
        query = f"""
        SELECT * FROM `{table_id}`
        ORDER BY last_updated DESC
        """
        result = self.client.query(query).result()
        return [dict(row) for row in result]
    
    def get_trades(self, start_date: Optional[date] = None, end_date: Optional[date] = None, 
                   symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get trades with optional filters."""
        table_id = f"{self.project_id}.{self.dataset_id}.trades"
        query = f"SELECT * FROM `{table_id}` WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND entry_time >= @start_date"
            params.append(bigquery.ScalarQueryParameter("start_date", "DATE", start_date))
        
        if end_date:
            query += " AND entry_time <= @end_date"
            params.append(bigquery.ScalarQueryParameter("end_date", "DATE", end_date))
        
        if symbol:
            query += " AND symbol = @symbol"
            params.append(bigquery.ScalarQueryParameter("symbol", "STRING", symbol))
        
        query += " ORDER BY entry_time DESC"
        
        job_config = bigquery.QueryJobConfig(query_parameters=params) if params else None
        result = self.client.query(query, job_config=job_config).result()
        return [dict(row) for row in result]
