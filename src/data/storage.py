"""Database storage abstraction layer.

Supports SQLite (dev) and PostgreSQL (prod) with unified interface.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from contextlib import contextmanager
import json
import os

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


class StorageAdapter(ABC):
    """Abstract storage adapter interface."""
    
    @abstractmethod
    def save_daily_bias(self, date: date, symbol: str, bias_data: Dict[str, Any]) -> None:
        """Save daily bias prediction."""
        pass
    
    @abstractmethod
    def get_daily_bias(self, date: date, symbol: str) -> Optional[Dict[str, Any]]:
        """Get daily bias for a symbol."""
        pass
    
    @abstractmethod
    def save_premarket_snapshot(self, date: date, symbol: str, snapshot: Dict[str, Any]) -> None:
        """Save premarket snapshot."""
        pass
    
    @abstractmethod
    def get_premarket_snapshot(self, date: date, symbol: str) -> Optional[Dict[str, Any]]:
        """Get premarket snapshot for a symbol."""
        pass
    
    @abstractmethod
    def save_market_analysis(self, analysis: Dict[str, Any]) -> int:
        """Save market analysis. Returns analysis ID."""
        pass
    
    @abstractmethod
    def save_live_loop_log(self, log_entry: Dict[str, Any]) -> None:
        """Save live loop log entry."""
        pass
    
    @abstractmethod
    def save_trade_signal(self, signal: Dict[str, Any]) -> int:
        """Save trade signal. Returns signal ID."""
        pass
    
    @abstractmethod
    def save_llm_validation(self, validation: Dict[str, Any]) -> int:
        """Save LLM validation. Returns validation ID."""
        pass
    
    @abstractmethod
    def save_trade(self, trade: Dict[str, Any]) -> int:
        """Save trade. Returns trade ID."""
        pass
    
    @abstractmethod
    def update_position(self, position: Dict[str, Any]) -> None:
        """Update or create position."""
        pass
    
    @abstractmethod
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        pass
    
    @abstractmethod
    def get_trades(self, start_date: Optional[date] = None, end_date: Optional[date] = None, 
                   symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get trades with optional filters."""
        pass


class SQLiteStorage(StorageAdapter):
    """SQLite storage adapter for local development."""
    
    def __init__(self, db_path: str = "data/trading.db"):
        """Initialize SQLite storage."""
        if not SQLITE_AVAILABLE:
            raise ImportError("sqlite3 not available")
        
        self.db_path = db_path
        # Create directory if it doesn't exist
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_schema()
    
    def _get_connection(self):
        """Get SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return dict-like rows
        return conn
    
    def _init_schema(self):
        """Initialize database schema."""
        # Get project root (2 levels up from src/data/storage.py)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        schema_path = os.path.join(
            project_root,
            "aws", "infrastructure", "database", "migrations", "001_initial_schema.sql"
        )
        
        # Read SQL file and adapt for SQLite
        with open(schema_path, 'r') as f:
            sql = f.read()
        
        # SQLite adaptations
        sql = sql.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
        sql = sql.replace("JSONB", "JSON")
        # Keep CURRENT_TIMESTAMP as-is (SQLite supports it)
        # Don't replace with datetime('now') - SQLite doesn't allow function calls in DEFAULT
        sql = sql.replace("DECIMAL(10, 4)", "REAL")
        sql = sql.replace("DECIMAL(10, 2)", "REAL")
        sql = sql.replace("DECIMAL(5, 2)", "REAL")
        
        # Split into individual statements, handling both CREATE TABLE and CREATE INDEX
        table_statements = []
        index_statements = []
        other_statements = []
        
        for s in sql.split(';'):
            s = s.strip()
            if not s:
                continue
            
            # Skip pure comment lines, but keep statements that contain CREATE commands even if they have comments
            if s.startswith('--') and 'CREATE' not in s.upper():
                continue
            
            # Remove comments from statement (SQLite doesn't support comments in CREATE statements)
            # Split by newlines, filter out comment lines and inline comments, rejoin with newlines to preserve structure
            lines = s.split('\n')
            cleaned_lines = []
            for line in lines:
                # Remove inline comments (-- after code)
                if '--' in line:
                    # Split on -- and take the part before it, but preserve if it's part of a string
                    parts = line.split('--', 1)
                    if parts[0].strip():  # Only keep if there's code before the comment
                        cleaned_lines.append(parts[0].rstrip())
                elif line.strip() and not line.strip().startswith('--'):
                    cleaned_lines.append(line)
            s = '\n'.join(cleaned_lines)
            
            if not s:
                continue
            
            # Check for CREATE TABLE (handle case-insensitive and whitespace)
            if "CREATE TABLE" in s.upper():
                if "IF NOT EXISTS" not in s.upper():
                    s = s.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS", 1)
                    s = s.replace("create table", "CREATE TABLE IF NOT EXISTS", 1)
                table_statements.append(s)
            # Check for CREATE INDEX
            elif "CREATE INDEX" in s.upper():
                if "IF NOT EXISTS" not in s.upper():
                    s = s.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS", 1)
                    s = s.replace("create index", "CREATE INDEX IF NOT EXISTS", 1)
                index_statements.append(s)
        
        # Execute in order: tables first, then indexes, then other statements
        statements = table_statements + index_statements + other_statements
        
        conn = self._get_connection()
        try:
            # Execute table statements one at a time
            for i, statement in enumerate(table_statements):
                try:
                    conn.execute(statement)
                except sqlite3.OperationalError as e:
                    error_msg = str(e).lower()
                    if "already exists" not in error_msg:
                        print(f"Error executing CREATE TABLE statement {i+1}/{len(table_statements)}")
                        print(f"Statement: {statement[:300]}...")
                        raise
            
            # Then execute index statements
            for i, statement in enumerate(index_statements):
                try:
                    conn.execute(statement)
                except sqlite3.OperationalError as e:
                    error_msg = str(e).lower()
                    if "already exists" not in error_msg and "duplicate" not in error_msg:
                        print(f"Error executing index statement {i+1}/{len(index_statements)}")
                        print(f"Statement: {statement[:150]}...")
                        raise
            
            conn.commit()
        finally:
            conn.close()
    
    def _json_dumps(self, data: Any) -> str:
        """Convert data to JSON string."""
        return json.dumps(data) if data is not None else None
    
    def _json_loads(self, data: str) -> Any:
        """Parse JSON string."""
        return json.loads(data) if data else None
    
    def save_daily_bias(self, date: date, symbol: str, bias_data: Dict[str, Any]) -> None:
        """Save daily bias prediction."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO daily_bias 
                (date, symbol, bias, confidence, model_output, news_summary, premarket_price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                date.isoformat(),
                symbol,
                bias_data.get("bias", "choppy"),
                bias_data.get("confidence", 50),
                self._json_dumps(bias_data.get("model_output")),
                bias_data.get("news_summary", ""),
                bias_data.get("premarket_price", 0.0)
            ))
            conn.commit()
        finally:
            conn.close()
    
    def get_daily_bias(self, date: date, symbol: str) -> Optional[Dict[str, Any]]:
        """Get daily bias for a symbol."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT * FROM daily_bias 
                WHERE date = ? AND symbol = ?
            """, (date.isoformat(), symbol))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "date": row["date"],
                "symbol": row["symbol"],
                "bias": row["bias"],
                "confidence": row["confidence"],
                "model_output": self._json_loads(row["model_output"]),
                "news_summary": row["news_summary"],
                "premarket_price": float(row["premarket_price"]) if row["premarket_price"] else 0.0,
                "created_at": row["created_at"]
            }
        finally:
            conn.close()
    
    def save_premarket_snapshot(self, date: date, symbol: str, snapshot: Dict[str, Any]) -> None:
        """Save premarket snapshot."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO premarket_snapshots 
                (date, symbol, htf_stats, bands_5m)
                VALUES (?, ?, ?, ?)
            """, (
                date.isoformat(),
                symbol,
                self._json_dumps(snapshot.get("htf")),
                self._json_dumps(snapshot.get("bands_5m"))
            ))
            conn.commit()
        finally:
            conn.close()
    
    def get_premarket_snapshot(self, date: date, symbol: str) -> Optional[Dict[str, Any]]:
        """Get premarket snapshot for a symbol."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT * FROM premarket_snapshots 
                WHERE date = ? AND symbol = ?
            """, (date.isoformat(), symbol))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "date": row["date"],
                "symbol": row["symbol"],
                "htf": self._json_loads(row["htf_stats"]),
                "bands_5m": self._json_loads(row["bands_5m"]),
                "created_at": row["created_at"]
            }
        finally:
            conn.close()
    
    def save_market_analysis(self, analysis: Dict[str, Any]) -> int:
        """Save market analysis."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                INSERT INTO market_analysis 
                (timestamp, analysis_text, threshold_multipliers, confidence, llm_model)
                VALUES (?, ?, ?, ?, ?)
            """, (
                analysis.get("timestamp", datetime.now()).isoformat() if isinstance(analysis.get("timestamp"), datetime) else analysis.get("timestamp"),
                analysis.get("analysis_text", ""),
                self._json_dumps(analysis.get("threshold_multipliers", {})),
                analysis.get("confidence", 0),
                analysis.get("llm_model", "")
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def save_live_loop_log(self, log_entry: Dict[str, Any]) -> None:
        """Save live loop log entry."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO live_loop_logs 
                (timestamp, symbol, z_score, mu, sigma, status, side, current_price, atr_percentile, ema_slope_hourly)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log_entry.get("timestamp", datetime.now()).isoformat() if isinstance(log_entry.get("timestamp"), datetime) else log_entry.get("timestamp"),
                log_entry.get("symbol", ""),
                log_entry.get("z_score"),
                log_entry.get("mu"),
                log_entry.get("sigma"),
                log_entry.get("status", "idle"),
                log_entry.get("side"),
                log_entry.get("current_price"),
                log_entry.get("atr_percentile"),
                log_entry.get("ema_slope_hourly")
            ))
            conn.commit()
        finally:
            conn.close()
    
    def save_trade_signal(self, signal: Dict[str, Any]) -> int:
        """Save trade signal."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                INSERT INTO trade_signals 
                (timestamp, symbol, setup_type, side, entry_price, z_score, threshold_multipliers)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.get("timestamp", datetime.now()).isoformat() if isinstance(signal.get("timestamp"), datetime) else signal.get("timestamp"),
                signal.get("symbol", ""),
                signal.get("setup_type", ""),
                signal.get("side", ""),
                signal.get("entry_price"),
                signal.get("z_score"),
                self._json_dumps(signal.get("threshold_multipliers", {}))
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def save_llm_validation(self, validation: Dict[str, Any]) -> int:
        """Save LLM validation."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                INSERT INTO llm_validations 
                (signal_id, timestamp, should_execute, confidence, reasoning, risk_assessment, llm_model)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                validation.get("signal_id"),
                validation.get("timestamp", datetime.now()).isoformat() if isinstance(validation.get("timestamp"), datetime) else validation.get("timestamp"),
                validation.get("should_execute", False),
                validation.get("confidence", 0),
                validation.get("reasoning", ""),
                validation.get("risk_assessment", ""),
                validation.get("llm_model", "")
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def save_trade(self, trade: Dict[str, Any]) -> int:
        """Save trade."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                INSERT INTO trades 
                (trade_id, symbol, side, entry_price, stop_loss, take_profit, qty, status, 
                 entry_time, exit_time, exit_price, pnl, exit_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.get("trade_id"),
                trade.get("symbol", ""),
                trade.get("side", ""),
                trade.get("entry_price"),
                trade.get("stop_loss"),
                trade.get("take_profit"),
                trade.get("qty", 0),
                trade.get("status", "pending"),
                trade.get("entry_time").isoformat() if isinstance(trade.get("entry_time"), datetime) else trade.get("entry_time"),
                trade.get("exit_time").isoformat() if isinstance(trade.get("exit_time"), datetime) else trade.get("exit_time"),
                trade.get("exit_price"),
                trade.get("pnl"),
                trade.get("exit_reason", "")
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def update_position(self, position: Dict[str, Any]) -> None:
        """Update or create position."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO positions 
                (trade_id, symbol, side, entry_price, current_price, stop_loss, take_profit, qty, unrealized_pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.get("trade_id"),
                position.get("symbol", ""),
                position.get("side", ""),
                position.get("entry_price"),
                position.get("current_price"),
                position.get("stop_loss"),
                position.get("take_profit"),
                position.get("qty", 0),
                position.get("unrealized_pnl", 0.0)
            ))
            conn.commit()
        finally:
            conn.close()
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT * FROM positions ORDER BY last_updated DESC
            """)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def get_trades(self, start_date: Optional[date] = None, end_date: Optional[date] = None, 
                   symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get trades with optional filters."""
        conn = self._get_connection()
        try:
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND entry_time >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND entry_time <= ?"
                params.append(end_date.isoformat())
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY entry_time DESC"
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


class PostgreSQLStorage(StorageAdapter):
    """PostgreSQL storage adapter for production."""
    
    def __init__(self, connection_string: str):
        """Initialize PostgreSQL storage."""
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 not available. Install with: pip install psycopg2-binary")
        
        self.conn_string = connection_string
    
    @contextmanager
    def _get_connection(self):
        """Get PostgreSQL connection with context manager."""
        conn = psycopg2.connect(self.conn_string)
        try:
            yield conn
            conn.commit()
        except:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def save_daily_bias(self, date: date, symbol: str, bias_data: Dict[str, Any]) -> None:
        """Save daily bias prediction."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO daily_bias 
                    (date, symbol, bias, confidence, model_output, news_summary, premarket_price)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, symbol) DO UPDATE SET
                        bias = EXCLUDED.bias,
                        confidence = EXCLUDED.confidence,
                        model_output = EXCLUDED.model_output,
                        news_summary = EXCLUDED.news_summary,
                        premarket_price = EXCLUDED.premarket_price
                """, (
                    date,
                    symbol,
                    bias_data.get("bias", "choppy"),
                    bias_data.get("confidence", 50),
                    json.dumps(bias_data.get("model_output")) if bias_data.get("model_output") else None,
                    bias_data.get("news_summary", ""),
                    bias_data.get("premarket_price", 0.0)
                ))
    
    def get_daily_bias(self, date: date, symbol: str) -> Optional[Dict[str, Any]]:
        """Get daily bias for a symbol."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM daily_bias 
                    WHERE date = %s AND symbol = %s
                """, (date, symbol))
                row = cur.fetchone()
                if not row:
                    return None
                return dict(row)
    
    def save_premarket_snapshot(self, date: date, symbol: str, snapshot: Dict[str, Any]) -> None:
        """Save premarket snapshot."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO premarket_snapshots 
                    (date, symbol, htf_stats, bands_5m)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (date, symbol) DO UPDATE SET
                        htf_stats = EXCLUDED.htf_stats,
                        bands_5m = EXCLUDED.bands_5m
                """, (
                    date,
                    symbol,
                    json.dumps(snapshot.get("htf")) if snapshot.get("htf") else None,
                    json.dumps(snapshot.get("bands_5m")) if snapshot.get("bands_5m") else None
                ))
    
    def get_premarket_snapshot(self, date: date, symbol: str) -> Optional[Dict[str, Any]]:
        """Get premarket snapshot for a symbol."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM premarket_snapshots 
                    WHERE date = %s AND symbol = %s
                """, (date, symbol))
                row = cur.fetchone()
                if not row:
                    return None
                return dict(row)
    
    def save_market_analysis(self, analysis: Dict[str, Any]) -> int:
        """Save market analysis."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                timestamp = analysis.get("timestamp")
                if isinstance(timestamp, datetime):
                    timestamp = timestamp
                elif isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                else:
                    timestamp = datetime.now()
                
                cur.execute("""
                    INSERT INTO market_analysis 
                    (timestamp, analysis_text, threshold_multipliers, confidence, llm_model)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    timestamp,
                    analysis.get("analysis_text", ""),
                    json.dumps(analysis.get("threshold_multipliers", {})) if analysis.get("threshold_multipliers") else None,
                    analysis.get("confidence", 0),
                    analysis.get("llm_model", "")
                ))
                return cur.fetchone()[0]
    
    def save_live_loop_log(self, log_entry: Dict[str, Any]) -> None:
        """Save live loop log entry."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                timestamp = log_entry.get("timestamp")
                if isinstance(timestamp, datetime):
                    timestamp = timestamp
                elif isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                else:
                    timestamp = datetime.now()
                
                cur.execute("""
                    INSERT INTO live_loop_logs 
                    (timestamp, symbol, z_score, mu, sigma, status, side, current_price, atr_percentile, ema_slope_hourly)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    timestamp,
                    log_entry.get("symbol", ""),
                    log_entry.get("z_score"),
                    log_entry.get("mu"),
                    log_entry.get("sigma"),
                    log_entry.get("status", "idle"),
                    log_entry.get("side"),
                    log_entry.get("current_price"),
                    log_entry.get("atr_percentile"),
                    log_entry.get("ema_slope_hourly")
                ))
    
    def save_trade_signal(self, signal: Dict[str, Any]) -> int:
        """Save trade signal."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                timestamp = signal.get("timestamp")
                if isinstance(timestamp, datetime):
                    timestamp = timestamp
                elif isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                else:
                    timestamp = datetime.now()
                
                cur.execute("""
                    INSERT INTO trade_signals 
                    (timestamp, symbol, setup_type, side, entry_price, z_score, threshold_multipliers)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    timestamp,
                    signal.get("symbol", ""),
                    signal.get("setup_type", ""),
                    signal.get("side", ""),
                    signal.get("entry_price"),
                    signal.get("z_score"),
                    json.dumps(signal.get("threshold_multipliers", {})) if signal.get("threshold_multipliers") else None
                ))
                return cur.fetchone()[0]
    
    def save_llm_validation(self, validation: Dict[str, Any]) -> int:
        """Save LLM validation."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                timestamp = validation.get("timestamp")
                if isinstance(timestamp, datetime):
                    timestamp = timestamp
                elif isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                else:
                    timestamp = datetime.now()
                
                cur.execute("""
                    INSERT INTO llm_validations 
                    (signal_id, timestamp, should_execute, confidence, reasoning, risk_assessment, llm_model)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    validation.get("signal_id"),
                    timestamp,
                    validation.get("should_execute", False),
                    validation.get("confidence", 0),
                    validation.get("reasoning", ""),
                    validation.get("risk_assessment", ""),
                    validation.get("llm_model", "")
                ))
                return cur.fetchone()[0]
    
    def save_trade(self, trade: Dict[str, Any]) -> int:
        """Save trade."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                entry_time = trade.get("entry_time")
                if isinstance(entry_time, datetime):
                    entry_time = entry_time
                elif isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time)
                else:
                    entry_time = None
                
                exit_time = trade.get("exit_time")
                if isinstance(exit_time, datetime):
                    exit_time = exit_time
                elif isinstance(exit_time, str):
                    exit_time = datetime.fromisoformat(exit_time)
                else:
                    exit_time = None
                
                cur.execute("""
                    INSERT INTO trades 
                    (trade_id, symbol, side, entry_price, stop_loss, take_profit, qty, status, 
                     entry_time, exit_time, exit_price, pnl, exit_reason)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (trade_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        exit_time = EXCLUDED.exit_time,
                        exit_price = EXCLUDED.exit_price,
                        pnl = EXCLUDED.pnl,
                        exit_reason = EXCLUDED.exit_reason
                    RETURNING id
                """, (
                    trade.get("trade_id"),
                    trade.get("symbol", ""),
                    trade.get("side", ""),
                    trade.get("entry_price"),
                    trade.get("stop_loss"),
                    trade.get("take_profit"),
                    trade.get("qty", 0),
                    trade.get("status", "pending"),
                    entry_time,
                    exit_time,
                    trade.get("exit_price"),
                    trade.get("pnl"),
                    trade.get("exit_reason", "")
                ))
                return cur.fetchone()[0]
    
    def update_position(self, position: Dict[str, Any]) -> None:
        """Update or create position."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO positions 
                    (trade_id, symbol, side, entry_price, current_price, stop_loss, take_profit, qty, unrealized_pnl)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (trade_id) DO UPDATE SET
                        current_price = EXCLUDED.current_price,
                        unrealized_pnl = EXCLUDED.unrealized_pnl,
                        last_updated = CURRENT_TIMESTAMP
                """, (
                    position.get("trade_id"),
                    position.get("symbol", ""),
                    position.get("side", ""),
                    position.get("entry_price"),
                    position.get("current_price"),
                    position.get("stop_loss"),
                    position.get("take_profit"),
                    position.get("qty", 0),
                    position.get("unrealized_pnl", 0.0)
                ))
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM positions ORDER BY last_updated DESC")
                return [dict(row) for row in cur.fetchall()]
    
    def get_trades(self, start_date: Optional[date] = None, end_date: Optional[date] = None, 
                   symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get trades with optional filters."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = "SELECT * FROM trades WHERE 1=1"
                params = []
                
                if start_date:
                    query += " AND entry_time >= %s"
                    params.append(start_date)
                
                if end_date:
                    query += " AND entry_time <= %s"
                    params.append(end_date)
                
                if symbol:
                    query += " AND symbol = %s"
                    params.append(symbol)
                
                query += " ORDER BY entry_time DESC"
                
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]


class Storage:
    """Factory for storage adapters."""
    
    @staticmethod
    def create(env: str = "dev", db_path: Optional[str] = None, 
               connection_string: Optional[str] = None,
               project_id: Optional[str] = None,
               dataset_id: Optional[str] = None,
               credentials_path: Optional[str] = None) -> StorageAdapter:
        """
        Create storage adapter.
        
        Args:
            env: Environment ("dev" for SQLite, "prod" for PostgreSQL, "bq" for BigQuery)
            db_path: SQLite database path (default: "data/trading.db")
            connection_string: PostgreSQL connection string (default: from DATABASE_URL env var)
            project_id: GCP project ID (required for BigQuery)
            dataset_id: BigQuery dataset ID (default: "trading_signals")
            credentials_path: Path to GCP service account JSON (or use GOOGLE_APPLICATION_CREDENTIALS)
        
        Returns:
            StorageAdapter instance
        """
        if env == "dev":
            return SQLiteStorage(db_path or "data/trading.db")
        elif env == "bq" or env == "prod":
            # Check if BigQuery is requested
            if env == "bq" or (env == "prod" and not connection_string):
                from .bigquery_storage import BigQueryStorage
                project_id = project_id or os.getenv("GCP_PROJECT_ID")
                dataset_id = dataset_id or os.getenv("GCP_DATASET_ID", "trading_signals")
                credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                
                if not project_id:
                    raise ValueError("GCP_PROJECT_ID environment variable or project_id parameter required for BigQuery")
                
                return BigQueryStorage(project_id, dataset_id, credentials_path)
            else:
                # PostgreSQL fallback
                conn_string = connection_string or os.getenv("DATABASE_URL")
                if not conn_string:
                    raise ValueError("DATABASE_URL environment variable required for PostgreSQL production")
                return PostgreSQLStorage(conn_string)
        else:
            raise ValueError(f"Unknown environment: {env}. Use 'dev', 'prod', or 'bq'")

