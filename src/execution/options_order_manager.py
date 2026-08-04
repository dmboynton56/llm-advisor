"""Paper-only options order manager for STDEV signals."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import hashlib
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderClass,
    OrderSide,
    OrderType,
    PositionIntent,
    QueryOrderStatus,
    TimeInForce,
)
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopOrderRequest,
)

from src.core.config import OptionsSettings, RiskSettings, Settings
from src.data.alpaca_options_client import AlpacaOptionsClient
from src.execution.options_strategy_mapper import OptionTradePlan, OptionsStrategyMapper
from src.utils.env_sanitize import getenv_strip

load_dotenv()

_OCC_OPTION_RE = re.compile(
    r"^(?P<underlying>[A-Z0-9.]{1,10})(?P<expiry>\d{6})(?P<contract_type>[CP])\d{8}$"
)
_ACTIVE_ORDER_STATUSES = {
    "accepted",
    "accepted_for_bidding",
    "held",
    "new",
    "partially_filled",
    "pending_cancel",
    "pending_new",
    "pending_replace",
    "pending_review",
}
_TERMINAL_CANCEL_STATUSES = {"canceled", "expired", "rejected", "replaced"}
_PROTECTIVE_STOP_PREFIX = "llma-stop-"
_TIERED_EXIT_PREFIX = "llma-tier-"


class OptionsOrderManager:
    """Executes paper option trades from stock-derived signals."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True,
        settings: Optional[Settings] = None,
        options_client: Optional[AlpacaOptionsClient] = None,
        risk_state_path: Optional[Path] = None,
    ):
        self.settings = settings or Settings.load()
        self.options_settings: OptionsSettings = self.settings.options
        self.risk_settings: RiskSettings = self.settings.risk

        if self.options_settings.paper_only and not paper:
            raise RuntimeError("Options engine is paper-only; set ALPACA_PAPER_TRADING=true")

        api_key = api_key or getenv_strip("ALPACA_API_KEY")
        api_secret = api_secret or getenv_strip("ALPACA_SECRET_KEY")
        if not api_key or not api_secret:
            raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY")

        self.trading_client = TradingClient(api_key=api_key, secret_key=api_secret, paper=paper)
        self.options_client = options_client or AlpacaOptionsClient(
            api_key=api_key,
            api_secret=api_secret,
            paper=paper,
            feed=self.options_settings.data_feed,
        )
        self.mapper = OptionsStrategyMapper(self.options_settings, self.risk_settings)
        self._risk_events: List[Dict[str, Any]] = []
        self._stopout_cooldowns: Dict[str, datetime] = {}
        self._last_exit_orders: Dict[str, str] = {}
        self._risk_state_path = Path(risk_state_path) if risk_state_path else None
        self._load_risk_state()

    @staticmethod
    def _failure(error: str, **extra: Any) -> Dict[str, Any]:
        return {"success": False, "error": error, **extra}

    def get_account_equity(self) -> float:
        account = self.trading_client.get_account()
        return float(account.equity)

    def get_buying_power(self) -> float:
        account = self.trading_client.get_account()
        return float(getattr(account, "options_buying_power", None) or account.buying_power)

    def get_open_positions(self) -> List[Dict[str, Any]]:
        try:
            positions = self.trading_client.get_all_positions()
            return [self._position_to_dict(pos) for pos in positions]
        except Exception as exc:
            print(f"  ! Failed to get positions: {exc}")
            return []

    def get_open_orders(
        self,
        symbols: Optional[List[str]] = None,
        *,
        raise_on_error: bool = False,
    ) -> List[Any]:
        """Return broker-open orders; callers use this for exposure and stop checks."""
        try:
            request = GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                limit=500,
                nested=True,
                symbols=symbols,
            )
            return list(self.trading_client.get_orders(filter=request))
        except Exception as exc:
            print(f"  ! Failed to get open orders: {exc}")
            if raise_on_error:
                raise
            return []

    def _find_order_by_client_order_id(
        self, symbol: str, client_order_id: str
    ) -> Optional[Any]:
        """Find an open or recently closed order by deterministic client id."""
        try:
            candidates = self.get_open_orders([symbol], raise_on_error=True)
            request = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=500,
                nested=True,
                symbols=[symbol],
            )
            candidates.extend(list(self.trading_client.get_orders(filter=request)))
        except Exception:
            raise
        for order in candidates:
            if (
                str(getattr(order, "symbol", "")).upper() == str(symbol).upper()
                and str(getattr(order, "client_order_id", "") or "") == str(client_order_id)
            ):
                return order
        return None

    def close_position(self, symbol: str) -> bool:
        """Cancel the protective stop, confirm it is inactive, then request a close."""
        tier_cancel = self.cancel_pending_tier_orders(symbol)
        if tier_cancel.get("status") == "filled":
            try:
                still_open = any(
                    str(pos.get("symbol", "")).upper() == str(symbol).upper()
                    and abs(float(pos.get("qty") or 0)) > 0
                    for pos in self.get_open_positions()
                )
            except Exception:
                still_open = True
            if not still_open:
                self._risk_events.append(
                    {
                        "event_type": "option_exit_already_filled",
                        "symbol": symbol,
                        "details": tier_cancel,
                    }
                )
                return True
        if tier_cancel.get("status") not in ("absent", "canceled"):
            print(
                f"  ! Pending tier close for {symbol} is not safely canceled "
                f"({tier_cancel.get('status')}); deferring full close"
            )
            return False
        cancel_result = self.cancel_protective_stop(symbol)
        if cancel_result["status"] == "filled":
            self._risk_events.append(
                {
                    "event_type": "option_exit_already_filled",
                    "symbol": symbol,
                    "details": cancel_result,
                }
            )
            return True
        if cancel_result["status"] not in ("absent", "canceled"):
            print(
                f"  ! Protective stop for {symbol} is not safely canceled "
                f"({cancel_result['status']}); deferring close"
            )
            return False
        try:
            order = self.trading_client.close_position(symbol)
            order_id = str(getattr(order, "id", "") or "")
            if order_id:
                self._last_exit_orders[symbol] = order_id
            print(f"  > Closed position {symbol}")
            return True
        except Exception as exc:
            print(f"  ! Failed to close position {symbol}: {exc}")
            return False

    def close_position_quantity(
        self,
        symbol: str,
        qty: int,
        *,
        client_order_id: Optional[str] = None,
        stage: Optional[str] = None,
        lifecycle_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit an idempotent paper sell-to-close market order for exact qty.

        Tiered exits must use an explicit quantity order so a partial close does
        not liquidate the entire net broker position. The protective stop is
        canceled and confirmed before this order is submitted; the tracker
        re-attaches protection after the fill is observed.
        """
        symbol = str(symbol).upper()
        qty = int(qty)
        if qty <= 0:
            return {"success": False, "error": "invalid_close_quantity", "symbol": symbol}
        client_order_id = client_order_id or self._tiered_client_order_id(
            symbol, stage or "exit", lifecycle_id=lifecycle_id
        )
        # Recovery/idempotency: a restart or duplicate poll must reuse the
        # broker order already submitted for this deterministic client id.
        try:
            existing = self._find_order_by_client_order_id(symbol, client_order_id)
        except Exception as exc:
            return {
                "success": False,
                "error": "tiered_exit_order_lookup_failed",
                "symbol": symbol,
                "qty": qty,
                "stage": stage,
                "detail": str(exc),
            }
        if existing is not None:
            order_id = str(getattr(existing, "id", "") or "")
            if order_id:
                if not hasattr(self, "_last_exit_orders"):
                    self._last_exit_orders = {}
                self._last_exit_orders[symbol] = order_id
            return {
                "success": True,
                "symbol": symbol,
                "qty": qty,
                "stage": stage,
                "order_id": order_id,
                "client_order_id": client_order_id,
                "status": self._order_status(existing),
                "recovered": True,
            }
        cancel_result = self.cancel_protective_stop(symbol)
        if cancel_result.get("status") == "filled":
            stop_order_id = str(cancel_result.get("stop_order_id") or "")
            if stop_order_id:
                self._last_exit_orders[symbol] = stop_order_id
            self._risk_events.append(
                {
                    "event_type": "option_exit_already_filled",
                    "symbol": symbol,
                    "details": {**cancel_result, "stage": stage, "qty": qty},
                }
            )
            return {
                "success": True,
                "already_filled": True,
                "symbol": symbol,
                "qty": qty,
                "stage": stage,
            }
        if cancel_result.get("status") not in ("absent", "canceled"):
            return {
                "success": False,
                "error": "protective_stop_not_canceled",
                "symbol": symbol,
                "qty": qty,
                "stage": stage,
                "cancel_result": cancel_result,
            }
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.SIMPLE,
            position_intent=PositionIntent.SELL_TO_CLOSE,
            client_order_id=client_order_id,
        )
        try:
            order = self.trading_client.submit_order(order_data=request)
            order_id = str(getattr(order, "id", "") or "")
            if order_id:
                if not hasattr(self, "_last_exit_orders"):
                    self._last_exit_orders = {}
                self._last_exit_orders[symbol] = order_id
            return {
                "success": True,
                "symbol": symbol,
                "qty": qty,
                "stage": stage,
                "order_id": order_id,
                "client_order_id": client_order_id,
                "status": self._order_status(order),
            }
        except Exception as exc:
            return {
                "success": False,
                "error": "tiered_exit_submit_failed",
                "symbol": symbol,
                "qty": qty,
                "stage": stage,
                "client_order_id": client_order_id,
                "detail": str(exc),
            }

    def cancel_exit_order(self, symbol: str, order_id: Optional[str]) -> Dict[str, Any]:
        """Cancel a pending tier close and confirm its terminal broker status."""
        symbol = str(symbol).upper()
        if not order_id:
            return {"status": "absent", "symbol": symbol}
        try:
            order = self.trading_client.get_order_by_id(str(order_id))
            status = self._order_status(order)
        except Exception as exc:
            return {"status": "query_failed", "symbol": symbol, "order_id": str(order_id), "error": str(exc)}
        if status == "filled":
            return {"status": "filled", "symbol": symbol, "order_id": str(order_id)}
        if status in _TERMINAL_CANCEL_STATUSES:
            return {"status": status, "symbol": symbol, "order_id": str(order_id)}
        try:
            self.trading_client.cancel_order_by_id(str(order_id))
            refreshed = self.trading_client.get_order_by_id(str(order_id))
            status = self._order_status(refreshed)
        except Exception as exc:
            return {"status": "cancel_failed", "symbol": symbol, "order_id": str(order_id), "error": str(exc)}
        return {"status": status or "pending_cancel", "symbol": symbol, "order_id": str(order_id)}

    def cancel_pending_tier_orders(self, symbol: str) -> Dict[str, Any]:
        """Cancel any open tier close before a forced/full lifecycle close."""
        symbol = str(symbol).upper()
        try:
            orders = self.get_open_orders([symbol], raise_on_error=True)
        except Exception as exc:
            return {"status": "query_failed", "symbol": symbol, "error": str(exc)}
        tier_orders = [
            order
            for order in orders
            if str(getattr(order, "symbol", "")).upper() == symbol
            and str(getattr(order, "client_order_id", "") or "").startswith(_TIERED_EXIT_PREFIX)
        ]
        if not tier_orders:
            return {"status": "absent", "symbol": symbol}
        for order in tier_orders:
            order_id = str(getattr(order, "id", "") or "")
            status = self._order_status(order)
            if status == "filled":
                return {"status": "filled", "symbol": symbol, "order_id": order_id}
            try:
                self.trading_client.cancel_order_by_id(order_id)
                refreshed = self.trading_client.get_order_by_id(order_id)
                status = self._order_status(refreshed)
            except Exception as exc:
                return {"status": "cancel_failed", "symbol": symbol, "order_id": order_id, "error": str(exc)}
            if status == "filled":
                return {"status": "filled", "symbol": symbol, "order_id": order_id}
            if status not in _TERMINAL_CANCEL_STATUSES:
                return {"status": status or "pending_cancel", "symbol": symbol, "order_id": order_id}
        return {"status": "canceled", "symbol": symbol}

    def get_exit_order_status(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Recover a persisted tier close after a process restart."""
        order = None
        if order_id:
            try:
                order = self.trading_client.get_order_by_id(str(order_id))
            except Exception as exc:
                return {"status": "query_failed", "error": str(exc)}
        elif client_order_id:
            try:
                order = self._find_order_by_client_order_id(str(symbol).upper(), client_order_id)
            except Exception as exc:
                return {"status": "query_failed", "error": str(exc)}
        if order is None:
            return {"status": "absent"}
        return {
            "status": self._order_status(order),
            "order_id": str(getattr(order, "id", "") or ""),
            "client_order_id": str(getattr(order, "client_order_id", "") or ""),
            "filled_qty": self._float_or_none(getattr(order, "filled_qty", None)),
            "filled_avg_price": self._float_or_none(getattr(order, "filled_avg_price", None)),
        }

    def execute_signal_trade(self, signal: Any, state: Any) -> Optional[Dict[str, Any]]:
        """Build and submit an option order for a stock signal."""
        if not state.trade:
            return self._failure("no_trade_plan")

        try:
            account_equity = self.get_account_equity()
            plan = self.mapper.build_trade_plan(
                signal=signal,
                state=state,
                options_client=self.options_client,
                account_equity=account_equity,
            )
        except Exception as exc:
            return self._failure(
                "option_plan_failed",
                detail=str(exc),
                diagnostics=getattr(self.mapper, "last_rejection", None),
            )

        if plan is None:
            return self._failure(
                "no_option_candidate",
                diagnostics=getattr(self.mapper, "last_rejection", None),
            )

        buying_power = self.get_buying_power()
        if plan.max_loss > buying_power:
            return self._failure(
                "insufficient_options_buying_power",
                required=plan.max_loss,
                available=buying_power,
                option_plan=plan.to_dict(),
            )

        guard_failure = self._entry_guard(plan)
        if guard_failure:
            return guard_failure

        return self.execute_option_trade(plan)

    def execute_option_trade(self, plan: OptionTradePlan) -> Optional[Dict[str, Any]]:
        if plan.side != "buy" or plan.position_intent != "buy_to_open":
            return self._failure("unsupported_option_order", option_plan=plan.to_dict())

        order_request = LimitOrderRequest(
            symbol=plan.option_symbol,
            qty=plan.qty,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.SIMPLE,
            limit_price=plan.limit_price,
            position_intent=PositionIntent.BUY_TO_OPEN,
        )

        try:
            print(
                f"  > Submitting PAPER option BUY {plan.qty} {plan.option_symbol} "
                f"@ limit ${plan.limit_price:.2f} ({plan.underlying_symbol} {plan.setup_type})"
            )
            order = self.trading_client.submit_order(order_data=order_request)
            return {
                "success": True,
                "asset_class": "option",
                "order_id": order.id,
                "symbol": order.symbol,
                "underlying_symbol": plan.underlying_symbol,
                "option_symbol": plan.option_symbol,
                "qty": order.qty,
                "side": "buy",
                "status": order.status,
                "limit_price": plan.limit_price,
                "submitted_at": datetime.now().isoformat(),
                "option_plan": plan.to_dict(),
            }
        except Exception as exc:
            return self._failure(
                "alpaca_option_submit_failed",
                detail=str(exc),
                option_plan=plan.to_dict(),
            )

    def _entry_guard(self, plan: OptionTradePlan) -> Optional[Dict[str, Any]]:
        """Enforce broker-truth exposure limits, including pending buy orders."""
        positions = self.get_open_positions()
        try:
            open_orders = self.get_open_orders(raise_on_error=True)
        except Exception as exc:
            return self._failure(
                "broker_order_query_failed",
                detail=str(exc),
                option_plan=plan.to_dict(),
            )
        desired_symbol = plan.option_symbol.upper()
        desired_underlying = plan.underlying_symbol.upper()
        desired_direction = str(plan.signal_side).lower()

        held_symbols = {str(pos.get("symbol", "")).upper() for pos in positions}
        pending_entries = [
            order
            for order in open_orders
            if self._order_side(order) == "buy" and self._order_status(order) in _ACTIVE_ORDER_STATUSES
        ]
        pending_symbols = {str(getattr(order, "symbol", "")).upper() for order in pending_entries}

        if desired_symbol in held_symbols or desired_symbol in pending_symbols:
            return self._failure(
                "duplicate_option_contract",
                option_symbol=plan.option_symbol,
                option_plan=plan.to_dict(),
            )

        exposure: List[Dict[str, str]] = []
        for pos in positions:
            symbol = str(pos.get("symbol", "")).upper()
            underlying, direction = self.option_exposure(symbol)
            if underlying:
                exposure.append(
                    {"symbol": symbol, "underlying": underlying, "direction": direction or ""}
                )
        for order in pending_entries:
            symbol = str(getattr(order, "symbol", "")).upper()
            underlying, direction = self.option_exposure(symbol)
            if underlying:
                exposure.append(
                    {"symbol": symbol, "underlying": underlying, "direction": direction or ""}
                )

        if any(
            item["underlying"] == desired_underlying
            and item["direction"] == desired_direction
            for item in exposure
        ):
            return self._failure(
                "underlying_direction_exposure",
                underlying_symbol=plan.underlying_symbol,
                direction=desired_direction,
                existing_exposure=exposure,
                option_plan=plan.to_dict(),
            )

        open_risk_count = len(positions) + len(pending_entries)
        max_concurrent = int(self.settings.trading.max_concurrent_trades)
        if open_risk_count >= max_concurrent:
            return self._failure(
                "max_concurrent_trades",
                open_positions=len(positions),
                pending_entries=len(pending_entries),
                max_concurrent_trades=max_concurrent,
                option_plan=plan.to_dict(),
            )

        cooldown_until = self._stopout_cooldowns.get(desired_underlying)
        now = datetime.now(timezone.utc)
        if cooldown_until and now < cooldown_until:
            return self._failure(
                "stopout_cooldown",
                underlying_symbol=desired_underlying,
                cooldown_until=cooldown_until.isoformat(),
                option_plan=plan.to_dict(),
            )
        return None

    def record_stopout(self, symbol: str, minutes: int = 60) -> None:
        underlying, _ = self.option_exposure(symbol)
        if underlying:
            self._stopout_cooldowns[underlying] = datetime.now(timezone.utc) + timedelta(
                minutes=max(1, int(minutes))
            )
            self._save_risk_state()

    def ensure_protective_stops(
        self,
        positions: Optional[List[Dict[str, Any]]] = None,
        order_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Attach a DAY broker stop to every open long-option position."""
        positions = positions if positions is not None else self.get_open_positions()
        order_meta = order_meta or {}
        events: List[Dict[str, Any]] = []
        option_positions = [
            pos
            for pos in positions
            if str(pos.get("asset_class", "")).lower() in ("option", "us_option")
        ]
        try:
            open_orders = self.get_open_orders(
                [str(pos.get("symbol")) for pos in option_positions if pos.get("symbol")],
                raise_on_error=True,
            )
        except Exception as exc:
            for pos in option_positions:
                event = {
                    "event_type": "option_protective_stop_failed",
                    "symbol": str(pos.get("symbol", "")).upper(),
                    "details": {
                        "error": f"broker_order_query_failed: {exc}",
                        "actual_filled_qty": abs(float(pos.get("qty", 0) or 0)),
                        "actual_entry_price": self._float_or_none(pos.get("entry_price")),
                    },
                }
                events.append(event)
                self._risk_events.append(event)
            return events

        for pos in option_positions:
            symbol = str(pos.get("symbol", "")).upper()
            qty = abs(float(pos.get("qty", 0) or 0))
            entry_price = self._float_or_none(pos.get("entry_price"))
            if not symbol or qty <= 0 or not entry_price:
                continue

            meta = order_meta.get(symbol, {})
            state = (
                (meta.get("tiered_exit_state") or meta.get("exit_state"))
                if isinstance(meta, dict)
                else None
            )
            if hasattr(state, "to_dict"):
                state = state.to_dict()
            elif isinstance(state, str):
                try:
                    state = json.loads(state)
                except json.JSONDecodeError:
                    state = None
            if isinstance(state, dict) and state.get("pending_stage") and int(state.get("pending_qty") or 0) > 0:
                # The tier sequence owns protection while a close is pending;
                # re-attachment happens only after confirmed fill/cancel.
                continue
            desired_stop = None
            if isinstance(state, dict):
                desired_stop = self._float_or_none(state.get("active_stop_price"))
            tiered_state_present = isinstance(state, dict) and bool(
                state.get("policy_version") == "tiered_v1"
            )
            stop_price = self._round_option_price(
                desired_stop
                if desired_stop is not None and desired_stop > 0
                else entry_price * (1.0 - float(self.options_settings.stop_loss_pct))
            )

            matching = [
                order
                for order in open_orders
                if str(getattr(order, "symbol", "")).upper() == symbol
                and self._is_protective_stop(order)
            ]
            if matching:
                protected_qty = sum(
                    abs(self._float_or_none(getattr(order, "qty", None)) or 0.0)
                    for order in matching
                )
                existing_prices = [
                    self._float_or_none(getattr(order, "stop_price", None))
                    for order in matching
                ]
                has_correct_price = any(
                    price is not None and abs(price - stop_price) < 0.005
                    for price in existing_prices
                )
                if abs(protected_qty - qty) < 1e-9 and (
                    has_correct_price or not tiered_state_present
                ):
                    continue
                cancel_result = self.cancel_protective_stop(symbol)
                if cancel_result.get("status") not in ("absent", "canceled"):
                    event = {
                        "event_type": "option_protective_stop_failed",
                        "symbol": symbol,
                        "details": {
                            "error": "protective_stop_qty_mismatch_cancel_failed",
                            "cancel_result": cancel_result,
                            "protected_qty": protected_qty,
                            "actual_filled_qty": qty,
                            "actual_entry_price": entry_price,
                        },
                    }
                    events.append(event)
                    self._risk_events.append(event)
                    continue

            client_order_id = self._protective_client_order_id(symbol)
            request = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                type=OrderType.STOP,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.SIMPLE,
                stop_price=stop_price,
                position_intent=PositionIntent.SELL_TO_CLOSE,
                client_order_id=client_order_id,
            )
            try:
                order = self.trading_client.submit_order(order_data=request)
                details = {
                    "stop_order_id": str(getattr(order, "id", "") or ""),
                    "entry_order_id": str(meta.get("order_id", "") or ""),
                    "actual_filled_qty": qty,
                    "actual_entry_price": entry_price,
                    "position": pos,
                    "stop_price": stop_price,
                    "stop_loss_pct": float(self.options_settings.stop_loss_pct),
                    "tiered_active_stop_price": desired_stop,
                    "protection_replacement": bool(matching),
                    "time_in_force": "day",
                }
                event = {
                    "event_type": "option_protective_stop_submitted",
                    "symbol": symbol,
                    "details": details,
                }
                events.append(event)
                self._risk_events.append(event)
                print(f"  > Protected {qty:g} {symbol} with stop @ ${stop_price:.2f}")
            except Exception as exc:
                event = {
                    "event_type": "option_protective_stop_failed",
                    "symbol": symbol,
                    "details": {
                        "error": str(exc),
                        "actual_filled_qty": qty,
                        "actual_entry_price": entry_price,
                        "stop_price": stop_price,
                    },
                }
                events.append(event)
                self._risk_events.append(event)
                print(f"  ! Failed to protect {symbol}: {exc}")
        return events

    def cancel_protective_stop(self, symbol: str) -> Dict[str, Any]:
        symbol = str(symbol).upper()
        try:
            open_orders = self.get_open_orders([symbol], raise_on_error=True)
        except Exception as exc:
            return {"status": "query_failed", "symbol": symbol, "error": str(exc)}
        stops = [
            order
            for order in open_orders
            if str(getattr(order, "symbol", "")).upper() == symbol
            and self._is_protective_stop(order)
        ]
        if not stops:
            return {"status": "absent", "symbol": symbol}

        for order in stops:
            order_id = str(getattr(order, "id", "") or "")
            status = self._order_status(order)
            if status == "filled":
                return {"status": "filled", "symbol": symbol, "stop_order_id": order_id}
            try:
                self.trading_client.cancel_order_by_id(order_id)
            except Exception as exc:
                return {
                    "status": "cancel_failed",
                    "symbol": symbol,
                    "stop_order_id": order_id,
                    "error": str(exc),
                }
            try:
                refreshed = self.trading_client.get_order_by_id(order_id)
                status = self._order_status(refreshed)
            except Exception:
                status = "pending_cancel"
            if status == "filled":
                return {"status": "filled", "symbol": symbol, "stop_order_id": order_id}
            if status not in _TERMINAL_CANCEL_STATUSES:
                return {
                    "status": status or "pending_cancel",
                    "symbol": symbol,
                    "stop_order_id": order_id,
                }
        return {"status": "canceled", "symbol": symbol}

    def get_latest_exit_fill(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return actual fill details for the most recently requested exit."""
        symbol = str(symbol).upper()
        order_id = self._last_exit_orders.get(symbol)
        protective = False
        if not order_id:
            try:
                request = GetOrdersRequest(
                    status=QueryOrderStatus.CLOSED,
                    limit=50,
                    nested=True,
                    symbols=[symbol],
                )
                closed_orders = list(self.trading_client.get_orders(filter=request))
                filled_stops = [
                    order
                    for order in closed_orders
                    if self._order_status(order) == "filled"
                    and self._is_protective_stop(order)
                ]
                if not filled_stops:
                    return None
                filled_stops.sort(
                    key=lambda order: str(getattr(order, "filled_at", "") or ""),
                    reverse=True,
                )
                order_id = str(getattr(filled_stops[0], "id", "") or "")
                protective = True
            except Exception:
                return None
        try:
            order = self.trading_client.get_order_by_id(order_id)
        except Exception:
            return {"order_id": order_id, "is_protective_stop": protective}
        return {
            "order_id": order_id,
            "status": self._order_status(order),
            "filled_qty": self._float_or_none(getattr(order, "filled_qty", None)),
            "filled_avg_price": self._float_or_none(
                getattr(order, "filled_avg_price", None)
            ),
            "filled_at": str(getattr(order, "filled_at", "") or ""),
            "is_protective_stop": protective or self._is_protective_stop(order),
        }

    @staticmethod
    def _tiered_client_order_id(
        symbol: str, stage: str, lifecycle_id: Optional[str] = None
    ) -> str:
        digest = hashlib.sha1(
            f"{lifecycle_id or symbol}:{str(stage).lower()}".encode("utf-8")
        ).hexdigest()[:14]
        return f"{_TIERED_EXIT_PREFIX}{str(stage).lower()}-{digest}"[:48]

    def pop_risk_events(self) -> List[Dict[str, Any]]:
        events = list(self._risk_events)
        self._risk_events.clear()
        return events

    @staticmethod
    def option_exposure(symbol: str) -> tuple[Optional[str], Optional[str]]:
        match = _OCC_OPTION_RE.fullmatch(str(symbol).upper().replace(" ", ""))
        if not match:
            return None, None
        direction = "long" if match.group("contract_type") == "C" else "short"
        return match.group("underlying"), direction

    @staticmethod
    def _order_status(order: Any) -> str:
        value = getattr(order, "status", "")
        return str(getattr(value, "value", value)).lower()

    @staticmethod
    def _order_side(order: Any) -> str:
        value = getattr(order, "side", "")
        return str(getattr(value, "value", value)).lower()

    @classmethod
    def _is_protective_stop(cls, order: Any) -> bool:
        order_type = getattr(order, "type", None) or getattr(order, "order_type", "")
        order_type = str(getattr(order_type, "value", order_type)).lower()
        client_order_id = str(getattr(order, "client_order_id", "") or "")
        return (
            cls._order_side(order) == "sell"
            and order_type in ("stop", "stop_limit")
            and (
                client_order_id.startswith(_PROTECTIVE_STOP_PREFIX)
                or str(getattr(order, "position_intent", "")).lower().endswith(
                    "sell_to_close"
                )
            )
        )

    @staticmethod
    def _protective_client_order_id(symbol: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%y%m%d%H%M%S")
        return f"{_PROTECTIVE_STOP_PREFIX}{symbol[-18:]}-{stamp}"[:48]

    @staticmethod
    def _round_option_price(price: float) -> float:
        return max(0.01, round(float(price) + 1e-9, 2))

    @staticmethod
    def _float_or_none(value: Any) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _load_risk_state(self) -> None:
        if not self._risk_state_path or not self._risk_state_path.exists():
            return
        try:
            payload = json.loads(self._risk_state_path.read_text(encoding="utf-8"))
            raw = payload.get("stopout_cooldowns", {})
            now = datetime.now(timezone.utc)
            for underlying, value in raw.items():
                cutoff = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                if cutoff.tzinfo is None:
                    cutoff = cutoff.replace(tzinfo=timezone.utc)
                if cutoff > now:
                    self._stopout_cooldowns[str(underlying).upper()] = cutoff
        except Exception as exc:
            print(f"  ! Failed to load risk state: {exc}")

    def _save_risk_state(self) -> None:
        if not self._risk_state_path:
            return
        try:
            self._risk_state_path.parent.mkdir(parents=True, exist_ok=True)
            self._risk_state_path.write_text(
                json.dumps(
                    {
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "stopout_cooldowns": {
                            key: value.isoformat()
                            for key, value in self._stopout_cooldowns.items()
                        },
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"  ! Failed to save risk state: {exc}")

    @staticmethod
    def _position_to_dict(pos: Any) -> Dict[str, Any]:
        symbol = str(getattr(pos, "symbol", ""))
        qty = float(getattr(pos, "qty", 0) or 0)
        market_value = float(getattr(pos, "market_value", 0) or 0)
        cost_basis = float(getattr(pos, "cost_basis", 0) or 0)
        raw_asset_class = str(getattr(pos, "asset_class", "option"))
        is_option = "option" in raw_asset_class.lower()
        multiplier = 100.0 if is_option else 1.0
        entry_price = abs(cost_basis) / (abs(qty) * multiplier) if qty else None
        current_price = abs(market_value) / (abs(qty) * multiplier) if qty else None
        return {
            "symbol": symbol,
            "option_symbol": symbol if is_option else None,
            "qty": qty,
            "side": getattr(pos, "side", ""),
            "market_value": market_value,
            "cost_basis": cost_basis,
            "unrealized_pl": float(getattr(pos, "unrealized_pl", 0) or 0),
            "unrealized_plpc": float(getattr(pos, "unrealized_plpc", 0) or 0),
            "entry_price": entry_price,
            "current_price": current_price,
            "asset_class": "option" if is_option else raw_asset_class,
        }
