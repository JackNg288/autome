#!/usr/bin/env python3
"""
Enhanced Trading Bot v4.0 with New Features:
- Adjustable minimum confidence (now 50%)
- Full symbols.txt monitoring
- Manual analysis command (/analysis SYMBOL DIRECTION PRICE)
- Bybit Futures trading support with 2x leverage
- $100 position sizing per trade
- FIXED: Working Bybit API integration with demo trading support
"""

import requests
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
import logging
import threading
import time
from datetime import datetime, timedelta
import sqlite3
import json
import hashlib
import hmac
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import warnings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot_v4.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedBybitFuturesClient:
    """Enhanced Bybit Futures API client with improved order placement"""
    
    def __init__(self, api_key=None, api_secret=None, demo_trading=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.demo_trading = demo_trading
        
        # Base URLs
        if self.demo_trading:
            self.base_url = "https://api-demo.bybit.com"
            logger.info("Using Bybit Demo Trading environment")
        else:
            self.base_url = "https://api.bybit.com"
            logger.info("Using Bybit Live Trading environment")
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TradingBot/5.0-Enhanced',
            'Content-Type': 'application/json'
        })
        
        # Default position sizing
        self.position_size_usd = 100  # Default $100 per trade
        self.leverage = 2  # Default 2x leverage
        
        # Symbol cache for instrument info
        self.symbol_cache = {}
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test API connection and authentication"""
        try:
            # Test public endpoint
            public_test = self._test_public_connection()
            logger.info(f"Public API connection: {'✅ OK' if public_test else '❌ Failed'}")
            
            # Test private endpoint if API keys provided
            if self.api_key and self.api_secret:
                auth_test = self._test_authentication()
                logger.info(f"Authentication: {'✅ OK' if auth_test else '❌ Failed'}")
                
                # For demo trading, ensure demo funds are available
                if self.demo_trading and auth_test:
                    self._ensure_demo_funds()
            else:
                logger.warning("No API keys provided - running in simulation mode")
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
    
    def _test_public_connection(self) -> bool:
        """Test public API connection"""
        try:
            endpoint = "/v5/market/time"
            response = self.session.get(f"{self.base_url}{endpoint}", timeout=10)
            return response.status_code == 200 and response.json().get("retCode") == 0
        except:
            return False
    
    def _test_authentication(self) -> bool:
        """Test API authentication"""
        try:
            result = self.get_api_key_info()
            return "error" not in result and result.get("retCode") == 0
        except:
            return False
    
    def _ensure_demo_funds(self):
        """Ensure demo account has sufficient funds"""
        try:
            balance = self.get_account_balance()
            if balance.get("summary", {}).get("totalUSDTBalance", 0) < 1000:
                logger.info("Low demo balance detected, requesting additional funds...")
                self.request_demo_funds()
        except Exception as e:
            logger.warning(f"Could not check/add demo funds: {e}")
    
    def _generate_signature(self, payload_str: str, timestamp: str = None, recv_window: str = "5000"):
        """Generate HMAC signature for Bybit v5 API"""
        if not self.api_key or not self.api_secret:
            return "", "", recv_window
            
        ts = timestamp or str(int(time.time() * 1000))
        prehash = ts + self.api_key + recv_window + payload_str
        sig = hmac.new(
            self.api_secret.encode("utf-8"),
            prehash.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return sig, ts, recv_window
    
    def _qs(self, d: dict) -> str:
        """Generate sorted query string"""
        return "&".join(f"{k}={v}" for k, v in sorted(d.items()))
    
    def _j(self, d: dict) -> str:
        """Generate minified JSON"""
        return json.dumps(d, separators=(",", ":"))
    
    def get_symbol_info(self, symbol: str, force_refresh: bool = False) -> Dict:
        """Get and cache symbol trading rules"""
        if not force_refresh and symbol in self.symbol_cache:
            return self.symbol_cache[symbol]
        
        try:
            endpoint = "/v5/market/instruments-info"
            params = {"category": "linear", "symbol": symbol}
            
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("retCode") == 0:
                    instruments = data.get("result", {}).get("list", [])
                    if instruments:
                        self.symbol_cache[symbol] = instruments[0]
                        return instruments[0]
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return {}
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               position_size_usd: Optional[float] = None,
                               leverage: Optional[int] = None) -> Dict:
        """Calculate position size with proper precision"""
        try:
            # Use provided values or defaults
            size_usd = position_size_usd or self.position_size_usd
            lev = leverage or self.leverage
            
            # Get symbol info for precision rules
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return {"error": "Failed to get symbol info"}
            
            # Extract trading rules
            lot_size_filter = symbol_info.get("lotSizeFilter", {})
            qty_step = float(lot_size_filter.get("qtyStep", "0.001"))
            min_qty = float(lot_size_filter.get("minOrderQty", "0.001"))
            max_qty = float(lot_size_filter.get("maxOrderQty", "10000"))
            
            price_filter = symbol_info.get("priceFilter", {})
            tick_size = float(price_filter.get("tickSize", "0.01"))
            
            # Calculate raw quantity
            position_value = float(size_usd) * float(lev)
            raw_quantity = position_value / float(entry_price)
            
            # Round quantity to proper precision (using floor to avoid exceeding balance)
            quantity = self._round_to_precision(raw_quantity, qty_step, mode="floor")
            
            # Ensure within limits
            quantity = max(min_qty, min(quantity, max_qty))
            
            # Calculate actual values
            actual_position_value = quantity * float(entry_price)
            required_margin = actual_position_value / float(lev)
            
            # Validate minimum notional value
            min_notional = float(lot_size_filter.get("minNotionalValue", "5"))
            if actual_position_value < min_notional:
                return {
                    "error": f"Position value ${actual_position_value:.2f} below minimum ${min_notional:.2f}"
                }
            
            return {
                "symbol": symbol,
                "quantity": quantity,
                "position_value": actual_position_value,
                "required_margin": required_margin,
                "leverage": lev,
                "qty_step": qty_step,
                "tick_size": tick_size,
                "min_qty": min_qty,
                "max_qty": max_qty
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {"error": str(e)}
    
    def _round_to_precision(self, value: float, precision: float, mode: str = "floor") -> float:
        """Round value to specified precision"""
        from decimal import Decimal, ROUND_DOWN, ROUND_UP
        
        d_value = Decimal(str(value))
        d_precision = Decimal(str(precision))
        
        if d_precision == 0:
            return float(value)
        
        quotient = d_value / d_precision
        
        if mode == "ceil":
            quotient = quotient.to_integral_value(rounding=ROUND_UP)
        else:  # floor
            quotient = quotient.to_integral_value(rounding=ROUND_DOWN)
        
        return float(quotient * d_precision)
    
    def set_leverage(self, symbol: str, leverage: Optional[int] = None) -> Dict:
        """Set leverage for a symbol"""
        try:
            if not self.api_key or not self.api_secret:
                return {"status": "simulation", "message": "No API keys"}
            
            lev = leverage or self.leverage
            endpoint = "/v5/position/set-leverage"
            
            params = {
                "category": "linear",
                "symbol": symbol,
                "buyLeverage": str(lev),
                "sellLeverage": str(lev)
            }
            
            payload_str = self._j(params)
            signature, timestamp, recv_window = self._generate_signature(payload_str)
            
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": recv_window,
                "X-BAPI-SIGN": signature,
                "Content-Type": "application/json"
            }
            
            response = self.session.post(
                f"{self.base_url}{endpoint}",
                data=payload_str,
                headers=headers,
                timeout=10
            )
            
            data = response.json()
            if data.get("retCode") == 0:
                logger.info(f"Leverage set to {lev}x for {symbol}")
                return {"status": "success", "leverage": lev}
            else:
                # Some symbols might not support leverage adjustment
                if "leverage not modified" in str(data.get("retMsg", "")).lower():
                    return {"status": "success", "leverage": lev, "note": "Already set"}
                return {"status": "error", "error": data.get("retMsg", "Unknown error")}
                
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return {"status": "error", "error": str(e)}
    
    def place_futures_order(self, symbol: str, side: str, 
                          entry_price: Optional[float] = None,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None,
                          orderType: str = None,
                          position_size_usd: Optional[float] = None,
                          leverage: Optional[int] = None,
                          reduce_only: bool = False,
                          time_in_force: str = None) -> Dict:
        """Enhanced order placement with better error handling"""
        try:
            # Simulation mode if no API keys
            if not self.api_key or not self.api_secret:
                size_info = self.calculate_position_size(
                    symbol, entry_price or 1.0, position_size_usd, leverage
                )
                return {
                    "status": "simulation",
                    "symbol": symbol,
                    "side": side,
                    "qty": size_info.get("quantity", 0),
                    "message": "No API keys - simulation mode"
                }
            
            # Normalize side
            side_upper = side.upper()
            side_map = {
                "LONG": "Buy",
                "SHORT": "Sell",
                "BUY": "Buy",
                "SELL": "Sell"
            }
            bybit_side = side_map.get(side_upper, "Buy")
            
            # Determine order type
            if orderType:
                order_type = orderType
            elif entry_price:
                order_type = "Limit"
            else:
                order_type = "Market"
            
            # For market orders, get current price for position sizing
            if order_type == "Market" and not entry_price:
                ticker = self.get_ticker_price(symbol)
                if ticker.get("error"):
                    return {"status": "error", "error": f"Cannot get price: {ticker['error']}"}
                entry_price = ticker.get("price", 0)
            
            # Set leverage
            lev = leverage or self.leverage
            leverage_result = self.set_leverage(symbol, lev)
            if leverage_result.get("status") == "error":
                logger.warning(f"Leverage setting warning: {leverage_result.get('error')}")
            
            # Calculate position size
            size_info = self.calculate_position_size(
                symbol, entry_price, position_size_usd, lev
            )
            if "error" in size_info:
                return {"status": "error", "error": size_info["error"]}
            
            quantity = size_info["quantity"]
            tick_size = size_info.get("tick_size", 0.01)
            
            # Build order parameters
            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": bybit_side,
                "orderType": order_type,
                "qty": str(quantity),
                "reduceOnly": reduce_only
            }
            
            # Add price for limit orders
            if order_type == "Limit":
                order_params["price"] = str(self._round_to_precision(entry_price, tick_size))
                order_params["timeInForce"] = time_in_force or "GTC"
            else:
                order_params["timeInForce"] = "IOC"  # Immediate or Cancel for market orders
            
            # Add TP/SL if provided
            if stop_loss and stop_loss > 0:
                order_params["stopLoss"] = str(self._round_to_precision(stop_loss, tick_size))
            
            if take_profit and take_profit > 0:
                order_params["takeProfit"] = str(self._round_to_precision(take_profit, tick_size))
            
            # Position mode
            order_params["positionIdx"] = 0  # One-way mode
            
            # Send order
            endpoint = "/v5/order/create"
            payload_str = self._j(order_params)
            signature, timestamp, recv_window = self._generate_signature(payload_str)
            
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": recv_window,
                "X-BAPI-SIGN": signature,
                "Content-Type": "application/json"
            }
            
            logger.info(f"Placing order: {symbol} {bybit_side} {quantity} @ {order_type}")
            
            response = self.session.post(
                f"{self.base_url}{endpoint}",
                data=payload_str,
                headers=headers,
                timeout=15
            )
            
            data = response.json()
            
            if data.get("retCode") == 0:
                result = data.get("result", {})
                return {
                    "status": "success",
                    "order_id": result.get("orderId"),
                    "symbol": symbol,
                    "side": bybit_side,
                    "qty": quantity,
                    "order_type": order_type,
                    "data": result
                }
            else:
                error_msg = data.get("retMsg", "Unknown error")
                error_code = data.get("retCode")
                
                # Handle specific error codes
                if error_code == 10001:
                    return {"status": "error", "error": "Invalid parameters", "details": error_msg}
                elif error_code == 10002:
                    return {"status": "error", "error": "Invalid API key", "details": error_msg}
                elif error_code == 10003:
                    return {"status": "error", "error": "Invalid signature", "details": error_msg}
                elif error_code == 10004:
                    return {"status": "error", "error": "Invalid timestamp", "details": error_msg}
                elif error_code == 110003:
                    return {"status": "error", "error": "Insufficient balance", "details": error_msg}
                elif error_code == 110004:
                    return {"status": "error", "error": "Insufficient available balance", "details": error_msg}
                elif error_code == 110025:
                    return {"status": "error", "error": "Position mode not set", "details": error_msg}
                elif error_code == 110026:
                    return {"status": "error", "error": "Cross/isolated margin mode error", "details": error_msg}
                else:
                    return {"status": "error", "error": error_msg, "code": error_code}
            
        except requests.exceptions.RequestException as e:
            return {"status": "error", "error": f"Network error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_ticker_price(self, symbol: str) -> Dict:
        """Get current ticker price"""
        try:
            endpoint = "/v5/market/tickers"
            params = {"category": "linear", "symbol": symbol}
            
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("retCode") == 0:
                    tickers = data.get("result", {}).get("list", [])
                    if tickers:
                        return {"price": float(tickers[0].get("lastPrice", 0))}
            
            return {"error": "Failed to get ticker price"}
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_open_orders(self, symbol: Optional[str] = None) -> Dict:
        """Get open orders"""
        try:
            if not self.api_key or not self.api_secret:
                return {"status": "error", "error": "No API keys"}
            
            endpoint = "/v5/order/realtime"
            params = {"category": "linear"}
            if symbol:
                params["symbol"] = symbol
            
            query_string = self._qs(params)
            signature, timestamp, recv_window = self._generate_signature(query_string)
            
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": recv_window,
                "X-BAPI-SIGN": signature,
                "Content-Type": "application/json"
            }
            
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                params=params,
                headers=headers,
                timeout=10
            )
            
            data = response.json()
            if data.get("retCode") == 0:
                return {
                    "status": "success",
                    "orders": data.get("result", {}).get("list", [])
                }
            else:
                return {"status": "error", "error": data.get("retMsg", "Unknown error")}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel an order"""
        try:
            if not self.api_key or not self.api_secret:
                return {"status": "error", "error": "No API keys"}
            
            endpoint = "/v5/order/cancel"
            params = {
                "category": "linear",
                "symbol": symbol,
                "orderId": order_id
            }
            
            payload_str = self._j(params)
            signature, timestamp, recv_window = self._generate_signature(payload_str)
            
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": recv_window,
                "X-BAPI-SIGN": signature,
                "Content-Type": "application/json"
            }
            
            response = self.session.post(
                f"{self.base_url}{endpoint}",
                data=payload_str,
                headers=headers,
                timeout=10
            )
            
            data = response.json()
            if data.get("retCode") == 0:
                return {"status": "success", "order_id": order_id}
            else:
                return {"status": "error", "error": data.get("retMsg", "Unknown error")}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_positions(self, symbol: Optional[str] = None) -> Dict:
        """Get open positions"""
        try:
            if not self.api_key or not self.api_secret:
                return {"status": "error", "error": "No API keys"}
            
            endpoint = "/v5/position/list"
            params = {"category": "linear", "settleCoin": "USDT"}
            if symbol:
                params["symbol"] = symbol
            
            query_string = self._qs(params)
            signature, timestamp, recv_window = self._generate_signature(query_string)
            
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": recv_window,
                "X-BAPI-SIGN": signature,
                "Content-Type": "application/json"
            }
            
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                params=params,
                headers=headers,
                timeout=10
            )
            
            data = response.json()
            if data.get("retCode") == 0:
                positions = data.get("result", {}).get("list", [])
                # Filter out positions with size = 0
                active_positions = [p for p in positions if float(p.get("size", 0)) > 0]
                return {
                    "status": "success",
                    "positions": active_positions
                }
            else:
                return {"status": "error", "error": data.get("retMsg", "Unknown error")}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def close_position(self, symbol: str, reduce_only: bool = True) -> Dict:
        """Close a position by placing a reduce-only market order"""
        try:
            # Get current position
            positions_result = self.get_positions(symbol)
            if positions_result.get("status") != "success":
                return positions_result
            
            positions = positions_result.get("positions", [])
            if not positions:
                return {"status": "error", "error": "No position to close"}
            
            position = positions[0]
            size = float(position.get("size", 0))
            side = position.get("side", "")
            
            if size == 0:
                return {"status": "error", "error": "Position size is 0"}
            
            # Opposite side to close
            close_side = "Sell" if side == "Buy" else "Buy"
            
            # Place reduce-only market order
            return self.place_futures_order(
                symbol=symbol,
                side=close_side,
                orderType="Market",
                position_size_usd=None,  # Will use actual position size
                reduce_only=True
            )
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_account_balance(self) -> Dict:
        """Get account balance"""
        try:
            if not self.api_key or not self.api_secret:
                return {"status": "no_auth", "message": "No API keys"}
            
            endpoint = "/v5/account/wallet-balance"
            account_type = "UNIFIED" if self.demo_trading else "CONTRACT"
            
            params = {"accountType": account_type}
            query_string = self._qs(params)
            
            signature, timestamp, recv_window = self._generate_signature(query_string)
            
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": recv_window,
                "X-BAPI-SIGN": signature,
                "Content-Type": "application/json"
            }
            
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                params=params,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("retCode") == 0:
                    return self._format_balance_response(data, account_type)
            
            return {"status": "error", "error": "Failed to get balance"}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _format_balance_response(self, data: Dict, account_type: str) -> Dict:
        """Format balance response"""
        result = data.get("result", {})
        accounts = result.get("list", [])
        
        summary = {
            "totalUSDTBalance": 0,
            "totalAvailable": 0,
            "totalEquity": 0,
            "totalUnrealisedPnl": 0,
            "hasBalance": False
        }
        
        for account in accounts:
            # Get totals
            try:
                summary["totalEquity"] += float(account.get("totalEquity", 0) or 0)
                summary["totalAvailable"] += float(account.get("totalAvailableBalance", 0) or 0)
                
                # Get coin balances
                for coin in account.get("coin", []):
                    if coin.get("coin") == "USDT":
                        summary["totalUSDTBalance"] = float(coin.get("walletBalance", 0) or 0)
                        if summary["totalUSDTBalance"] > 0:
                            summary["hasBalance"] = True
                    
                    unrealised = float(coin.get("unrealisedPnl", 0) or 0)
                    summary["totalUnrealisedPnl"] += unrealised
                    
            except (ValueError, TypeError):
                continue
        
        return {
            "status": "success",
            "accountType": account_type,
            "summary": summary,
            "accounts": accounts
        }
    
    def request_demo_funds(self, coins: list = None) -> Dict:
        """Request demo trading funds"""
        try:
            if not self.demo_trading:
                return {"status": "error", "error": "Only available for demo trading"}
            
            if not self.api_key or not self.api_secret:
                return {"status": "error", "error": "API keys required"}
            
            # Default demo funds
            if not coins:
                coins = [
                    {"coin": "USDT", "amountStr": "50000"},
                    {"coin": "USDC", "amountStr": "50000"}
                ]
            
            endpoint = "/v5/account/demo-apply-money"
            params = {
                "adjustType": 0,  # 0 = add funds
                "utaDemoApplyMoney": coins
            }
            
            payload_str = self._j(params)
            signature, timestamp, recv_window = self._generate_signature(payload_str)
            
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": recv_window,
                "X-BAPI-SIGN": signature,
                "Content-Type": "application/json"
            }
            
            response = self.session.post(
                f"{self.base_url}{endpoint}",
                data=payload_str,
                headers=headers,
                timeout=10
            )
            
            data = response.json()
            if data.get("retCode") == 0:
                logger.info("Demo funds added successfully")
                return {"status": "success", "message": "Demo funds added"}
            else:
                error_msg = data.get("retMsg", "Unknown error")
                if "sufficient" in error_msg.lower():
                    return {"status": "success", "message": "Funds already sufficient"}
                return {"status": "error", "error": error_msg}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_api_key_info(self) -> Dict:
        """Get API key information"""
        try:
            endpoint = "/v5/user/query-api"
            payload_str = ""
            signature, timestamp, recv_window = self._generate_signature(payload_str)
            
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": recv_window,
                "X-BAPI-SIGN": signature,
                "Content-Type": "application/json"
            }
            
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                headers=headers,
                timeout=10
            )
            
            return response.json()
            
        except Exception as e:
            return {"error": str(e)}


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize client
    client = EnhancedBybitFuturesClient(
        api_key=os.getenv("BYBIT_API_KEY"),
        api_secret=os.getenv("BYBIT_API_SECRET"),
        demo_trading=True
    )
    
    # Test functions
    print("\n=== Testing Enhanced Bybit Client ===\n")
    
    # 1. Check balance
    print("1. Checking balance...")
    balance = client.get_account_balance()
    if balance.get("status") == "success":
        summary = balance.get("summary", {})
        print(f"   ✅ USDT Balance: ${summary.get('totalUSDTBalance', 0):,.2f}")
        print(f"   Available: ${summary.get('totalAvailable', 0):,.2f}")
    else:
        print(f"   ❌ Error: {balance.get('error')}")
    
    # 2. Get symbol info
    print("\n2. Getting BTCUSDT info...")
    symbol_info = client.get_symbol_info("BTCUSDT")
    if symbol_info:
        print(f"   ✅ Symbol: {symbol_info.get('symbol')}")
        print(f"   Status: {symbol_info.get('status')}")
    else:
        print("   ❌ Failed to get symbol info")
    
    # 3. Calculate position size
    print("\n3. Calculating position size...")
    size_info = client.calculate_position_size("BTCUSDT", 65000, 100, 2)
    if "error" not in size_info:
        print(f"   ✅ Quantity: {size_info['quantity']:.6f} BTC")
        print(f"   Position Value: ${size_info['position_value']:.2f}")
        print(f"   Required Margin: ${size_info['required_margin']:.2f}")
    else:
        print(f"   ❌ Error: {size_info['error']}")
    
    # 4. Test order placement (simulation)
    print("\n4. Testing order placement...")
    order_result = client.place_futures_order(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=65000,
        stop_loss=63000,
        take_profit=67000,
        orderType="Limit",
        position_size_usd=100,
        leverage=2
    )
    if order_result.get("status") == "success":
        print(f"   ✅ Order placed: {order_result.get('order_id')}")
    elif order_result.get("status") == "simulation":
        print(f"   🧪 Simulation: Would place {order_result.get('qty')} BTC")
    else:
        print(f"   ❌ Error: {order_result.get('error')}")
    
    # 5. Get open positions
    print("\n5. Checking open positions...")
    positions = client.get_positions()
    if positions.get("status") == "success":
        pos_list = positions.get("positions", [])
        if pos_list:
            for pos in pos_list:
                print(f"   📊 {pos['symbol']}: {pos['side']} {pos['size']} @ {pos['avgPrice']}")
        else:
            print("   No open positions")
    else:
        print(f"   ❌ Error: {positions.get('error')}")
    
    print("\n=== Test Complete ===")


# Enhanced Telegram command handler integration


warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot_v4.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



class EnhancedTelegramHandler:
    """Enhanced Telegram handler with manual analysis commands"""

    def __init__(self, token, chat_id, bot_instance):
        self.token = token
        self.chat_id = str(chat_id)
        self.bot_instance = bot_instance
        self.last_update_id = None
        self.running = True

    def start_listener(self):
        """Start the Telegram command listener in a separate thread"""
        listener_thread = threading.Thread(target=self._listen_for_commands, daemon=True)
        listener_thread.start()
        logger.info("Enhanced Telegram listener started")

    def _listen_for_commands(self):
        """Listen for Telegram commands"""
        while self.running:
            try:
                url = f"https://api.telegram.org/bot{self.token}/getUpdates"
                params = {"timeout": 10}
                if self.last_update_id:
                    params["offset"] = self.last_update_id + 1

                r = requests.get(url, params=params, timeout=15)
                updates = r.json().get("result", [])

                for update in updates:
                    self.last_update_id = update.get("update_id", self.last_update_id)
                    message = update.get("message", {}) or {}
                    chat = message.get("chat", {}) or {}
                    from_chat_id = str(chat.get("id", ""))
                    text = (message.get("text", "") or "").strip()

                    if not text:
                        continue
                    if from_chat_id != self.chat_id:
                        continue

                    self.process_command(text)

            except Exception as e:
                logger.error(f"Balance fetch error: {e}")

            time.sleep(3)

    def process_command(self, command: str):
        """Process telegram commands including analysis requests"""
        try:
            cmd_lower = command.lower()

            # Analysis command: /analysis SYMBOL DIRECTION PRICE
            if cmd_lower.startswith("/analysis "):
                self.handle_analysis_command(command)

            elif cmd_lower == "/start":
                self.bot_instance.running = True
                self.send_message("🤖 Bot v4.0 started!")

            elif cmd_lower == "/stop":
                self.bot_instance.running = False
                self.send_message("🛑 Bot stopped! Send /start to resume.")

            elif cmd_lower == "/status":
                self.send_status_update()

            elif cmd_lower == "/help":
                self.send_help_message()

            elif cmd_lower == "/stats":
                self.send_statistics()

            elif cmd_lower == "/active":
                self.send_active_positions()

            elif cmd_lower == "/balance":
                self.send_balance_info()

            elif cmd_lower == "/symbols":
                self.send_symbol_list()

            
            
            elif cmd_lower.startswith("/order "):
                # Format: /order SYMBOL DIRECTION [PRICE] [USD_SIZE] [LEVERAGE]
                try:
                    parts = command.split()
                    if len(parts) < 3:
                        self.send_message("❌ Use: `/order SYMBOL DIRECTION [PRICE] [USD_SIZE] [LEVERAGE]`")
                    else:
                        _, sym, dirn, *rest = parts
                        sym = sym.upper()
                        dirn = dirn.upper()
                        if not sym.endswith('USDT'):
                            sym = sym + 'USDT'
                        price = None
                        usd_size = None
                        lev = None
                        if len(rest) >= 1:
                            try: price = float(rest[0])
                            except: price = None
                        if len(rest) >= 2:
                            try: usd_size = float(rest[1])
                            except: usd_size = None
                        if len(rest) >= 3:
                            try: lev = int(float(rest[2]))
                            except: lev = None

                        if price is None:
                            # fetch current price from bot (Binance/MEXC fallback)
                            try:
                                price = float(self.bot_instance.fetch_price(sym))
                            except Exception:
                                price = None

                        # SL/TP from analyzer heuristic
                        stops = self.bot_instance.analyzer.calculate_dynamic_stop_loss(None, dirn, price or 0)
                        sl = stops.get("stop_loss", None)
                        tps = stops.get("targets", [])
                        tp = tps[1] if len(tps) > 1 else (price * (1.02 if dirn == "LONG" else 0.98) if price else None)

                        if not self.bot_instance.bybit_client:
                            self.send_message("⚠️ Bybit not configured in this bot")
                        else:
                            res = self.bot_instance.bybit_client.place_futures_order(
                                symbol=sym, side=dirn, entry_price=price,
                                stop_loss=sl, take_profit=tp,
                                orderType=None,  # auto: Limit if price else Market
                                position_size_usd=usd_size, leverage=lev
                            )
                            if res.get("status") == "success":
                                self.send_message(f"✅ Order placed: `{sym}` {dirn} qty `{res.get('qty')}` @ `{price}`")
                            elif res.get("status") == "simulation":
                                self.send_message(f"🧪 Simulated order: `{sym}` {dirn} qty `{res.get('qty')}` @ `{price}` (no API keys)")
                            else:
                                self.send_message(f"❌ Order error: {res.get('error')}")
                except Exception as e:
                    self.send_message(f"❌ /order failed: {e}")
            
            elif cmd_lower.startswith("/ordermarket "):
                # /ordermarket SYMBOL DIRECTION [USD_SIZE] [LEVERAGE]
                try:
                    parts = command.split()
                    if len(parts) < 3:
                        self.send_message("❌ Use: `/ordermarket SYMBOL DIRECTION [USD_SIZE] [LEVERAGE]`")
                    else:
                        _, sym, dirn, *rest = parts
                        sym = sym.upper()
                        if not sym.endswith("USDT"):
                            sym = sym + "USDT"
                        dirn = dirn.upper()
                        usd_size = float(rest[0]) if len(rest) >= 1 else None
                        lev = int(float(rest[1])) if len(rest) >= 2 else None
                        price = float(self.bot_instance.fetch_price(sym))
                        stops = self.bot_instance.analyzer.calculate_dynamic_stop_loss(None, dirn, price or 0)
                        sl = stops.get("stop_loss", None)
                        tps = stops.get("targets", [])
                        tp = tps[1] if len(tps) > 1 else None
                        res = self.bot_instance.bybit_client.place_futures_order(
                            symbol=sym, side=dirn, entry_price=price, orderType="Market",
                            stop_loss=sl, take_profit=tp, position_size_usd=usd_size, leverage=lev
                        )
                        if res.get("status") == "success":
                            self.send_message(f"✅ Market order: `{sym}` {dirn} qty `{res.get('qty')}`")
                        elif res.get("status") == "simulation":
                            self.send_message(f"🧪 Simulated market: `{sym}` {dirn} qty `{res.get('qty')}` (no API keys)")
                        else:
                            self.send_message(f"❌ Order error: {res.get('error')}")
                except Exception as e:
                    self.send_message(f"❌ /ordermarket failed: {e}")

            elif cmd_lower.startswith("/orderlimit "):
                # /orderlimit SYMBOL DIRECTION PRICE [USD_SIZE] [LEVERAGE]
                try:
                    parts = command.split()
                    if len(parts) < 4:
                        self.send_message("❌ Use: `/orderlimit SYMBOL DIRECTION PRICE [USD_SIZE] [LEVERAGE]`")
                    else:
                        _, sym, dirn, p, *rest = parts
                        sym = sym.upper()
                        if not sym.endswith("USDT"):
                            sym = sym + "USDT"
                        dirn = dirn.upper()
                        price = float(p)
                        usd_size = float(rest[0]) if len(rest) >= 1 else None
                        lev = int(float(rest[1])) if len(rest) >= 2 else None
                        stops = self.bot_instance.analyzer.calculate_dynamic_stop_loss(None, dirn, price or 0)
                        sl = stops.get("stop_loss", None)
                        tps = stops.get("targets", [])
                        tp = tps[1] if len(tps) > 1 else None
                        res = self.bot_instance.bybit_client.place_futures_order(
                            symbol=sym, side=dirn, entry_price=price, orderType="Limit",
                            stop_loss=sl, take_profit=tp, position_size_usd=usd_size, leverage=lev
                        )
                        if res.get("status") == "success":
                            self.send_message(f"✅ Limit order: `{sym}` {dirn} qty `{res.get('qty')}` @ `{price}`")
                        elif res.get("status") == "simulation":
                            self.send_message(f"🧪 Simulated limit: `{sym}` {dirn} qty `{res.get('qty')}` @ `{price}` (no API keys)")
                        else:
                            self.send_message(f"❌ Order error: {res.get('error')}")
                except Exception as e:
                    self.send_message(f"❌ /orderlimit failed: {e}")

            elif cmd_lower.startswith("/size "):
                try:
                    parts = command.split()
                    if len(parts) != 2:
                        self.send_message("❌ Use: `/size USD_AMOUNT` e.g. `/size 1000`")
                    else:
                        val = float(parts[1])
                        self.bot_instance.bybit_client.position_size_usd = val
                        self.send_message(f"✅ Default position size set to `${val:.2f}`")
                except Exception as e:
                    self.send_message(f"❌ /size failed: {e}")

            elif cmd_lower.startswith("/lev "):
                try:
                    parts = command.split()
                    if len(parts) != 2:
                        self.send_message("❌ Use: `/lev X` e.g. `/lev 2`")
                    else:
                        lv = int(float(parts[1]))
                        self.bot_instance.bybit_client.leverage = lv
                        self.send_message(f"✅ Default leverage set to `{lv}x`")
                except Exception as e:
                    self.send_message(f"❌ /lev failed: {e}")
            elif cmd_lower.startswith("/confidence "):
                self.handle_confidence_adjustment(command)

        except Exception as e:
            logger.error(f"Error processing command: {e}")
            self.send_message(f"❌ Error processing command: {str(e)}")

    def handle_analysis_command(self, command: str):
        """Handle manual analysis command: /analysis SYMBOL DIRECTION PRICE"""
        try:
            parts = command.split()
            if len(parts) != 4:
                self.send_message(
                    "❌ Invalid format!\n"
                    "Use: `/analysis SYMBOL DIRECTION PRICE`\n"
                    "Example: `/analysis BTCUSDT LONG 118500`"
                )
                return

            _, symbol, direction, price_str = parts
            symbol = symbol.upper()
            direction = direction.upper()

            # Validate inputs
            if direction not in ["LONG", "SHORT"]:
                self.send_message("❌ Direction must be LONG or SHORT")
                return

            try:
                entry_price = float(price_str)
            except ValueError:
                self.send_message("❌ Invalid price format")
                return

            # Check if symbol is valid
            if not symbol.endswith("USDT"):
                symbol = symbol + "USDT"

            self.send_message(f"🔍 Analyzing {symbol} {direction} @ ${entry_price:.2f}...")

            # Perform comprehensive analysis
            analysis_result = self.bot_instance.perform_manual_analysis(
                symbol, direction, entry_price
            )

            if analysis_result:
                self.send_analysis_result(analysis_result)
                # Auto-place order if confidence >= 50 and Bybit is configured
                try:
                    if self.bot_instance and self.bot_instance.bybit_client and analysis_result.get('confidence', 0) >= 50:
                        signal_data = {
                            'symbol': analysis_result['symbol'],
                            'direction': analysis_result['direction'],
                            'entry_price': analysis_result['entry_price'],
                            'confidence': analysis_result['confidence']
                        }
                        self.bot_instance.process_signal_with_bybit(signal_data, analysis_result)
                except Exception as _e:
                    logger.error(f"Manual auto-order error: {_e}")
            else:
                self.send_message("❌ Analysis failed. Please check if symbol is valid.")

        except Exception as e:
            logger.error(f"Error in analysis command: {e}")
            self.send_message(f"❌ Analysis error: {str(e)}")

    def send_analysis_result(self, analysis: Dict):
        """Send formatted analysis result"""
        symbol = analysis['symbol']
        direction = analysis['direction']
        entry_price = analysis['entry_price']
        confidence = analysis['confidence']

        # Determine recommendation
        if confidence >= 75:
            recommendation = "✅ STRONG SIGNAL"
            emoji = "🟢"
        elif confidence >= 60:
            recommendation = "⚠️ MODERATE SIGNAL"
            emoji = "🟡"
        elif confidence >= 50:
            recommendation = "⚠️ WEAK SIGNAL"
            emoji = "🟠"
        else:
            recommendation = "❌ NOT RECOMMENDED"
            emoji = "🔴"

        message = (
            f"📊 *MANUAL ANALYSIS RESULT*\n\n"
            f"Symbol: `{symbol}`\n"
            f"Direction: *{direction}*\n"
            f"Entry Price: `${entry_price:.4f}`\n\n"
            f"🎯 *Confidence Score: {emoji} {confidence:.1f}%*\n"
            f"📝 {recommendation}\n\n"
        )

        # Add technical details
        if 'technical' in analysis:
            tech = analysis['technical']
            message += (
                f"📈 *Technical Analysis:*\n"
                f"RSI: `{tech.get('rsi', 0):.1f}`\n"
                f"MACD: `{tech.get('macd_signal', 'N/A')}`\n"
                f"EMA Alignment: `{tech.get('ema_aligned', False)}`\n"
                f"Volume Ratio: `{tech.get('volume_ratio', 1):.2f}x`\n\n"
            )

        # Add advanced metrics
        if 'advanced' in analysis:
            adv = analysis['advanced']
            message += (
                f"🔍 *Advanced Metrics:*\n"
                f"Market Structure: `{adv.get('market_structure', 'UNKNOWN')}`\n"
                f"BTC Correlation: `{adv.get('btc_correlation', 'UNKNOWN')}`\n"
                f"MTF Confluence: `{adv.get('mtf_confluence', 0):.2f}`\n"
                f"Signal Strength: `{adv.get('signal_strength', 0):.1f}%`\n\n"
            )

        # Add risk management
        if 'risk' in analysis:
            risk = analysis['risk']
            message += (
                f"⚠️ *Risk Management:*\n"
                f"Stop Loss: `${risk.get('stop_loss', 0):.4f}`\n"
                f"Target 1: `${risk.get('target1', 0):.4f}`\n"
                f"Target 2: `${risk.get('target2', 0):.4f}`\n"
                f"Target 3: `${risk.get('target3', 0):.4f}`\n"
                f"Risk: `{risk.get('risk_percent', 0):.2f}%`\n"
                f"R:R Ratio: `1:{risk.get('risk_reward_ratio', 0):.1f}`\n\n"
            )

        # Add Bybit futures info
        if getattr(self.bot_instance, "bybit_client", None):
            size_info = self.bot_instance.bybit_client.calculate_position_size(
                symbol, entry_price
            )
            if "quantity" in size_info:
                message += (
                    f"💰 *Bybit Futures (2x):*\n"
                    f"Position Size: `{size_info['quantity']:.4f}` {symbol.replace('USDT', '')}\n"
                    f"Position Value: `${size_info['position_value']:.2f}`\n"
                    f"Required Margin: `${size_info['required_margin']:.2f}`\n"
                )

        # Add reasoning
        if 'reasons' in analysis:
            message += f"\n📝 *Analysis Notes:*\n"
            for reason in analysis['reasons'][:3]:  # Top 3 reasons
                message += f"• {reason}\n"

        self.send_message(message)

    def handle_confidence_adjustment(self, command: str):
        """Handle confidence threshold adjustment"""
        try:
            parts = command.split()
            if len(parts) != 2:
                self.send_message(
                    "❌ Invalid format!\n"
                    "Use: `/confidence VALUE`\n"
                    "Example: `/confidence 60`"
                )
                return

            new_confidence = float(parts[1])
            if 30 <= new_confidence <= 95:
                self.bot_instance.min_confidence_threshold = new_confidence
                self.send_message(
                    f"✅ Minimum confidence updated to {new_confidence}%"
                )
            else:
                self.send_message("❌ Confidence must be between 30 and 95")

        except ValueError:
            self.send_message("❌ Invalid confidence value")

    def send_status_update(self):
        """Send current bot status"""
        stats = self.bot_instance.analyzer.get_advanced_stats()
        message = (
            f"📊 *Bot Status v4.0*\n\n"
            f"Running: `{self.bot_instance.running}`\n"
            f"Min Confidence: `{self.bot_instance.min_confidence_threshold}%`\n"
            f"Symbols Monitored: `{len(self.bot_instance.symbols)}`\n"
            f"Total Signals: `{stats['total']}`\n"
            f"Active: `{stats['active']}`\n"
            f"Win Rate: `{stats['win_rate']:.1f}%`\n"
            f"Daily PnL: `{self.bot_instance.daily_pnl:.2f}%`\n"
            f"Consecutive Losses: `{self.bot_instance.consecutive_losses}`\n"
            f"Bybit Trading: `{'Enabled' if self.bot_instance.bybit_client else 'Disabled'}`"
        )
        self.send_message(message)

    def send_balance_info(self):
        """Send Bybit account balance with improved formatting."""
        if not getattr(self.bot_instance, "bybit_client", None):
            self.send_message("❌ Bybit trading not configured")
            return

        balance_data = self.bot_instance.bybit_client.get_account_balance()

        if "error" in balance_data:
            self.send_message(f"❌ Error: {balance_data['error']}")
            return

        if "status" in balance_data and balance_data["status"] == "no_auth":
            self.send_message("⚠️ Bybit API keys not configured - simulation mode")
            return

        # Format the balance message
        message = "💰 *Bybit Account Balance*\n\n"

        if "summary" in balance_data:
            summary = balance_data["summary"]

            if summary.get("hasBalance"):
                if summary.get("totalUSDTBalance", 0) > 0:
                    message += f"💵 *USDT Balance:* `${summary['totalUSDTBalance']:.2f}`\n"
                if summary.get("totalUSDCBalance", 0) > 0:
                    message += f"💵 *USDC Balance:* `${summary['totalUSDCBalance']:.2f}`\n"
                if summary.get("totalUnrealisedPnl", 0) != 0:
                    pnl_emoji = "📈" if summary['totalUnrealisedPnl'] > 0 else "📉"
                    message += f"{pnl_emoji} *Unrealised PnL:* `${summary['totalUnrealisedPnl']:.2f}`\n"

                message += f"\n📊 *Account Type:* `{balance_data.get('accountType', 'Unknown')}`\n"

                # Add detailed account info if available
                if "accounts" in balance_data and balance_data["accounts"]:
                    account = balance_data["accounts"][0]
                    if "totalAvailableBalance" in account:
                        message += f"✅ *Available:* `${float(account.get('totalAvailableBalance', 0)):.2f}`\n"
                    if "totalInitialMargin" in account:
                        message += f"🔒 *Initial Margin:* `${float(account.get('totalInitialMargin', 0)):.2f}`\n"
                    if "totalMarginBalance" in account:
                        message += f"💼 *Margin Balance:* `${float(account.get('totalMarginBalance', 0)):.2f}`\n"
            else:
                message += "⚠️ No balance found in account\n"
                message += f"Account Type checked: `{balance_data.get('accountType', 'Unknown')}`\n"
        else:
            message += "❌ Unable to parse balance data\n"

        # Add environment indicator
        if getattr(self.bot_instance.bybit_client, "demo_trading", False):
            message += "\n🧪 *Environment:* Demo Trading"
        else:
            message += "\n🌐 *Environment:* Live Trading"

        self.send_message(message)

    def send_symbol_list(self):
        """Send list of monitored symbols"""
        symbols = self.bot_instance.symbols
        message = f"📋 *Monitoring {len(symbols)} Symbols:*\n\n"

        # Group symbols in columns
        for i in range(0, len(symbols), 3):
            row = symbols[i:i+3]
            message += " | ".join([f"`{s}`" for s in row]) + "\n"

        self.send_message(message)

    def send_help_message(self):
        """Send help message with all commands"""
        message = (
            "📚 *Bot v4.0 Commands:*\n\n"
            "🎯 *Analysis:*\n"
            "`/analysis SYMBOL DIRECTION PRICE` - Manual analysis\n"
            "Example: `/analysis BTCUSDT LONG 118500`\n\n"
            "🤖 *Control:*\n"
            "/start - Start the bot\n"
            "/stop - Stop the bot\n"
            "/status - Current status\n"
            "/stats - Detailed statistics\n"
            "/active - Show active signals\n\n"
            "⚙️ *Settings:*\n"
            "`/confidence VALUE` - Set min confidence (30-95)\n"
            "/symbols - List monitored symbols\n\n"
            "💰 *Trading:*\n"
            "/balance - Bybit account balance\n\n"
            "❓ /help - Show this message"
        )
        self.send_message(message)

    def send_message(self, text: str):
        """Send message to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }
            requests.post(url, data=payload, timeout=10)
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    def send_statistics(self):
        """Send detailed statistics"""
        stats = self.bot_instance.analyzer.get_advanced_stats()
        message = (
            f"📈 *Detailed Statistics*\n\n"
            f"Total Signals: `{stats['total']}`\n"
            f"Wins: `{stats['wins']}`\n"
            f"Losses: `{stats['losses']}`\n"
            f"Active: `{stats['active']}`\n"
            f"Win Rate: `{stats['win_rate']:.1f}%`\n"
            f"Avg Confidence: `{stats.get('avg_confidence', 0):.1f}%`\n"
            f"Avg Strength: `{stats.get('avg_strength', 0):.1f}%`\n"
            f"Divergence WR: `{stats.get('divergence_win_rate', 0):.1f}%`\n"
        )

        # Add session stats if available
        if 'session_stats' in stats and stats['session_stats']:
            message += "\n*Session Performance:*\n"
            for session in stats['session_stats']:
                if session[1] > 0:  # Has trades
                    wr = (session[2] / session[1]) * 100 if session[1] > 0 else 0
                    message += f"`{session[0]}`: {wr:.1f}% ({session[2]}/{session[1]})\n"

        self.send_message(message)

    def send_active_positions(self):
        """Send active signals/positions"""
        conn = sqlite3.connect(self.bot_instance.analyzer.db_file)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol, direction, entry_price, current_price, pnl_percent, confidence, stop_loss, target_price
            FROM signals WHERE status = 'ACTIVE'
            ORDER BY timestamp DESC
        """)
        active = cursor.fetchall()
        conn.close()

        if active:
            message = "*📊 Active Signals:*\n\n"
            for sig in active:
                symbol, direction, entry, current, pnl, conf, sl, tp = sig
                emoji = "🟢" if direction == "LONG" else "🔴"
                pnl_emoji = "📈" if pnl > 0 else "📉"
                message += (
                    f"{emoji} *{symbol}* {direction}\n"
                    f"Entry: `${entry:.4f}`\n"
                    f"Current: `${current:.4f}`\n"
                    f"SL: `${sl:.4f}` | TP: `${tp:.4f}`\n"
                    f"PnL: `{pnl:.2f}%` {pnl_emoji}\n"
                    f"Confidence: `{conf:.1f}%`\n\n"
                )
            self.send_message(message)
        else:
            self.send_message("📭 No active signals at the moment.")


class AdvancedSignalAnalyzer:
    """
    Minimal embedded analyzer stub so V4_full.py can run without v3.py.
    Replace with your full analyzer for production.
    """
    def __init__(self):
        import sqlite3
        self._ensure_db()

        import os
        self.db_file = os.getenv("SIGNALS_DB", "signals_v4.sqlite3")
    def calculate_rsi(self, series, period: int = 14):
        import pandas as pd
        s = pd.Series(series).astype(float)
        delta = s.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.ewm(alpha=1/period, adjust=False).mean()
        roll_down = down.ewm(alpha=1/period, adjust=False).mean()
        rs = roll_up / (roll_down.replace(0, 1e-9))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)
    def calculate_macd(self, series, fast=12, slow=26, signal=9):
        import pandas as pd
        s = pd.Series(series).astype(float)
        ema_fast = s.ewm(span=fast, adjust=False).mean()
        ema_slow = s.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal
    def analyze_volume_profile(self, df): return {"volume_ratio": 1.0}
    def identify_market_structure(self, df): return {"structure": "RANGE"}
    def detect_divergences(self, df): return []
    def analyze_order_flow(self, df): return {"pressure": "NEUTRAL"}
    def calculate_dynamic_stop_loss(self, df, direction, entry_price):
        price = float(entry_price); buff = price * 0.015
        if str(direction).upper() == "LONG":
            return {"stop_loss": price - buff, "targets": [price+buff, price+2*buff, price+3*buff]}
        return {"stop_loss": price + buff, "targets": [price-buff, price-2*buff, price-3*buff]}
    def analyze_btc_trend(self, btc_df): return {"trend": "UNKNOWN"}
    def determine_btc_correlation(self, symbol, direction, btc_trend):
        """
        Return a dict with confidence modifier used by the bot.
        """
        t = (btc_trend or {}).get("trend", "UNKNOWN")
        if t == "UPTREND" and str(direction).upper() == "LONG":
            return {"type": "CORRELATED", "description": "BTC trend supports LONG", "confidence_modifier": 1.05}
        if t == "DOWNTREND" and str(direction).upper() == "SHORT":
            return {"type": "CORRELATED", "description": "BTC trend supports SHORT", "confidence_modifier": 1.05}
        if t in ("UPTREND","DOWNTREND"):
            return {"type": "DIVERGENT", "description": "BTC trend diverges", "confidence_modifier": 0.97}
        return {"type": "UNKNOWN", "description": "BTC trend unknown", "confidence_modifier": 1.0}

    def is_optimal_trading_session(self):
        return {"session": "ANY", "is_optimal": True}

    
    def check_rsi_crossover_accuracy(self, df, direction):
        """
        Stubbed RSI validation: returns a dict with keys expected by the bot.
        """
        try:
            # Basic heuristic: if RSI between 45-55, weak; outside -> stronger
            import pandas as pd
            rsi = pd.to_numeric(df['rsi'], errors='coerce').iloc[-1] if 'rsi' in df else 50.0
            if 45 <= rsi <= 55:
                return {"valid": True, "accuracy_score": 0.6, "reason": f"RSI neutral zone {rsi:.1f}"}
            elif rsi > 55 and str(direction).upper() == "LONG":
                return {"valid": True, "accuracy_score": 0.8, "reason": f"RSI supports LONG {rsi:.1f}"}
            elif rsi < 45 and str(direction).upper() == "SHORT":
                return {"valid": True, "accuracy_score": 0.8, "reason": f"RSI supports SHORT {rsi:.1f}"}
            return {"valid": True, "accuracy_score": 0.7, "reason": f"RSI {rsi:.1f}"}
        except Exception:
            return {"valid": True, "accuracy_score": 0.7, "reason": "RSI fallback"}

    
    def get_advanced_stats(self): return {"total": 0, "wins": 0, "losses": 0, "active": 0, "win_rate": 0.0}
    def capture_advanced_signal(self, signal_data, analysis_results):
        return "✅ Signal saved"
    def update_active_signals(self, *args, **kwargs): return None

    

    def calculate_position_size(self, symbol: str, entry_price: float) -> Dict:
        """
        Calculate simple position sizing for USDT margined contracts.
        We treat position_size_usd as the desired *position value* at 1x, then apply leverage.
        quantity = (position_size_usd * leverage) / entry_price
        required_margin ≈ position_value / leverage = position_size_usd
        """
        try:
            price = float(entry_price)
            if price <= 0:
                return {"error": "invalid_price"}
            position_value = float(self.position_size_usd) * float(self.leverage)
            quantity = position_value / price
            required_margin = position_value / float(self.leverage)
            return {
                "symbol": symbol,
                "entry_price": price,
                "quantity": quantity,
                "position_value": position_value,
                "required_margin": required_margin
            }
        except Exception as e:
            return {"error": str(e)}

    def _ensure_db(self):
        import sqlite3, os
        if not hasattr(self, "db_file"):
            self.db_file = os.getenv("SIGNALS_DB", "signals_v4.sqlite3")
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL,
                current_price REAL,
                pnl_percent REAL,
                confidence REAL,
                stop_loss REAL,
                target_price REAL,
                status TEXT DEFAULT 'ACTIVE',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    

class EnhancedMEXCBot:
    """Enhanced bot with all requested features"""
    
    def __init__(self):
        self.analyzer = AdvancedSignalAnalyzer()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'EnhancedBot/4.0'})
        
        # Load all symbols from symbols.txt
        self.symbols = self.load_all_symbols()
        
        self.base_url = "https://api.mexc.com"
        self.binance_url = "https://api.binance.com"
        
        # Updated parameters
        self.min_confidence_threshold = 55  # Reduced from 60 to 50
        self.max_concurrent_positions = 50  # Increased for more symbols
        self.daily_loss_limit = -3.0  # Percent
        self.consecutive_loss_limit = 7
        
        # Tracking
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.active_positions = 0
        
        # Multi-timeframe settings
        self.timeframes = ['1m', '5m', '15m', '1h', '4h']
        self.primary_timeframes = ['5m', '15m']
        
        # Telegram
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.telegram_handler = None
        
        # FIXED: Bybit Futures client with proper demo trading support
        self.bybit_client = None
        if os.getenv("BYBIT_API_KEY"):
            self.bybit_client = EnhancedBybitFuturesClient(
                demo_trading=True  # Use True for demo, False for live trading
            )
        
        # Control
        self.running = True
        
        logger.info(f"Enhanced Bot v4.0 initialized with {len(self.symbols)} symbols")
    
    def load_all_symbols(self):
        """Load all symbols from symbols.txt"""
        if os.path.exists('symbols.txt'):
            try:
                with open('symbols.txt', 'r') as f:
                    symbols = [line.strip().upper() for line in f.readlines() if line.strip()]
                    if symbols:
                        logger.info(f"Loaded {len(symbols)} symbols from symbols.txt")
                        return symbols
            except Exception as e:
                logger.error(f"Error loading symbols: {e}")
        
        # Fallback to default symbols
        default_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
        logger.warning("Using default symbols")
        return default_symbols
    
    def fetch_price(self, symbol):
        """Fetch current price with fallback"""
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            response = self.session.get(url, params={"symbol": symbol}, timeout=5)
            if response.status_code == 200:
                return float(response.json().get('price', 0))
        except:
            pass
        
        try:
            url = f"{self.binance_url}/api/v3/ticker/price"
            response = self.session.get(url, params={"symbol": symbol}, timeout=5)
            if response.status_code == 200:
                return float(response.json().get('price', 0))
        except:
            pass
        
        return None
    
    def fetch_klines(self, symbol, interval, limit=100):
        """Fetch candlestick data with fallback"""
        try:
            # Try MEXC first
            url = f"{self.base_url}/api/v3/klines"
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            response = self.session.get(url, params=params, timeout=10)
            source = "MEXC"
            
            # If MEXC fails, try Binance
            if response.status_code != 200:
                url = f"{self.binance_url}/api/v3/klines"
                response = self.session.get(url, params=params, timeout=10)
                source = "Binance"
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    # Check the number of columns in the response
                    num_columns = len(data[0])
                    
                    if num_columns == 8:
                        # MEXC format (8 columns)
                        df = pd.DataFrame(data, columns=[
                            "timestamp", "open", "high", "low", "close", "volume", 
                            "close_time", "quote_volume"
                        ])
                    elif num_columns == 12:
                        # Binance format (12 columns)
                        df = pd.DataFrame(data, columns=[
                            "timestamp", "open", "high", "low", "close", "volume", 
                            "close_time", "quote_volume", "trades", "taker_buy_base",
                            "taker_buy_quote", "ignore"
                        ])
                    else:
                        # Fallback for other formats
                        df = pd.DataFrame(data)
                        essential_cols = ["timestamp", "open", "high", "low", "close", "volume", 
                                        "close_time", "quote_volume"]
                        df.columns = essential_cols[:len(df.columns)] + list(df.columns[len(essential_cols):])
                    
                    # Convert numeric columns
                    numeric_cols = ["open", "high", "low", "close", "volume"]
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Convert timestamp to datetime
                    df["timestamp"] = pd.to_numeric(df["timestamp"], errors='coerce')
                    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                    
                    # Remove any NaN values in essential columns
                    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
                    
                    return df.sort_values("timestamp").reset_index(drop=True)
                    
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
        
        return None
    
    def perform_manual_analysis(self, symbol: str, direction: str, entry_price: float) -> Dict:
        """Perform comprehensive analysis for manual request"""
        try:
            logger.info(f"Performing manual analysis for {symbol} {direction} @ {entry_price}")
            
            # Fetch data for multiple timeframes
            analysis_results = {
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'confidence': 0,
                'technical': {},
                'advanced': {},
                'risk': {},
                'reasons': []
            }
            
            # Primary timeframe analysis (15m)
            df = self.fetch_klines(symbol, '15m', limit=100)
            if df is None or len(df) < 50:
                return None
            
            # Calculate indicators
            df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
            df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
            df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
            df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
            df["rsi"] = self.analyzer.calculate_rsi(df["close"])
            df["macd"], df["macd_signal"] = self.analyzer.calculate_macd(df["close"])
            
            latest = df.iloc[-1]
            
            # Technical analysis
            analysis_results['technical'] = {
                'rsi': float(latest["rsi"]),
                'macd': float(latest["macd"]),
                'macd_signal': 'BULLISH' if latest["macd"] > latest["macd_signal"] else 'BEARISH',
                'ema_aligned': bool(latest["ema5"] > latest["ema10"] > latest["ema20"]) if direction == 'LONG' else bool(latest["ema5"] < latest["ema10"] < latest["ema20"]),
                'volume_ratio': float(latest["volume"] / df["volume"].rolling(20).mean().iloc[-1]) if df["volume"].rolling(20).mean().iloc[-1] > 0 else 1
            }
            
            # Advanced analysis
            volume_profile = self.analyzer.analyze_volume_profile(df)
            market_structure = self.analyzer.identify_market_structure(df)
            divergences = self.analyzer.detect_divergences(df)
            order_flow = self.analyzer.analyze_order_flow(df)
            dynamic_stops = self.analyzer.calculate_dynamic_stop_loss(df, direction, entry_price)
            confluence = self.check_mtf_confluence(symbol)
            
            # BTC trend analysis
            btc_df = self.fetch_klines("BTCUSDT", "15m", limit=100)
            btc_trend = self.analyzer.analyze_btc_trend(btc_df) if btc_df is not None else {"trend": "UNKNOWN"}
            btc_correlation = self.analyzer.determine_btc_correlation(symbol, direction, btc_trend)
            
            # RSI accuracy check
            rsi_analysis = self.analyzer.check_rsi_crossover_accuracy(df, direction)
            
            analysis_results['advanced'] = {
                'market_structure': market_structure.get('structure', 'UNDEFINED'),
                'btc_correlation': btc_correlation.get('type', 'UNKNOWN'),
                'mtf_confluence': confluence.get('strength', 0),
                'signal_strength': 0,  # Will calculate below
                'vwap_aligned': volume_profile.get('above_vwap', False) if direction == 'LONG' else not volume_profile.get('above_vwap', False),
                'divergence_detected': divergences.get('has_divergence', False),
                'order_flow_aligned': (order_flow.get('buy_pressure', False) and direction == 'LONG') or (not order_flow.get('buy_pressure', False) and direction == 'SHORT')
            }
            
            # Risk management
            analysis_results['risk'] = {
                'stop_loss': dynamic_stops['stop_loss'],
                'target1': dynamic_stops['target1'],
                'target2': dynamic_stops['target2'],
                'target3': dynamic_stops['target3'],
                'risk_percent': dynamic_stops['risk_percent'],
                'risk_reward_ratio': dynamic_stops['risk_reward_ratio']
            }
            
            # Calculate confidence score
            base_confidence = 50
            
            # Technical factors
            if analysis_results['technical']['ema_aligned']:
                base_confidence += 10
                analysis_results['reasons'].append("EMA alignment confirmed")
            
            if analysis_results['technical']['volume_ratio'] > 1.5:
                base_confidence += 8
                analysis_results['reasons'].append("High volume confirmation")
            
            # RSI factors
            if direction == 'LONG' and 40 < latest["rsi"] < 60:
                base_confidence += 5
            elif direction == 'SHORT' and 40 < latest["rsi"] < 60:
                base_confidence += 5
            
            if rsi_analysis.get('valid', False):
                base_confidence += rsi_analysis.get('accuracy_score', 0) * 15
                analysis_results['reasons'].append(f"RSI validation: {rsi_analysis.get('reason', '')}")
            
            # Market structure
            if (market_structure['structure'] == 'UPTREND' and direction == 'LONG') or \
               (market_structure['structure'] == 'DOWNTREND' and direction == 'SHORT'):
                base_confidence += 12
                analysis_results['reasons'].append(f"Market structure aligned: {market_structure['structure']}")
            elif market_structure['structure'] in ['TRANSITION_UP', 'TRANSITION_DOWN']:
                base_confidence += 5
            
            # BTC correlation
            base_confidence *= btc_correlation.get('confidence_modifier', 1.0)
            if btc_correlation.get('type') == 'CORRELATED':
                analysis_results['reasons'].append(f"BTC correlation: {btc_correlation.get('description', '')}")
            
            # MTF confluence
            if confluence['aligned']:
                base_confidence += 10
                analysis_results['reasons'].append(f"Multi-timeframe confluence: {confluence['strength']:.2f}")
            
            # Advanced factors
            if analysis_results['advanced']['vwap_aligned']:
                base_confidence += 5
                analysis_results['reasons'].append("VWAP alignment confirmed")
            
            if analysis_results['advanced']['divergence_detected']:
                base_confidence += 8
                analysis_results['reasons'].append("Divergence detected")
            
            if analysis_results['advanced']['order_flow_aligned']:
                base_confidence += 5
                analysis_results['reasons'].append("Order flow aligned")
            
            # Price action relative to entry
            current_price = float(latest['close'])
            price_diff_pct = abs((entry_price - current_price) / current_price) * 100
            
            if price_diff_pct > 2:
                base_confidence -= 10
                analysis_results['reasons'].append(f"Entry price {price_diff_pct:.1f}% away from current")
            
            # Cap confidence
            base_confidence = max(min(base_confidence, 95), 20)
            
            analysis_results['confidence'] = base_confidence
            analysis_results['advanced']['signal_strength'] = base_confidence
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in manual analysis: {e}")
            return None
    
    def check_mtf_confluence(self, symbol: str) -> Dict:
        """Check signal alignment across multiple timeframes"""
        alignments = []
        
        for tf in self.timeframes:
            try:
                df = self.fetch_klines(symbol, tf, limit=50)
                if df is None or len(df) < 20:
                    continue
                
                # Calculate trend for each timeframe
                df['ema20'] = df['close'].ewm(span=20).mean()
                df['rsi'] = self.analyzer.calculate_rsi(df['close'])
                
                latest = df.iloc[-1]
                
                # Determine bias
                if latest['close'] > latest['ema20'] and latest['rsi'] > 50:
                    bias = 'BULLISH'
                elif latest['close'] < latest['ema20'] and latest['rsi'] < 50:
                    bias = 'BEARISH'
                else:
                    bias = 'NEUTRAL'
                
                weight = {'1m': 0.1, '5m': 0.2, '15m': 0.3, '1h': 0.25, '4h': 0.15}.get(tf, 0.1)
                alignments.append({
                    'timeframe': tf,
                    'bias': bias,
                    'weight': weight
                })
            except Exception as e:
                logger.error(f"Error checking {tf} timeframe for {symbol}: {e}")
                continue
        
        # Calculate confluence score
        bull_score = sum(a['weight'] for a in alignments if a['bias'] == 'BULLISH')
        bear_score = sum(a['weight'] for a in alignments if a['bias'] == 'BEARISH')
        
        return {
            'bull_confluence': bull_score,
            'bear_confluence': bear_score,
            'aligned': abs(bull_score - bear_score) > 0.5,
            'direction': 'BULLISH' if bull_score > bear_score else 'BEARISH',
            'strength': abs(bull_score - bear_score)
        }
    
    def check_advanced_signals(self, df, symbol):
        """Check for trading signals with DataFrame validation"""
        signals = []
        if df is None or len(df) < 50:
            return signals
        
        try:
            # Calculate all indicators
            df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
            df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
            df["ema15"] = df["close"].ewm(span=15, adjust=False).mean()
            df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
            df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
            df["rsi"] = self.analyzer.calculate_rsi(df["close"])
            df["macd"], df["macd_signal"] = self.analyzer.calculate_macd(df["close"])
            df["bb_mid"] = df["close"].rolling(window=20).mean()
            df["bb_std"] = df["close"].rolling(window=20).std()
            df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
            df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
            df["vol_avg"] = df["volume"].rolling(window=20).mean()
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Volume analysis
            volume_ratio = latest["volume"] / latest["vol_avg"] if latest["vol_avg"] > 0 else 1
            high_volume = volume_ratio > 1.5
            
            # EMA alignment check
            ema_bullish = latest["ema5"] > latest["ema10"] > latest["ema15"] > latest["ema20"]
            ema_bearish = latest["ema5"] < latest["ema10"] < latest["ema15"] < latest["ema20"]
            
            # Enhanced EMA crossover signals
            bullish_cross = (latest["ema5"] > latest["ema10"]) and (prev["ema5"] <= prev["ema10"])
            bearish_cross = (latest["ema5"] < latest["ema10"]) and (prev["ema5"] >= prev["ema10"])
            
            # MACD signals with momentum
            macd_bull = ((latest["macd"] > latest["macd_signal"]) and 
                        (prev["macd"] <= prev["macd_signal"]) and 
                        latest["rsi"] > 45 and latest["macd"] > 0)
            
            macd_bear = ((latest["macd"] < latest["macd_signal"]) and 
                        (prev["macd"] >= prev["macd_signal"]) and 
                        latest["rsi"] < 55 and latest["macd"] < 0)
            
            # RSI reversal signals
            rsi_oversold_recovery = (prev["rsi"] <= 30 and latest["rsi"] > 35 and 
                                   latest["close"] > prev["close"])
            rsi_overbought_decline = (prev["rsi"] >= 70 and latest["rsi"] < 65 and 
                                    latest["close"] < prev["close"])
            
            # Bollinger Band signals
            bb_squeeze = latest["bb_std"] < df["bb_std"].rolling(20).mean().iloc[-1] * 0.75
            bb_expansion = latest["bb_std"] > df["bb_std"].rolling(20).mean().iloc[-1] * 1.25
            price_at_lower_bb = latest["close"] <= latest["bb_lower"] * 1.01
            price_at_upper_bb = latest["close"] >= latest["bb_upper"] * 0.99
            
            # Volume breakout signals
            price_breakout_high = latest["close"] > df["close"].iloc[-21:-1].max()
            price_breakout_low = latest["close"] < df["close"].iloc[-21:-1].min()
            
            # Collect signals with reasons
            signal_triggers = []
            
            # LONG signals
            if bullish_cross and latest["rsi"] > 40 and latest["rsi"] < 70:
                signal_triggers.append({
                    "signal": "LONG",
                    "reason": "EMA5/10 bullish crossover with RSI confirmation",
                    "strength": 0.75,
                    "primary": True
                })
            
            if macd_bull and high_volume:
                signal_triggers.append({
                    "signal": "LONG",
                    "reason": "MACD bullish crossover with volume surge",
                    "strength": 0.8,
                    "primary": True
                })
            
            if rsi_oversold_recovery and (ema_bullish or latest["ema5"] > latest["ema20"]):
                signal_triggers.append({
                    "signal": "LONG",
                    "reason": "RSI oversold recovery with EMA support",
                    "strength": 0.85,
                    "primary": True
                })
            
            if price_at_lower_bb and latest["rsi"] < 35 and bb_squeeze:
                signal_triggers.append({
                    "signal": "LONG",
                    "reason": "Bollinger Band squeeze at lower band with oversold RSI",
                    "strength": 0.7,
                    "primary": False
                })
            
            if price_breakout_high and high_volume and latest["rsi"] < 75:
                signal_triggers.append({
                    "signal": "LONG",
                    "reason": "Price breakout with volume confirmation",
                    "strength": 0.85,
                    "primary": True
                })
            
            # SHORT signals
            if bearish_cross and latest["rsi"] < 60 and latest["rsi"] > 30:
                signal_triggers.append({
                    "signal": "SHORT",
                    "reason": "EMA5/10 bearish crossover with RSI confirmation",
                    "strength": 0.75,
                    "primary": True
                })
            
            if macd_bear and high_volume:
                signal_triggers.append({
                    "signal": "SHORT",
                    "reason": "MACD bearish crossover with volume surge",
                    "strength": 0.8,
                    "primary": True
                })
            
            if rsi_overbought_decline and (ema_bearish or latest["ema5"] < latest["ema20"]):
                signal_triggers.append({
                    "signal": "SHORT",
                    "reason": "RSI overbought decline with EMA resistance",
                    "strength": 0.85,
                    "primary": True
                })
            
            if price_at_upper_bb and latest["rsi"] > 65 and bb_squeeze:
                signal_triggers.append({
                    "signal": "SHORT",
                    "reason": "Bollinger Band squeeze at upper band with overbought RSI",
                    "strength": 0.7,
                    "primary": False
                })
            
            if price_breakout_low and high_volume and latest["rsi"] > 25:
                signal_triggers.append({
                    "signal": "SHORT",
                    "reason": "Price breakdown with volume confirmation",
                    "strength": 0.85,
                    "primary": True
                })
            
            # Filter and enhance signals
            for trigger in signal_triggers:
                # Only use primary signals or very strong secondary signals
                if not trigger["primary"] and trigger["strength"] < 0.8:
                    continue
                
                trigger.update({
                    "symbol": symbol,
                    "price": float(latest["close"]),
                    "ema5": float(latest["ema5"]),
                    "ema10": float(latest["ema10"]),
                    "ema20": float(latest["ema20"]),
                    "ema50": float(latest["ema50"]),
                    "rsi": float(latest["rsi"]),
                    "macd": float(latest["macd"]),
                    "macd_signal": float(latest["macd_signal"]),
                    "bb_upper": float(latest["bb_upper"]),
                    "bb_lower": float(latest["bb_lower"]),
                    "volume": float(latest["volume"]),
                    "volume_ratio": volume_ratio,
                    "ema_aligned": ema_bullish if trigger["signal"] == "LONG" else ema_bearish
                })
                
                signals.append(trigger)
                
        except Exception as e:
            logger.error(f"Signal calculation error for {symbol}: {e}")
        
        return signals
    
    def should_take_trade(self, confidence, session_info):
        """Determine if trade should be taken based on risk management"""
        # Check daily loss limit
        if self.daily_pnl <= self.daily_loss_limit:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f}%")
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= self.consecutive_loss_limit:
            logger.warning(f"Consecutive loss limit reached: {self.consecutive_losses}")
            return False
        
        # Check max concurrent positions
        if self.active_positions >= self.max_concurrent_positions:
            logger.info(f"Max concurrent positions reached: {self.active_positions}")
            return False
        
        # Check minimum confidence (now 50%)
        if confidence < self.min_confidence_threshold:
            logger.info(f"Confidence too low: {confidence:.1f}% < {self.min_confidence_threshold}%")
            return False
        
        return True
    
    def process_signal_with_bybit(self, signal_data: Dict, analysis_results: Dict):
        """Process signal and optionally place Bybit futures order"""
        try:
            # Check if Bybit trading is enabled
            if not self.bybit_client:
                logger.info("Bybit trading not configured - signal only mode")
                return
            
            # Check if we should take the trade
            confidence = signal_data.get('confidence', 0)
            if confidence < 50:  # Higher threshold for real trading
                logger.info(f"Confidence {confidence:.1f}% too low for Bybit trading")
                return
            
            symbol = signal_data['symbol']
            direction = signal_data['direction']
            entry_price = signal_data['entry_price']
            
            # Get stop loss and take profit from analysis
            stop_loss = analysis_results['dynamic_stops']['stop_loss']
            take_profit = analysis_results['dynamic_stops']['target2']  # Use middle target
            
            # Place the order
            order_result = self.bybit_client.place_futures_order(
                symbol=symbol,
                side=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if order_result.get('status') == 'success':
                message = (
                    f"✅ *BYBIT FUTURES ORDER PLACED*\n\n"
                    f"Symbol: `{symbol}`\n"
                    f"Direction: *{direction}*\n"
                    f"Entry: `${entry_price:.4f}`\n"
                    f"Stop Loss: `${stop_loss:.4f}`\n"
                    f"Take Profit: `${take_profit:.4f}`\n"
                    f"Leverage: `{self.bybit_client.leverage}x`\n"
                    f"Position Size: `$100`\n"
                    f"Order ID: `{order_result.get('order_id', 'N/A')}`"
                )
                if self.telegram_handler:
                    self.telegram_handler.send_message(message)
                logger.info(f"Bybit order placed: {order_result}")
                
            elif order_result.get('status') == 'simulation':
                logger.info(f"Bybit simulation: {order_result}")
            else:
                logger.error(f"Bybit order failed: {order_result}")
                
        except Exception as e:
            logger.error(f"Error processing Bybit order: {e}")
    
    def run_analysis_cycle(self):
        """Main analysis cycle for all symbols"""
        logger.info(f"Running analysis cycle for {len(self.symbols)} symbols...")
        
        # First, analyze BTC trend
        btc_df = self.fetch_klines("BTCUSDT", "15m", limit=100)
        btc_trend = self.analyzer.analyze_btc_trend(btc_df) if btc_df is not None else {"trend": "UNKNOWN"}
        
        signals_found = 0
        symbols_processed = 0
        
        # Process symbols in batches to avoid overwhelming the API
        batch_size = 5
        for i in range(0, len(self.symbols), batch_size):
            batch = self.symbols[i:i+batch_size]
            
            for symbol in batch:
                if symbol == "BTCUSDT":
                    continue
                
                try:
                    symbols_processed += 1
                    
                    for tf in self.primary_timeframes:
                        df = self.fetch_klines(symbol, tf, limit=100)
                        if df is None:
                            continue
                        
                        # Check for signals
                        signals = self.check_advanced_signals(df, symbol)
                        
                        for sig in signals:
                            # Check for existing active signals
                            conn = sqlite3.connect(self.analyzer.db_file)
                            cursor = conn.cursor()
                            cursor.execute('''
                                SELECT COUNT(*) FROM signals 
                                WHERE symbol=? AND direction=? AND status='ACTIVE'
                            ''', (symbol, sig['signal']))
                            already_active = cursor.fetchone()[0]
                            conn.close()
                            
                            if already_active:
                                continue
                            
                            # Perform comprehensive analysis (simplified for performance)
                            analysis_results = self.perform_basic_analysis(symbol, df, sig, btc_trend)
                            if not analysis_results:
                                continue
                            
                            # Calculate confidence
                            confidence = analysis_results.get('confidence', 0)
                            
                            # Check if we should take the trade
                            if not self.should_take_trade(confidence, analysis_results.get('session_info', (True, 'UNKNOWN'))):
                                continue
                            
                            # Prepare signal data
                            signal_data = {
                                'symbol': symbol,
                                'direction': sig['signal'],
                                'entry_price': sig['price'],
                                'confidence': confidence,
                                'reason': sig.get('reason', '')
                            }
                            
                            # Capture the signal
                            result = self.analyzer.capture_advanced_signal(signal_data, analysis_results)
                            
                            if result.startswith("✅"):
                                self.active_positions += 1
                                signals_found += 1
                                
                                # Send alert
                                if self.telegram_handler:
                                    self.send_signal_alert(symbol, sig, analysis_results, confidence)
                                
                                # Process with Bybit if enabled
                                self.process_signal_with_bybit(signal_data, analysis_results)
                                
                                logger.info(f"Signal: {symbol} {sig['signal']} @ {sig['price']:.4f} ({confidence:.1f}%)")
                    
                    # Small delay between symbols
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
            
            # Delay between batches
            time.sleep(2)
        
        # Update active signals
        self.analyzer.update_active_signals(self.fetch_price, 
                                           self.telegram_handler.send_message if self.telegram_handler else None)
        
        # Update statistics
        self.update_tracking_variables()
        
        logger.info(
            f"Cycle complete - Processed: {symbols_processed}, "
            f"Signals: {signals_found}, Active: {self.active_positions}"
        )
    
    def perform_basic_analysis(self, symbol, df, signal, btc_trend):
        """Simplified analysis for performance with many symbols"""
        try:
            # Essential analysis only
            volume_profile = self.analyzer.analyze_volume_profile(df)
            market_structure = self.analyzer.identify_market_structure(df)
            dynamic_stops = self.analyzer.calculate_dynamic_stop_loss(df, signal['signal'], signal['price'])
            
            # RSI check
            rsi_analysis = self.analyzer.check_rsi_crossover_accuracy(df, signal['signal'])
            if not rsi_analysis['valid']:
                return None
            
            # BTC correlation
            btc_correlation = self.analyzer.determine_btc_correlation(symbol, signal['signal'], btc_trend)
            
            # Calculate confidence
            base_confidence = signal.get('strength', 0.7) * 100
            base_confidence *= rsi_analysis.get('accuracy_score', 1.0)
            base_confidence *= btc_correlation.get('confidence_modifier', 1.0)
            
            if market_structure['structure'] == 'UPTREND' and signal['signal'] == 'LONG':
                base_confidence *= 1.1
            elif market_structure['structure'] == 'DOWNTREND' and signal['signal'] == 'SHORT':
                base_confidence *= 1.1
            
            base_confidence = max(min(base_confidence, 95), 30)
            
            return {
                'confidence': base_confidence,
                'volume_profile': volume_profile,
                'market_structure': market_structure,
                'dynamic_stops': dynamic_stops,
                'rsi_analysis': rsi_analysis,
                'btc_correlation': btc_correlation,
                'session_info': self.analyzer.is_optimal_trading_session()
            }
            
        except Exception as e:
            logger.error(f"Error in basic analysis: {e}")
            return None
    
    def send_signal_alert(self, symbol, signal, analysis_results, confidence):
        """Send signal alert to Telegram"""
        direction_emoji = "🟢" if signal['signal'] == 'LONG' else "🔴"
        
        message = (
            f"{direction_emoji} *{signal['signal']} SIGNAL: {symbol}*\n\n"
            f"Entry: `${signal['price']:.4f}`\n"
            f"Stop Loss: `${analysis_results['dynamic_stops']['stop_loss']:.4f}`\n"
            f"Target: `${analysis_results['dynamic_stops']['target2']:.4f}`\n"
            f"Confidence: `{confidence:.1f}%`\n"
            f"Reason: {signal['reason']}\n"
        )
        
        if self.telegram_handler:
            self.telegram_handler.send_message(message)
    
    def update_tracking_variables(self):
        """Update tracking variables from database"""
        stats = self.analyzer.get_advanced_stats()
        
        conn = sqlite3.connect(self.analyzer.db_file)
        cursor = conn.cursor()
        
        # Daily PnL
        cursor.execute('''
            SELECT SUM(pnl_percent) FROM signals 
            WHERE DATE(timestamp) = DATE('now') AND status IN ('WIN', 'LOSS')
        ''')
        daily_result = cursor.fetchone()[0]
        self.daily_pnl = daily_result if daily_result else 0
        
        # Consecutive losses
        cursor.execute('''
            SELECT status FROM signals 
            WHERE status IN ('WIN', 'LOSS')
            ORDER BY timestamp DESC LIMIT 5
        ''')
        recent_trades = cursor.fetchall()
        self.consecutive_losses = 0
        for trade in recent_trades:
            if trade[0] == 'LOSS':
                self.consecutive_losses += 1
            else:
                break
        
        self.active_positions = stats['active']
        conn.close()
    
    def run(self):
        """Main execution loop"""
        # Setup Telegram handler
        if self.telegram_token and self.chat_id:
            self.telegram_handler = EnhancedTelegramHandler(
                self.telegram_token, self.chat_id, self
            )
            self.telegram_handler.start_listener()
        
        # Startup message
        startup_message = (
            f"🚀 *Enhanced Trading Bot v4.0 Started* 🚀\n\n"
            f"📊 Monitoring: `{len(self.symbols)}` symbols\n"
            f"🎯 Min Confidence: `{self.min_confidence_threshold}%`\n"
            f"💰 Bybit Trading: `{'Enabled' if self.bybit_client else 'Disabled'}`\n"
            f"⚡ Features:\n"
            f"• Manual analysis command\n"
            f"• Full symbols.txt monitoring\n"
            f"• Bybit Futures support (2x, $100/trade)\n"
            f"• Adjustable confidence threshold\n\n"
            f"Use /help for commands"
        )
        
        if self.telegram_handler:
            self.telegram_handler.send_message(startup_message)
        
        cycle_count = 0
        
        while self.running:
            try:
                cycle_count += 1
                self.run_analysis_cycle()
                
                # Status update every 20 cycles
                if cycle_count % 20 == 0:
                    stats = self.analyzer.get_advanced_stats()
                    status_msg = (
                        f"📊 *Status Update*\n"
                        f"Cycles: `{cycle_count}`\n"
                        f"Active: `{stats['active']}`\n"
                        f"Win Rate: `{stats['win_rate']:.1f}%`\n"
                        f"Daily PnL: `{self.daily_pnl:.2f}%`"
                    )
                    if self.telegram_handler:
                        self.telegram_handler.send_message(status_msg)
                
                # Dynamic sleep based on number of symbols
                sleep_time = max(30, len(self.symbols) * 2)  # At least 30 seconds
                time.sleep(min(sleep_time, 120))  # Max 2 minutes
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)
        
        # Shutdown message
        if self.telegram_handler:
            stats = self.analyzer.get_advanced_stats()
            self.telegram_handler.send_message(
                f"🛑 *Bot Stopped*\n\n"
                f"Total Signals: `{stats['total']}`\n"
                f"Win Rate: `{stats['win_rate']:.1f}%`"
            )


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         ENHANCED TRADING BOT v4.0 - FIXED               ║
    ║                                                          ║
    ║  🎯 New Features:                                        ║
    ║  • Minimum confidence: 55% (adjustable 30-95%)          ║
    ║  • Full symbols.txt monitoring                           ║
    ║  • Manual analysis: /analysis SYMBOL DIRECTION PRICE     ║
    ║  • WORKING Bybit Futures support (2x, $100/trade)       ║
    ║  • FIXED: Demo trading URL and authentication            ║
    ║  • FIXED: Balance parsing with empty string handling     ║
    ║                                                          ║
    ║  💡 Commands:                                            ║
    ║  /analysis BTCUSDT LONG 118500 - Analyze specific trade  ║
    ║  /confidence 60 - Adjust minimum confidence              ║
    ║  /symbols - List all monitored symbols                   ║
    ║  /balance - Check Bybit balance (NOW WORKING!)           ║
    ║  /help - Show all commands                               ║
    ║                                                          ║
    ║  🎉 Your Demo Account: $2.1M+ available for trading!    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        bot = EnhancedMEXCBot()
        bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")


# --- Attach legacy compatible alias at import time ---
def _attach_placefuturesorder_alias():
    def _shim(self, symbol: str, side: str, entry_price: float = None,
              stop_loss: float = None, take_profit: float = None,
              ordertype: str = None, positionsizeusd: float = None,
              leverage: int = None, **kwargs):
        try:
            sym = str(symbol).upper()
            if not sym.endswith("USDT"):
                sym = sym + "USDT"
            canon = { (k.lower().replace("_","") if isinstance(k, str) else k): v for k,v in kwargs.items() }
            if positionsizeusd is None:
                positionsizeusd = canon.get("positionsizeusd", canon.get("orderqtyusd", None))
            if leverage is None:
                leverage = canon.get("lev", canon.get("leveragex", None))
            if take_profit is None:
                take_profit = canon.get("takeprofit", None)
            if stop_loss is None:
                stop_loss = canon.get("stoploss", None)
            if ordertype is None:
                ordertype = canon.get("ordertype", None)
            orderType = ordertype if ordertype else ("Limit" if entry_price else "Market")
            return self.place_futures_order(
                symbol=sym, side=side, entry_price=entry_price,
                stop_loss=stop_loss, take_profit=take_profit,
                orderType=orderType, position_size_usd=positionsizeusd, leverage=leverage
            )
        except Exception as e:
            return {"status": "error", "error": str(e)}
    try:
        BybitFuturesClient.placefuturesorder = _shim
    except Exception:
        pass

_attach_placefuturesorder_alias()
