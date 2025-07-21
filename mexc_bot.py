#!/usr/bin/env python3
"""
Sig_288bot - MEXC EMA5/EMA10 Crossover + RSI Strategy with 24/7 Operation & Futures API
Features:
- EMA5/EMA10 crossover as main signal
- EMA15 as base price reference
- RSI filter: >55 for Long, <45 for Short
- 5m and 15m timeframes
- 24/7 continuous operation
- MEXC Futures API integration
- Dynamic symbol management via Telegram
"""

import requests
import pandas as pd
import numpy as np
import os
import logging
import time
import hmac
import hashlib
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import threading
from urllib.parse import urlencode

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mexc_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MEXCFuturesAPI:
    """MEXC Futures API Client"""
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://contract.mexc.com" if not testnet else "https://contract-test.mexc.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Sig_288bot/2.0',
            'Content-Type': 'application/json'
        })

    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _make_request(self, method: str, endpoint: str, params: dict = None, signed: bool = False) -> dict:
        """Make API request with proper authentication"""
        url = f"{self.base_url}{endpoint}"
        if params is None:
            params = {}
        if signed:
            timestamp = int(time.time() * 1000)
            params['timestamp'] = timestamp
            params['recvWindow'] = 5000
            query_string = urlencode(sorted(params.items()))
            signature = self._generate_signature(query_string)
            params['signature'] = signature
            headers = {'X-MEXC-APIKEY': self.api_key}
            self.session.headers.update(headers)
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, timeout=10)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=params, timeout=10)
            else:
                response = self.session.request(method, url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return {"error": str(e)}

    def get_account_info(self) -> dict:
        """Get futures account information"""
        return self._make_request('GET', '/api/v1/private/account/assets', signed=True)

    def get_position_info(self, symbol: str = None) -> dict:
        """Get position information"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_request('GET', '/api/v1/private/position/list/history_positions', params, signed=True)

    def place_order(self, symbol: str, side: str, order_type: str, vol: float, price: float = None, **kwargs) -> dict:
        """
        Place futures order
        side: 1=long, 2=short, 3=close_long, 4=close_short
        order_type: 1=limit, 2=post_only, 3=reduce_only, 4=market, 5=stop_limit, 6=stop_market
        """
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'vol': vol
        }
        if price:
            params['price'] = price
        params.update(kwargs)
        return self._make_request('POST', '/api/v1/private/order/submit', params, signed=True)

    def get_ticker(self, symbol: str) -> dict:
        """Get 24hr ticker statistics"""
        params = {'symbol': symbol}
        return self._make_request('GET', '/api/v1/contract/ticker', params)

class MEXCBot:
    def __init__(self):
        self.symbols_file = "symbols.txt"
        self.symbols = self.load_symbols()
        self.base_url = "https://api.mexc.com"
        # Strategy parameters
        self.ema5_period = 5
        self.ema10_period = 10
        self.ema15_period = 15
        self.rsi_period = 14
        self.rsi_long_threshold = 55
        self.rsi_short_threshold = 45
        # 24/7 operation settings
        self.scan_interval = 60  # seconds between scans
        self.running = True
        # Telegram credentials
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        # MEXC Futures API credentials
        self.mexc_api_key = os.getenv("MEXC_API_KEY")
        self.mexc_api_secret = os.getenv("MEXC_API_SECRET")
        self.mexc_testnet = os.getenv("MEXC_TESTNET", "true").lower() == "true"
        # Initialize APIs
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Sig_288bot/2.0'})
        if self.mexc_api_key and self.mexc_api_secret:
            self.futures_api = MEXCFuturesAPI(self.mexc_api_key, self.mexc_api_secret, self.mexc_testnet)
            logger.info(f"MEXC Futures API initialized (testnet: {self.mexc_testnet})")
        else:
            self.futures_api = None
            logger.warning("MEXC API credentials not found - trading disabled")
        # Track last processed update to avoid duplicates
        self.last_update_id = self.load_last_update_id()

    def load_symbols(self) -> List[str]:
        """Load symbols from symbols.txt file"""
        default_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "THEUSDT", "XRPUSDT", "SUIUSDT","CHESSUSDT","OGUSDT","MASKUSDT","EDUUSDT","SHIBUSDT"]
        try:
            if os.path.exists(self.symbols_file):
                with open(self.symbols_file, 'r') as f:
                    symbols = [line.strip().upper() for line in f.readlines() if line.strip()]
                    if symbols:
                        logger.info(f"Loaded {len(symbols)} symbols from {self.symbols_file}")
                        return symbols
                    else:
                        logger.info("Empty symbols file, using defaults")
                        self.save_symbols(default_symbols)
                        return default_symbols
            else:
                logger.info("Symbols file not found, creating with defaults")
                self.save_symbols(default_symbols)
                return default_symbols
        except Exception as e:
            logger.error(f"Error loading symbols: {e}, using defaults")
            return default_symbols

    def save_symbols(self, symbols: List[str]) -> bool:
        """Save symbols to symbols.txt file"""
        try:
            with open(self.symbols_file, 'w') as f:
                for symbol in symbols:
                    f.write(f"{symbol.upper()}\n")
            logger.info(f"Saved {len(symbols)} symbols to {self.symbols_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving symbols: {e}")
            return False

    def load_last_update_id(self) -> int:
        """Load last processed update ID to avoid duplicate processing"""
        try:
            if os.path.exists("last_update.txt"):
                with open("last_update.txt", 'r') as f:
                    return int(f.read().strip())
        except:
            pass
        return 0

    def save_last_update_id(self, update_id: int):
        """Save last processed update ID"""
        try:
            with open("last_update.txt", 'w') as f:
                f.write(str(update_id))
        except Exception as e:
            logger.error(f"Error saving last update ID: {e}")

    def fetch_klines(self, symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch kline data with increased limit for better indicator calculations"""
        url = f"{self.base_url}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list) or len(data) == 0:
                logger.warning(f"No kline data for {symbol} ({interval})")
                return None
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume"
            ])
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df.sort_values("timestamp").reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol} ({interval}): {e}")
            return None

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return pd.Series([50] * len(prices))  # Return neutral RSI on error

    def get_high_low_levels(self, df: pd.DataFrame, candles: int = 5) -> Dict[str, float]:
        """Get highest and lowest prices from last N candles"""
        if df is None or len(df) < candles:
            return {"highest": 0.0, "lowest": 0.0}
        try:
            last_candles = df.tail(candles)
            highest = last_candles["high"].max()
            lowest = last_candles["low"].min()
            return {"highest": float(highest), "lowest": float(lowest)}
        except Exception as e:
            logger.error(f"Error calculating high/low levels: {e}")
            return {"highest": 0.0, "lowest": 0.0}

    def check_ema_crossover(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for EMA5/EMA10 crossover signal with RSI filter"""
        if df is None or len(df) < max(self.ema15_period, self.rsi_period) + 5:
            return {"signal": None, "reason": "Insufficient data"}
        try:
            # Calculate EMAs
            df["ema5"] = df["close"].ewm(span=self.ema5_period, adjust=False).mean()
            df["ema10"] = df["close"].ewm(span=self.ema10_period, adjust=False).mean()
            df["ema15"] = df["close"].ewm(span=self.ema15_period, adjust=False).mean()
            # Calculate RSI
            df["rsi"] = self.calculate_rsi(df["close"], self.rsi_period)
            # Get latest and previous values
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            # Check for crossovers
            bullish_cross = (latest["ema5"] > latest["ema10"]) and (prev["ema5"] <= prev["ema10"])
            bearish_cross = (latest["ema5"] < latest["ema10"]) and (prev["ema5"] >= prev["ema10"])
            # Apply RSI filter
            rsi_long_ok = latest["rsi"] > self.rsi_long_threshold
            rsi_short_ok = latest["rsi"] < self.rsi_short_threshold
            signal = None
            if bullish_cross and rsi_long_ok:
                signal = "LONG"
            elif bearish_cross and rsi_short_ok:
                signal = "SHORT"
            return {
                "signal": signal,
                "price": latest["close"],
                "ema5": latest["ema5"],
                "ema10": latest["ema10"],
                "ema15": latest["ema15"],  # Base price reference
                "rsi": latest["rsi"],
                "volume": latest["volume"],
                "timestamp": latest["datetime"],
                "bullish_cross": bullish_cross,
                "bearish_cross": bearish_cross,
                "rsi_long_ok": rsi_long_ok,
                "rsi_short_ok": rsi_short_ok
            }
        except Exception as e:
            logger.error(f"Signal calculation error: {e}")
            return {"signal": None, "reason": str(e)}

    def format_trading_info(self, symbol: str, signal_5m: Dict, signal_15m: Dict, signal_type: str, levels_15m: Dict) -> str:
        """Format trading information with MEXC Futures details"""
        emoji = "🟢" if signal_type == "LONG" else "🔴"
        message = (
            f"{emoji} *{signal_type} SIGNAL: {symbol}*\n"
            f"💰 Current Price: ${signal_5m['price']:.4f}\n"
            f"📊 Base Price (EMA15): ${signal_5m['ema15']:.4f}\n"
            f"\n📈 *MEXC Futures Trading Info:*\n"
            f"   Symbol: {symbol}\n"
            f"   Type: LIMIT Order\n"
            f"   Side: {'LONG (Buy)' if signal_type == 'LONG' else 'SHORT (Sell)'}\n"
            f"   📊 15m Chart Levels (Last 5 candles):\n"
            f"   • Highest: ${levels_15m['highest']:.4f}\n"
            f"   • Lowest: ${levels_15m['lowest']:.4f}\n"
            f"\n📈 *5M Timeframe:*\n"
            f"   EMA5: ${signal_5m['ema5']:.4f}\n"
            f"   EMA10: ${signal_5m['ema10']:.4f}\n"
            f"   RSI: {signal_5m['rsi']:.1f}\n"
            f"   Volume: {signal_5m['volume']:.0f}\n"
            f"\n📈 *15M Timeframe:*\n"
            f"   EMA5: ${signal_15m['ema5']:.4f}\n"
            f"   EMA10: ${signal_15m['ema10']:.4f}\n"
            f"   RSI: {signal_15m['rsi']:.1f}\n"
            f"   Volume: {signal_15m['volume']:.0f}\n"
            f"\n⏰ Time: {signal_5m['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return message

    def execute_futures_trade(self, symbol: str, signal_type: str, price: float, levels: Dict) -> Dict:
        """Execute futures trade via MEXC API"""
        if not self.futures_api:
            return {"success": False, "reason": "Futures API not initialized"}
        try:
            futures_symbol = symbol.replace('USDT', '_USDT') if 'USDT' in symbol else symbol
            side = 1 if signal_type == "LONG" else 2  # 1=long, 2=short
            order_type = 1  # 1=limit order
            vol = 1  # Default volume - should be configurable
            if signal_type == "LONG":
                limit_price = min(price, levels['lowest'] * 1.001)
            else:
                limit_price = max(price, levels['highest'] * 0.999)
            result = self.futures_api.place_order(
                symbol=futures_symbol,
                side=side,
                order_type=order_type,
                vol=vol,
                price=limit_price
            )
            if 'error' not in result:
                logger.info(f"Futures order placed: {symbol} {signal_type} at {limit_price}")
                return {"success": True, "order_id": result.get('data', 'unknown'), "price": limit_price}
            else:
                logger.error(f"Futures order failed: {result['error']}")
                return {"success": False, "reason": result['error']}
        except Exception as e:
            logger.error(f"Error executing futures trade: {e}")
            return {"success": False, "reason": str(e)}

    def add_symbol(self, symbol: str) -> bool:
        """Add a new symbol to the list"""
        symbol = symbol.upper()
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            return self.save_symbols(self.symbols)
        return False

    def remove_symbol(self, symbol: str) -> bool:
        """Remove a symbol from the list"""
        symbol = symbol.upper()
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            return self.save_symbols(self.symbols)
        return False

    def list_symbols(self) -> str:
        """Return formatted list of current symbols"""
        return f"Current symbols ({len(self.symbols)}):\n" + "\n".join([f"• {symbol}" for symbol in self.symbols])

    def process_telegram_command(self, message_text: str) -> str:
        """Process telegram commands for symbol management"""
        try:
            parts = message_text.strip().split()
            if not parts:
                return "Invalid command format"
            command = parts[0].lower()
            if command == "/add" and len(parts) == 2:
                symbol = parts[1].upper()
                if self.add_symbol(symbol):
                    return f"✅ Added {symbol} to watchlist"
                else:
                    return f"⚠️ {symbol} already in watchlist"
            elif command == "/remove" and len(parts) == 2:
                symbol = parts[1].upper()
                if self.remove_symbol(symbol):
                    return f"✅ Removed {symbol} from watchlist"
                else:
                    return f"⚠️ {symbol} not found in watchlist"
            elif command == "/list":
                return self.list_symbols()
            elif command == "/update" and len(parts) >= 2:
                new_symbols = [s.upper() for s in parts[1:]]
                self.symbols = new_symbols
                if self.save_symbols(self.symbols):
                    return f"✅ Updated watchlist with {len(new_symbols)} symbols:\n" + "\n".join([f"• {s}" for s in new_symbols])
                else:
                    return "❌ Failed to update symbols file"
            elif command == "/stop":
                self.running = False
                return "🛑 Bot stopping..."
            elif command == "/start":
                self.running = True
                return "▶️ Bot starting..."
            elif command == "/interval" and len(parts) == 2:
                try:
                    new_interval = int(parts[1])
                    if 30 <= new_interval <= 3600:
                        self.scan_interval = new_interval
                        return f"✅ Scan interval set to {new_interval} seconds"
                    else:
                        return "❌ Interval must be between 30 and 3600 seconds"
                except ValueError:
                    return "❌ Invalid interval value"
            elif command == "/help":
                return (
                    "📋 *Available Commands:*\n"
                    "/add SYMBOL - Add symbol to watchlist\n"
                    "/remove SYMBOL - Remove symbol from watchlist\n"
                    "/list - Show current symbols\n"
                    "/update SYMBOL1 SYMBOL2... - Replace all symbols\n"
                    "/stop - Stop bot\n"
                    "/start - Start bot\n"
                    "/interval SECONDS - Set scan interval\n"
                    "/status - Show bot status\n"
                    "/help - Show this help"
                )
            elif command == "/status":
                status = "🟢 Running" if self.running else "🔴 Stopped"
                api_status = "✅ Connected" if self.futures_api else "❌ Not configured"
                return (
                    f"🤖 *Bot Status*\n"
                    f"Status: {status}\n"
                    f"Symbols: {len(self.symbols)}\n"
                    f"Scan Interval: {self.scan_interval}s\n"
                    f"Futures API: {api_status}\n"
                    f"Strategy: EMA5/EMA10 + RSI\n"
                    f"Timeframes: 5m & 15m\n"
                    f"RSI: Long >55, Short <45"
                )
            else:
                return "❌ Unknown command. Use /help for available commands"
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            return f"❌ Error processing command: {str(e)}"

    def check_telegram_updates(self):
        """Check for new Telegram messages and process commands"""
        if not self.telegram_token or not self.chat_id:
            return
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates"
            params = {"offset": self.last_update_id + 1, "limit": 10}
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("ok") and data.get("result"):
                for update in data["result"]:
                    update_id = update.get("update_id", 0)
                    message = update.get("message", {})
                    if str(message.get("chat", {}).get("id", "")) == str(self.chat_id):
                        text = message.get("text", "")
                        if text.startswith("/"):
                            logger.info(f"Processing command: {text}")
                            response_text = self.process_telegram_command(text)
                            self.send_telegram_alert(response_text)
                    if update_id > self.last_update_id:
                        self.last_update_id = update_id
                        self.save_last_update_id(self.last_update_id)
        except Exception as e:
            logger.error(f"Error checking Telegram updates: {e}")

    def send_telegram_alert(self, message: str) -> bool:
        """Send alert to Telegram"""
        if not self.telegram_token or not self.chat_id:
            logger.warning("Missing TELEGRAM_TOKEN or CHAT_ID.")
            return False
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            res = self.session.post(url, data=payload, timeout=10)
            res.raise_for_status()
            logger.info("Telegram message sent.")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False

    def run_single_analysis(self):
        """Run a single analysis cycle"""
        logger.info("Running EMA Crossover + RSI analysis...")
        self.check_telegram_updates()
        if not self.running:
            return
        alerts = []
        trades_executed = []
        for symbol in self.symbols:
            df_5m = self.fetch_klines(symbol, "5m", 100)
            df_15m = self.fetch_klines(symbol, "15m", 100)
            if df_5m is None or df_15m is None:
                logger.warning(f"Could not fetch data for {symbol}")
                continue
            signal_5m = self.check_ema_crossover(df_5m)
            signal_15m = self.check_ema_crossover(df_15m)
            levels_15m = self.get_high_low_levels(df_15m, 5)
            signal_detected = None
            if (signal_5m["signal"] == "LONG" and signal_15m["signal"] == "LONG"):
                signal_detected = "LONG"
            elif (signal_5m["signal"] == "SHORT" and signal_15m["signal"] == "SHORT"):
                signal_detected = "SHORT"
            elif signal_5m["signal"] == "LONG" and signal_15m["signal"] != "SHORT":
                if signal_15m["rsi"] > self.rsi_long_threshold:
                    signal_detected = "LONG"
            elif signal_5m["signal"] == "SHORT" and signal_15m["signal"] != "LONG":
                if signal_15m["rsi"] < self.rsi_short_threshold:
                    signal_detected = "SHORT"
            if signal_detected:
                message = self.format_trading_info(symbol, signal_5m, signal_15m, signal_detected, levels_15m)
                alerts.append(message)
                logger.info(f"{signal_detected} signal detected for {symbol}")
                if self.futures_api:
                    trade_result = self.execute_futures_trade(
                        symbol, signal_detected, signal_5m['price'], levels_15m
                    )
                    trades_executed.append(f"{symbol}: {trade_result}")
                logger.info(f"{symbol} 5m - EMA5: {signal_5m['ema5']:.4f}, EMA10: {signal_5m['ema10']:.4f}, RSI: {signal_5m['rsi']:.1f}")
                logger.info(f"{symbol} 15m - EMA5: {signal_15m['ema5']:.4f}, EMA10: {signal_15m['ema10']:.4f}, RSI: {signal_15m['rsi']:.1f}")
                logger.info(f"{symbol} 15m Levels - High: {levels_15m['highest']:.4f}, Low: {levels_15m['lowest']:.4f}")
            else:
                logger.info(f"No clear signal for {symbol}")
        if alerts:
            for alert in alerts:
                self.send_telegram_alert(alert)
            if trades_executed:
                trade_summary = "🔄 *Trades Executed:*\n" + "\n".join(trades_executed)
                self.send_telegram_alert(trade_summary)
        logger.info(f"Analysis complete. {len(alerts)} signals detected, {len(trades_executed)} trades attempted.")

    def run_24_7(self):
        logger.info("Starting 24/7 bot operation...")
        startup_msg = (
            "🤖 *Sig_288bot v2.0 - 24/7 Mode Started*\n"
            "Strategy: EMA5/EMA10 Crossover + RSI Filter\n"
            "Timeframes: 5m & 15m\n"
            "RSI Thresholds: Long >55, Short <45\n"
            f"Scan Interval: {self.scan_interval}s\n"
            f"Futures API: {'✅ Active' if self.futures_api else '❌ Disabled'}\n\n"
            f"📋 Monitoring {len(self.symbols)} symbols:\n" +
            "\n".join([f"• {symbol}" for symbol in self.symbols]) +
            "\n\nUse /help for commands"
        )
        self.send_telegram_alert(startup_msg)
        while True:
            try:
                if self.running:
                    self.run_single_analysis()
                else:
                    logger.info("Bot paused. Waiting for /start command...")
                time.sleep(self.scan_interval)
            except KeyboardInterrupt:
                logger.info("Bot stopped by user (Ctrl+C)")
                self.send_telegram_alert("🛑 Bot manually stopped.")
                break
            except Exception as e:
                logger.error(f"Critical error in main loop: {e}")
                self.send_telegram_alert(f"❗️Bot error: {e}")
                time.sleep(1)

if __name__ == "__main__":
    bot = MEXCBot()
    bot.run_24_7()
