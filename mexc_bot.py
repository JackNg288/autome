#!/usr/bin/env python3
"""
Sig_288bot - MEXC EMA5/EMA10 Crossover + RSI Strategy with Price Alerts
Features:
- EMA5/EMA10 crossover as main signal
- EMA15 as base price reference
- RSI filter: >55 for Long, <45 for Short
- 5m and 15m timeframes
- 24/7 continuous operation
- Price alert system via Telegram
- Auto-alert on base price (EMA15) when signal detected
"""
import requests
import pandas as pd
import numpy as np
import os
import os
from dotenv import load_dotenv
load_dotenv()
import logging
import time
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import threading
from urllib.parse import urlencode

# Additional imports for ChatGPT integration
try:
    import openai  # type: ignore
except ImportError:
    openai = None  # openai may not be available if the package is not installed


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

class MEXCBot:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Sig_288bot/2.0'})
        self.symbols_file = "symbols.txt"
        self.alerts_file = "price_alerts.json"
        self.symbols = self.load_symbols()
        self.price_alerts = self.load_price_alerts()
        self.base_url = "https://api.mexc.com"
        
        # Strategy parameters
        self.ema5_period = 5
        self.ema10_period = 10
        self.ema15_period = 15
        self.rsi_period = 14
        self.rsi_long_threshold = 55
        self.rsi_short_threshold = 45
        
        # 24/7 operation settings
        self.scan_interval = 30  # seconds between scans
        self.running = True
        
        # Telegram credentials
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        # OpenAI credentials for ChatGPT analysis
        # You must set OPENAI_API_KEY in your environment for this to work
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai and self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # Track last processed update to avoid duplicates
        self.last_update_id = self.load_last_update_id()
        
    def load_symbols(self) -> List[str]:
        """Load symbols from symbols.txt file"""
        default_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "THEUSDT", "XRPUSDT", "SUIUSDT","CHESSUSDT","OGUSDT","MASKUSDT","EDUUSDT","SHIBUSDT","TRUMPUSDT","FUNUSDT","HYPEUSDT","LDOUSDT","FETUSDT","EIGENUSDT","TOKENUSDT","ZKUSDT","JASMYUSDT","ADAUSDT","OMUSDT","LTCUSDT","APTUSDT"]
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

    def load_price_alerts(self) -> Dict[str, List[Dict]]:
        """Load price alerts from JSON file"""
        try:
            if os.path.exists(self.alerts_file):
                with open(self.alerts_file, 'r') as f:
                    alerts = json.load(f)
                    logger.info(f"Loaded {sum(len(v) for v in alerts.values())} price alerts")
                    return alerts
            else:
                return {}
        except Exception as e:
            logger.error(f"Error loading price alerts: {e}")
            return {}

    def save_price_alerts(self) -> bool:
        """Save price alerts to JSON file"""
        try:
            with open(self.alerts_file, 'w') as f:
                json.dump(self.price_alerts, f, indent=2)
            logger.info(f"Saved {sum(len(v) for v in self.price_alerts.values())} price alerts")
            return True
        except Exception as e:
            logger.error(f"Error saving price alerts: {e}")
            return False

    def add_price_alert(self, symbol: str, target_price: float, alert_type: str = "manual") -> bool:
        """Add a price alert for a symbol"""
        symbol = symbol.upper()
        if symbol not in self.price_alerts:
            self.price_alerts[symbol] = []
        
        # Check if alert already exists
        for alert in self.price_alerts[symbol]:
            if abs(alert['price'] - target_price) < 0.0001:  # Prevent duplicate alerts
                return False
        
        alert = {
            "price": target_price,
            "type": alert_type,
            "created": datetime.now().isoformat(),
            "id": f"{symbol}_{target_price}_{int(time.time())}"
        }
        
        self.price_alerts[symbol].append(alert)
        self.save_price_alerts()
        return True

    def remove_price_alert(self, symbol: str, target_price: float = None, alert_id: str = None) -> bool:
        """Remove a price alert"""
        symbol = symbol.upper()
        if symbol not in self.price_alerts:
            return False
        
        original_count = len(self.price_alerts[symbol])
        
        if alert_id:
            self.price_alerts[symbol] = [a for a in self.price_alerts[symbol] if a['id'] != alert_id]
        elif target_price is not None:
            self.price_alerts[symbol] = [a for a in self.price_alerts[symbol] if abs(a['price'] - target_price) > 0.0001]
        
        # Clean up empty symbol entries
        if not self.price_alerts[symbol]:
            del self.price_alerts[symbol]
        
        removed = original_count - len(self.price_alerts.get(symbol, []))
        if removed > 0:
            self.save_price_alerts()
            return True
        return False

    def check_price_alerts(self, symbol: str, current_price: float) -> List[Dict]:
        """Check if current price triggers any alerts"""
        triggered_alerts = []
        if symbol not in self.price_alerts:
            return triggered_alerts
        
        alerts_to_remove = []
        for alert in self.price_alerts[symbol]:
            target_price = alert['price']
            
            # Check if price target is reached (with small tolerance for floating point)
            if abs(current_price - target_price) <= (target_price * 0.002):  # 0.2% tolerance
                triggered_alerts.append({
                    'symbol': symbol,
                    'target_price': target_price,
                    'current_price': current_price,
                    'type': alert['type'],
                    'created': alert['created'],
                    'alert_id': alert['id']
                })
                alerts_to_remove.append(alert['id'])
        
        # Remove triggered alerts
        for alert_id in alerts_to_remove:
            self.remove_price_alert(symbol, alert_id=alert_id)
        
        return triggered_alerts

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
        """Format trading information without trading execution"""
        emoji = "🟢" if signal_type == "LONG" else "🔴"
        message = (
            f"{emoji} *{signal_type} SIGNAL: {symbol}*\n"
            f"💰 Current Price: ${signal_5m['price']:.4f}\n"
            f"🎯 Base Price (EMA15): ${signal_5m['ema15']:.4f}\n"
            f"📊 Alert Set: Price reaches ${signal_5m['ema15']:.4f}\n"
            f"\n📈 *Market Analysis:*\n"
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

    # ------------------------------------------------------------------
    # Trend analysis helpers
    def get_trend_data(self, symbol: str, df_15m: pd.DataFrame) -> Dict[str, Any]:
        """Compute 24h high/low and trend directions for 15m and 1h charts.

        Returns a dictionary with keys: high_24h, low_24h, trend_15m, change_15m,
        trend_1h, change_1h. If insufficient data is available, values may be
        None. This helper can be called after fetching klines in run_single_analysis.
        """
        trend_data: Dict[str, Any] = {
            "high_24h": None,
            "low_24h": None,
            "trend_15m": None,
            "change_15m": None,
            "trend_1h": None,
            "change_1h": None
        }
        try:
            # Fetch 1h klines for the last 24 hours (24 candles)
            df_1h = self.fetch_klines(symbol, "1h", 24)
            if df_1h is not None and len(df_1h) >= 1:
                # Calculate high and low across 24 hours
                trend_data["high_24h"] = float(df_1h["high"].max())
                trend_data["low_24h"] = float(df_1h["low"].min())
                # Determine 1h trend direction based on first open and last close
                first_open_1h = float(df_1h.iloc[0]["open"])
                last_close_1h = float(df_1h.iloc[-1]["close"])
                if last_close_1h > first_open_1h:
                    trend_data["trend_1h"] = "uptrend"
                elif last_close_1h < first_open_1h:
                    trend_data["trend_1h"] = "downtrend"
                else:
                    trend_data["trend_1h"] = "flat"
                if first_open_1h != 0:
                    trend_data["change_1h"] = ((last_close_1h - first_open_1h) / first_open_1h) * 100.0
            # Determine 15m trend direction using the provided df_15m (last 4 candles ~1h)
            if df_15m is not None and len(df_15m) >= 5:
                # Use last 5 candles for smoother trend estimation
                subset = df_15m.tail(5)
                first_open_15m = float(subset.iloc[0]["open"])
                last_close_15m = float(subset.iloc[-1]["close"])
                if last_close_15m > first_open_15m:
                    trend_data["trend_15m"] = "uptrend"
                elif last_close_15m < first_open_15m:
                    trend_data["trend_15m"] = "downtrend"
                else:
                    trend_data["trend_15m"] = "flat"
                if first_open_15m != 0:
                    trend_data["change_15m"] = ((last_close_15m - first_open_15m) / first_open_15m) * 100.0
        except Exception as e:
            logger.error(f"Error computing trend data for {symbol}: {e}")
        return trend_data

    def call_chatgpt_analysis(self, symbol: str, trend_data: Dict[str, Any]) -> str:
        """Generate a detailed analysis string using ChatGPT based on trend data.

        If the OpenAI API key is not available or the openai package is missing,
        this returns a simple analysis summarizing the computed metrics. Otherwise,
        it sends a prompt to ChatGPT to perform deeper research, including
        identifying recent events that may influence the symbol's price. The
        returned string is intended to be appended to the Telegram alert.
        """
        # Fallback summarization if ChatGPT integration is unavailable
        def fallback_summary(trend: Dict[str, Any]) -> str:
            summary_parts = []
            high_24h = trend.get("high_24h")
            low_24h = trend.get("low_24h")
            if high_24h is not None and low_24h is not None:
                summary_parts.append(f"24h High: ${high_24h:.2f}, Low: ${low_24h:.2f}.")
            if trend.get("trend_15m"):
                change_15m = trend.get("change_15m")
                if change_15m is not None:
                    summary_parts.append(
                        f"15m trend: {trend['trend_15m']} (≈{change_15m:+.2f}% over last 5 candles)."
                    )
            if trend.get("trend_1h"):
                change_1h = trend.get("change_1h")
                if change_1h is not None:
                    summary_parts.append(
                        f"1h trend: {trend['trend_1h']} (≈{change_1h:+.2f}% over last 24 candles)."
                    )
            summary_parts.append("Recent events: Unable to fetch news without ChatGPT API.")
            return "\n".join(summary_parts)

        # If OpenAI or API key is missing, use fallback
        if openai is None or not self.openai_api_key:
            return fallback_summary(trend_data)

        # Construct the prompt for ChatGPT
        high_24h = trend_data.get("high_24h")
        low_24h = trend_data.get("low_24h")
        trend_15m = trend_data.get("trend_15m")
        change_15m = trend_data.get("change_15m")
        trend_1h = trend_data.get("trend_1h")
        change_1h = trend_data.get("change_1h")

        # Format numbers gracefully
        high_str = f"${high_24h:,.2f}" if high_24h is not None else "N/A"
        low_str = f"${low_24h:,.2f}" if low_24h is not None else "N/A"
        change_15m_str = f"{change_15m:+.2f}%" if change_15m is not None else "N/A"
        change_1h_str = f"{change_1h:+.2f}%" if change_1h is not None else "N/A"
        trend_15m_str = trend_15m if trend_15m else "N/A"
        trend_1h_str = trend_1h if trend_1h else "N/A"

        prompt = (
            f"You are a cryptocurrency market analyst. Analyze the following metrics for the trading pair {symbol} "
            f"and provide a concise yet insightful summary. The data is as follows: 24-hour high is {high_str}, "
            f"24-hour low is {low_str}. On the 15-minute chart the price shows a {trend_15m_str} with a change of {change_15m_str}. "
            f"On the 1-hour chart the price shows a {trend_1h_str} with a change of {change_1h_str}. "
            f"In addition to summarizing this data, search for and mention any recent news events or developments in the last 24 "
            f"hours that could influence {symbol}'s price movement. Use bullet points for different news items and explain why "
            f"they might matter. If no recent events are found, explicitly state that. Keep the response under 5 bullet points."
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in cryptocurrency market analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )
            analysis_text = response["choices"][0]["message"]["content"].strip()
            return analysis_text
        except Exception as e:
            logger.error(f"ChatGPT analysis error: {e}")
            return fallback_summary(trend_data)

    def format_price_alert(self, alert: Dict) -> str:
        """Format price alert message"""
        alert_type_emoji = "🤖" if alert['type'] == "auto" else "🔔"
        return (
            f"{alert_type_emoji} *PRICE ALERT: {alert['symbol']}*\n"
            f"🎯 Target Price: ${alert['target_price']:.4f}\n"
            f"💰 Current Price: ${alert['current_price']:.4f}\n"
            f"📊 Alert Type: {alert['type'].upper()}\n"
            f"⏰ Triggered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

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

    def list_alerts(self) -> str:
        """Return formatted list of current price alerts"""
        if not self.price_alerts:
            return "📭 No active price alerts"
        
        message = "🔔 *Active Price Alerts:*\n\n"
        total_alerts = 0
        for symbol, alerts in self.price_alerts.items():
            if alerts:
                message += f"*{symbol}:*\n"
                for alert in alerts:
                    alert_type = "🤖 Auto" if alert['type'] == "auto" else "👤 Manual"
                    message += f"  • ${alert['price']:.4f} ({alert_type})\n"
                total_alerts += len(alerts)
                message += "\n"
        
        message += f"Total: {total_alerts} alerts"
        return message

    def process_telegram_command(self, message_text: str) -> str:
        """Process telegram commands for symbol and alert management"""
        try:
            parts = message_text.strip().split()
            if not parts:
                return "Invalid command format"
            command = parts[0].lower()
            
            if command == "/alert" and len(parts) == 3:
                symbol = parts[1].upper()
                try:
                    price = float(parts[2])
                    if self.add_price_alert(symbol, price, "manual"):
                        return f"🔔 Price alert set for {symbol} at ${price:.4f}"
                    else:
                        return f"⚠️ Alert for {symbol} at ${price:.4f} already exists"
                except ValueError:
                    return "❌ Invalid price format"
            elif command == "/removealert" and len(parts) == 3:
                symbol = parts[1].upper()
                try:
                    price = float(parts[2])
                    if self.remove_price_alert(symbol, price):
                        return f"✅ Removed alert for {symbol} at ${price:.4f}"
                    else:
                        return f"⚠️ Alert not found for {symbol} at ${price:.4f}"
                except ValueError:
                    return "❌ Invalid price format"
            elif command == "/alerts":
                return self.list_alerts()
            elif command == "/add" and len(parts) == 2:
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
                    "📋 *Available Commands:*\n\n"
                    "*Symbol Management:*\n"
                    "/add SYMBOL - Add symbol to watchlist\n"
                    "/remove SYMBOL - Remove symbol from watchlist\n"
                    "/list - Show current symbols\n"
                    "/update SYMBOL1 SYMBOL2... - Replace all symbols\n\n"
                    "*Price Alerts:*\n"
                    "/alert SYMBOL PRICE - Set price alert\n"
                    "/removealert SYMBOL PRICE - Remove price alert\n"
                    "/alerts - Show all active alerts\n\n"
                    "*Bot Control:*\n"
                    "/stop - Stop bot\n"
                    "/start - Start bot\n"
                    "/interval SECONDS - Set scan interval\n"
                    "/status - Show bot status\n"
                    "/help - Show this help\n\n"
                    "*Examples:*\n"
                    "/alert BTCUSDT 119000\n"
                    "/removealert BTCUSDT 119000"
                )
            elif command == "/status":
                status = "🟢 Running" if self.running else "🔴 Stopped"
                total_alerts = sum(len(v) for v in self.price_alerts.values())
                return (
                    f"🤖 *Bot Status*\n"
                    f"Status: {status}\n"
                    f"Symbols: {len(self.symbols)}\n"
                    f"Active Alerts: {total_alerts}\n"
                    f"Scan Interval: {self.scan_interval}s\n"
                    f"Strategy: EMA5/EMA10 + RSI + Price Alerts\n"
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
        price_alerts_triggered = []
        
        for symbol in self.symbols:
            df_5m = self.fetch_klines(symbol, "5m", 100)
            df_15m = self.fetch_klines(symbol, "15m", 100)
            if df_5m is None or df_15m is None:
                logger.warning(f"Could not fetch data for {symbol}")
                continue
            
            current_price = float(df_5m.iloc[-1]['close'])
            
            # Check price alerts first
            triggered = self.check_price_alerts(symbol, current_price)
            for alert in triggered:
                alert_message = self.format_price_alert(alert)
                price_alerts_triggered.append(alert_message)
                logger.info(f"Price alert triggered for {symbol} at ${current_price:.4f}")
            
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
                # Prepare the basic trading info message
                base_message = self.format_trading_info(symbol, signal_5m, signal_15m, signal_detected, levels_15m)

                # Gather trend metrics and request ChatGPT analysis
                trend_data = self.get_trend_data(symbol, df_15m)
                analysis_text = self.call_chatgpt_analysis(symbol, trend_data)

                # Combine base message with analysis
                full_message = base_message + "\n\n📋 *Trend & News Analysis:*\n" + analysis_text

                alerts.append(full_message)
                logger.info(f"{signal_detected} signal detected for {symbol}")
                
                # Auto-set alert for base price (EMA15)
                base_price = signal_5m['ema15']
                if self.add_price_alert(symbol, base_price, "auto"):
                    logger.info(f"Auto-alert set for {symbol} at base price ${base_price:.4f}")
                
                logger.info(f"{symbol} 5m - EMA5: {signal_5m['ema5']:.4f}, EMA10: {signal_5m['ema10']:.4f}, RSI: {signal_5m['rsi']:.1f}")
                logger.info(f"{symbol} 15m - EMA5: {signal_15m['ema5']:.4f}, EMA10: {signal_15m['ema10']:.4f}, RSI: {signal_15m['rsi']:.1f}")
                logger.info(f"{symbol} 15m Levels - High: {levels_15m['highest']:.4f}, Low: {levels_15m['lowest']:.4f}")
            else:
                logger.info(f"No clear signal for {symbol}")
        
        # Send all alerts
        if alerts:
            for alert in alerts:
                self.send_telegram_alert(alert)
        
        if price_alerts_triggered:
            for price_alert in price_alerts_triggered:
                self.send_telegram_alert(price_alert)
        
        logger.info(f"Analysis complete. {len(alerts)} signals detected, {len(price_alerts_triggered)} price alerts triggered.")

    def run_24_7(self):
        logger.info("Starting 24/7 bot operation...")
        startup_msg = (
            "🤖 *Sig_288bot v2.0 - 24/7 Mode Started*\n"
            "Strategy: EMA5/EMA10 Crossover + RSI Filter + Price Alerts\n"
            "Timeframes: 5m & 15m\n"
            "RSI Thresholds: Long >55, Short <45\n"
            f"Scan Interval: {self.scan_interval}s\n"
            f"Trading: ❌ Disabled (Alert Mode Only)\n\n"
            f"📋 Monitoring {len(self.symbols)} symbols:\n" +
            "\n".join([f"• {symbol}" for symbol in self.symbols]) +
            "\n\n🔔 Features:\n"
            "• Manual alerts: /alert SYMBOL PRICE\n"
            "• Auto alerts on base price (EMA15)\n"
            "• Real-time price monitoring\n\n"
            "Use /help for commands"
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
                time.sleep(60)  # Wait longer on error

if __name__ == "__main__":
    bot = MEXCBot()
    bot.run_24_7()
