#!/usr/bin/env python3
"""
Fully Integrated MEXC Bot + Signal Analyzer System
Direct signal transmission without JSON files
"""

import requests
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
import logging
import time
import json
from datetime import datetime
import threading
from typing import Optional, Dict, Any, List
import sqlite3

# Import your existing modules
from signal_analyzer import SignalAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IntegratedMEXCBot:
    """MEXC Bot with direct signal analyzer integration"""
    
    def __init__(self, analyzer: SignalAnalyzer):
        """Initialize with direct analyzer reference"""
        self.analyzer = analyzer
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'IntegratedBot/1.0'})
        
        # Load existing configuration
        self.symbols_file = "symbols.txt"
        self.alerts_file = "price_alerts.json"
        self.symbols = self.load_symbols()
        self.price_alerts = self.load_price_alerts()
        self.base_url = "https://api.mexc.com"
        
        # Strategy parameters (from your original bot)
        self.ema5_period = 5
        self.ema10_period = 10
        self.ema15_period = 15
        self.rsi_period = 14
        self.rsi_long_threshold = 55
        self.rsi_short_threshold = 45
        
        # Operation settings
        self.scan_interval = 30
        self.running = True
        
        # Telegram credentials
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.last_update_id = self.load_last_update_id()
        
        logger.info("Integrated MEXC Bot initialized with direct analyzer connection")
    
    def load_symbols(self) -> List[str]:
        """Load symbols from file"""
        default_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
        try:
            if os.path.exists(self.symbols_file):
                with open(self.symbols_file, 'r') as f:
                    symbols = [line.strip().upper() for line in f.readlines() if line.strip()]
                    if symbols:
                        logger.info(f"Loaded {len(symbols)} symbols")
                        return symbols
            return default_symbols
        except Exception as e:
            logger.error(f"Error loading symbols: {e}")
            return default_symbols
    
    def load_price_alerts(self) -> Dict[str, List[Dict]]:
        """Load price alerts from JSON file"""
        try:
            if os.path.exists(self.alerts_file):
                with open(self.alerts_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading price alerts: {e}")
            return {}
    
    def load_last_update_id(self) -> int:
        """Load last Telegram update ID"""
        try:
            if os.path.exists("last_update.txt"):
                with open("last_update.txt", 'r') as f:
                    return int(f.read().strip())
        except:
            pass
        return 0
    
    def save_last_update_id(self, update_id: int):
        """Save last Telegram update ID"""
        try:
            with open("last_update.txt", 'w') as f:
                f.write(str(update_id))
        except Exception as e:
            logger.error(f"Error saving update ID: {e}")
    
    def fetch_klines(self, symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch kline data from MEXC"""
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
                "timestamp", "open", "high", "low", "close", "volume", 
                "close_time", "quote_volume"
            ])
            
            # Convert to numeric
            numeric_cols = ["open", "high", "low", "close", "volume"]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
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
            return pd.Series([50] * len(prices))
    
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
                "price": float(latest["close"]),
                "ema5": float(latest["ema5"]),
                "ema10": float(latest["ema10"]),
                "ema15": float(latest["ema15"]),
                "rsi": float(latest["rsi"]),
                "volume": float(latest["volume"]),
                "timestamp": latest["datetime"]
            }
            
        except Exception as e:
            logger.error(f"Signal calculation error: {e}")
            return {"signal": None, "reason": str(e)}
    
    def get_high_low_levels(self, df: pd.DataFrame, candles: int = 5) -> Dict[str, float]:
        """Get highest and lowest prices from last N candles"""
        if df is None or len(df) < candles:
            return {"highest": 0.0, "lowest": 0.0}
        try:
            last_candles = df.tail(candles)
            return {
                "highest": float(last_candles["high"].max()),
                "lowest": float(last_candles["low"].min())
            }
        except Exception as e:
            logger.error(f"Error calculating high/low levels: {e}")
            return {"highest": 0.0, "lowest": 0.0}
    
    def format_signal_message(self, symbol: str, signal_type: str, signal_data: Dict, levels: Dict) -> str:
        """Format signal message for Telegram"""
        emoji = "🟢" if signal_type == "LONG" else "🔴"
        
        message = (
            f"{emoji} *{signal_type} SIGNAL DETECTED: {symbol}*\n"
            f"💰 Entry Price: ${signal_data['price']:.4f}\n"
            f"🎯 Base Price (EMA15): ${signal_data['ema15']:.4f}\n\n"
            f"📊 *Technical Indicators:*\n"
            f"• EMA5: ${signal_data['ema5']:.4f}\n"
            f"• EMA10: ${signal_data['ema10']:.4f}\n"
            f"• RSI: {signal_data['rsi']:.1f}\n"
            f"• Volume: {signal_data['volume']:.0f}\n\n"
            f"📈 *15m Levels:*\n"
            f"• High: ${levels['highest']:.4f}\n"
            f"• Low: ${levels['lowest']:.4f}\n\n"
            f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"📊 Signal sent to analyzer for tracking"
        )
        
        return message
    
    def send_telegram_alert(self, message: str) -> bool:
        """Send alert to Telegram"""
        if not self.telegram_token or not self.chat_id:
            logger.warning("Missing Telegram credentials")
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
            logger.info("Telegram message sent")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
    
    def check_telegram_updates(self):
        """Check for Telegram updates and process commands"""
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
                            # Forward to analyzer for processing
                            response_text = self.analyzer.process_telegram_command(text)
                            self.send_telegram_alert(response_text)
                    
                    if update_id > self.last_update_id:
                        self.last_update_id = update_id
                        self.save_last_update_id(self.last_update_id)
                        
        except Exception as e:
            logger.error(f"Error checking Telegram updates: {e}")
    
    def run_single_analysis(self):
        """Run single analysis cycle with direct analyzer integration"""
        logger.info("Running integrated analysis cycle...")
        
        # Check Telegram updates first
        self.check_telegram_updates()
        
        if not self.running:
            return
        
        signals_detected = 0
        
        for symbol in self.symbols:
            try:
                # Fetch data
                df_5m = self.fetch_klines(symbol, "5m", 100)
                df_15m = self.fetch_klines(symbol, "15m", 100)
                
                if df_5m is None or df_15m is None:
                    logger.warning(f"Could not fetch data for {symbol}")
                    continue
                
                # Check for signals
                signal_5m = self.check_ema_crossover(df_5m)
                signal_15m = self.check_ema_crossover(df_15m)
                levels_15m = self.get_high_low_levels(df_15m, 5)
                
                # Determine if we have a valid signal
                signal_detected = None
                
                if signal_5m["signal"] == "LONG" and signal_15m["signal"] == "LONG":
                    signal_detected = "LONG"
                elif signal_5m["signal"] == "SHORT" and signal_15m["signal"] == "SHORT":
                    signal_detected = "SHORT"
                elif signal_5m["signal"] == "LONG" and signal_15m["signal"] != "SHORT":
                    if signal_15m.get("rsi", 50) > self.rsi_long_threshold:
                        signal_detected = "LONG"
                elif signal_5m["signal"] == "SHORT" and signal_15m["signal"] != "LONG":
                    if signal_15m.get("rsi", 50) < self.rsi_short_threshold:
                        signal_detected = "SHORT"
                
                if signal_detected:
                    # Prepare signal data for analyzer
                    signal_data = {
                        'symbol': symbol,
                        'direction': signal_detected,
                        'entry_price': signal_5m['price'],
                        'base_price': signal_5m['ema15'],
                        'confidence': self._calculate_confidence(signal_5m, signal_15m),
                        'source': 'BOT',
                        'timeframe': '5m',
                        'indicators': {
                            'ema5': signal_5m['ema5'],
                            'ema10': signal_5m['ema10'],
                            'ema15': signal_5m['ema15'],
                            'rsi': signal_5m['rsi'],
                            'volume': signal_5m['volume']
                        },
                        'atr_percent': 2.0  # Default 2% for TP/SL
                    }
                    
                    # Send signal directly to analyzer
                    analyzer_result = self.analyzer.capture_bot_signal(signal_data)
                    logger.info(f"Signal sent to analyzer: {analyzer_result}")
                    
                    # Format and send Telegram message
                    telegram_message = self.format_signal_message(symbol, signal_detected, signal_5m, levels_15m)
                    self.send_telegram_alert(telegram_message)
                    
                    signals_detected += 1
                    
                    logger.info(
                        f"{signal_detected} signal for {symbol} - "
                        f"Price: ${signal_5m['price']:.4f}, "
                        f"RSI: {signal_5m['rsi']:.1f}"
                    )
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Update active signals in analyzer
        self.analyzer.update_active_signals()
        
        logger.info(f"Analysis cycle complete. {signals_detected} signals detected and sent to analyzer")
    
    def _calculate_confidence(self, signal_5m: Dict, signal_15m: Dict) -> float:
        """Calculate signal confidence based on multiple factors"""
        confidence = 50.0  # Base confidence
        
        # RSI confirmation
        rsi_5m = signal_5m.get('rsi', 50)
        if 55 < rsi_5m < 70:  # Good RSI range for long
            confidence += 10
        elif 30 < rsi_5m < 45:  # Good RSI range for short
            confidence += 10
        
        # EMA alignment
        if signal_5m.get('ema5', 0) > signal_5m.get('ema10', 0) > signal_5m.get('ema15', 0):
            confidence += 10  # Bullish alignment
        elif signal_5m.get('ema5', 0) < signal_5m.get('ema10', 0) < signal_5m.get('ema15', 0):
            confidence += 10  # Bearish alignment
        
        # Both timeframes agree
        if signal_5m.get('signal') == signal_15m.get('signal') and signal_5m.get('signal') is not None:
            confidence += 20
        
        return min(confidence, 90)  # Cap at 90%
    
    def run_forever(self, interval: int = 30):
        """Run the integrated bot continuously"""
        logger.info(f"🚀 Starting Integrated MEXC Bot + Analyzer System")
        
        startup_msg = (
            "🤖 *Integrated Bot + Analyzer Started*\n\n"
            "✅ Direct signal transmission enabled\n"
            "✅ Real-time performance tracking\n"
            "✅ Automatic win/loss detection\n\n"
            f"📊 Monitoring {len(self.symbols)} symbols\n"
            f"⏱ Scan interval: {interval}s\n\n"
            "Use /help for analyzer commands"
        )
        self.send_telegram_alert(startup_msg)
        
        try:
            while True:
                try:
                    self.run_single_analysis()
                except Exception as e:
                    logger.error(f"Error in analysis cycle: {e}")
                    time.sleep(60)  # Wait longer on error
                    continue
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            self.send_telegram_alert("🛑 Integrated Bot stopped")


def run_integrated_system():
    """Main function to run the integrated system"""
    logger.info("Initializing Integrated Bot + Analyzer System...")
    
    # Create analyzer instance
    analyzer = SignalAnalyzer()
    
    # Create integrated bot with analyzer
    bot = IntegratedMEXCBot(analyzer)
    
    # Start analyzer update thread
    analyzer_thread = threading.Thread(
        target=analyzer.run_forever,
        args=(60, 3600),  # Update every 60s, report every hour
        daemon=True
    )
    analyzer_thread.start()
    logger.info("Analyzer thread started")
    
    # Run bot in main thread
    bot.run_forever(interval=30)


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║          INTEGRATED MEXC BOT + SIGNAL ANALYZER SYSTEM             ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    🚀 Features:
    • Direct signal transmission (no JSON files)
    • Real-time performance tracking
    • Automatic win/loss detection
    • Comprehensive analytics
    • Telegram integration
    
    📊 Commands:
    • /long SYMBOL PRICE - Add manual long signal
    • /short SYMBOL PRICE - Add manual short signal
    • /performance - View performance report
    • /winrate - Check win rate statistics
    • /active - Show active signals
    • /help - All commands
    
    Starting system...
    """)
    
    run_integrated_system()
