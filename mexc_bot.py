#!/usr/bin/env python3
"""
Sig_288bot - MEXC EMA5/EMA10 Crossover + RSI Strategy
Features:
- EMA5/EMA10 crossover as main signal
- EMA15 as base price reference
- RSI filter: >55 for Long, <45 for Short
- 5m and 15m timeframes
"""

import requests
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MEXCBot:
    def __init__(self):
        self.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "THEUSDT", "XRPUSDT", "SUIUSDT","CHESSUSDT","OGUSDT","MASKUSDT","EDUUSDT","SHIBUSDT"]
        self.base_url = "https://api.mexc.com"
        
        # Strategy parameters
        self.ema5_period = 5
        self.ema10_period = 10
        self.ema15_period = 15
        self.rsi_period = 14
        self.rsi_long_threshold = 55
        self.rsi_short_threshold = 45

        # Telegram credentials
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Sig_288bot/2.0'})

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

    def format_signal_message(self, symbol: str, signal_5m: Dict, signal_15m: Dict, signal_type: str) -> str:
        """Format signal message for Telegram"""
        emoji = "🟢" if signal_type == "LONG" else "🔴"
        
        message = (
            f"{emoji} *{signal_type} SIGNAL: {symbol}*\n"
            f"💰 Current Price: ${signal_5m['price']:.4f}\n"
            f"📊 Base Price (EMA15): ${signal_5m['ema15']:.4f}\n"
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

    def run_analysis(self):
        """Main analysis function"""
        logger.info("Running EMA Crossover + RSI analysis...")
        alerts = []

        for symbol in self.symbols:
            # Fetch data for both timeframes
            df_5m = self.fetch_klines(symbol, "5m", 100)
            df_15m = self.fetch_klines(symbol, "15m", 100)

            if df_5m is None or df_15m is None:
                logger.warning(f"Could not fetch data for {symbol}")
                continue

            # Check signals on both timeframes
            signal_5m = self.check_ema_crossover(df_5m)
            signal_15m = self.check_ema_crossover(df_15m)

            # Determine if we have a valid signal
            signal_detected = None
            
            # Both timeframes must agree on signal direction
            if (signal_5m["signal"] == "LONG" and signal_15m["signal"] == "LONG"):
                signal_detected = "LONG"
            elif (signal_5m["signal"] == "SHORT" and signal_15m["signal"] == "SHORT"):
                signal_detected = "SHORT"
            # Alternative: Allow signal if at least 5m has signal and 15m doesn't contradict
            elif signal_5m["signal"] == "LONG" and signal_15m["signal"] != "SHORT":
                if signal_15m["rsi"] > self.rsi_long_threshold:
                    signal_detected = "LONG"
            elif signal_5m["signal"] == "SHORT" and signal_15m["signal"] != "LONG":
                if signal_15m["rsi"] < self.rsi_short_threshold:
                    signal_detected = "SHORT"

            if signal_detected:
                message = self.format_signal_message(symbol, signal_5m, signal_15m, signal_detected)
                alerts.append(message)
                logger.info(f"{signal_detected} signal detected for {symbol}")
                
                # Log detailed signal info
                logger.info(f"{symbol} 5m - EMA5: {signal_5m['ema5']:.4f}, EMA10: {signal_5m['ema10']:.4f}, RSI: {signal_5m['rsi']:.1f}")
                logger.info(f"{symbol} 15m - EMA5: {signal_15m['ema5']:.4f}, EMA10: {signal_15m['ema10']:.4f}, RSI: {signal_15m['rsi']:.1f}")
            else:
                logger.info(f"No clear signal for {symbol}")

        # Send alerts
        if alerts:
            # Send each alert separately to avoid message length limits
            for alert in alerts:
                self.send_telegram_alert(alert)
        else:
            summary_msg = "📊 *Market Scan Complete*\nNo EMA crossover signals detected with RSI confirmation."
            self.send_telegram_alert(summary_msg)

        logger.info(f"Analysis complete. {len(alerts)} signals detected.")

    def get_market_overview(self) -> str:
        """Generate a quick market overview"""
        try:
            overview_data = []
            for symbol in self.symbols[:5]:  # Limit to first 5 for overview
                df = self.fetch_klines(symbol, "15m", 50)
                if df is not None:
                    signal = self.check_ema_crossover(df)
                    if signal["signal"] is None:
                        trend = "NEUTRAL"
                    else:
                        trend = signal["signal"]
                    
                    overview_data.append(f"{symbol}: {trend} (RSI: {signal.get('rsi', 0):.1f})")
            
            return "\n".join(overview_data)
        except Exception as e:
            logger.error(f"Error generating market overview: {e}")
            return "Market overview unavailable"

def main():
    """Main function"""
    bot = MEXCBot()
    
    # Optional: Send startup message
    startup_msg = (
        "🤖 *Sig_288bot v2.0 Started*\n"
        "Strategy: EMA5/EMA10 Crossover + RSI Filter\n"
        "Timeframes: 5m & 15m\n"
        "RSI Thresholds: Long >55, Short <45"
    )
    bot.send_telegram_alert(startup_msg)
    
    # Run analysis
    bot.run_analysis()

if __name__ == "__main__":
    main()
