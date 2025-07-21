#!/usr/bin/env python3
"""
Sig_288bot - MEXC EMA20 Volume Strategy
"""

import requests
import pandas as pd
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
        self.ema_period = 20
        self.base_url = "https://api.mexc.com"

        # Telegram credentials
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Sig_288bot/1.0'})

    def fetch_klines(self, symbol: str, interval: str, limit: int = 50) -> Optional[pd.DataFrame]:
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
            df[["close", "volume"]] = df[["close", "volume"]].apply(pd.to_numeric)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol} ({interval}): {e}")
            return None

    def check_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or len(df) < self.ema_period:
            return {"signal": False, "reason": "Insufficient data"}

        try:
            df["ema"] = df["close"].ewm(span=self.ema_period, adjust=False).mean()
            vol_avg = df["volume"].rolling(window=self.ema_period).mean()
            latest = df.iloc[-1]
            prev_vol_avg = vol_avg.iloc[-2] if len(vol_avg) > 1 else vol_avg.iloc[-1]
            signal = latest["volume"] > prev_vol_avg
            return {
                "signal": signal,
                "price": latest["close"],
                "ema": latest["ema"],
                "volume": latest["volume"],
                "vol_avg": prev_vol_avg,
                "price_above_ema": latest["close"] > latest["ema"],
                "volume_spike": latest["volume"] > prev_vol_avg,
                "timestamp": latest["datetime"]
            }
        except Exception as e:
            logger.error(f"Signal calculation error: {e}")
            return {"signal": False, "reason": str(e)}

    def send_telegram_alert(self, message: str) -> bool:
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

    def run_analysis(self):
        logger.info("Running analysis...")
        alerts = []

        for symbol in self.symbols:
            df_15m = self.fetch_klines(symbol, "15m")
            df_30m = self.fetch_klines(symbol, "30m")

            if df_15m is None or df_30m is None:
                logger.warning(f"Could not fetch data for {symbol}")
                continue

            s15 = self.check_signal(df_15m)
            s30 = self.check_signal(df_30m)

            long_signal = (
                s15["signal"] and s30["signal"]
                and s15["price_above_ema"] and s30["price_above_ema"]
            )
            short_signal = (
                s15["signal"] and s30["signal"]
                and not s15["price_above_ema"] and not s30["price_above_ema"]
            )

            if long_signal:
                message = (
                    f"🟢 *LONG SIGNAL: {symbol}*\n"
                    f"Price: ${s15['price']:.2f}\n"
                    f"15m EMA: ${s15['ema']:.2f}, Vol: {s15['volume']:.2f}\n"
                    f"30m EMA: ${s30['ema']:.2f}, Vol: {s30['volume']:.2f}\n"
                    f"Time: {s15['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
                )
                alerts.append(message)
                logger.info(f"LONG signal for {symbol}")

            elif short_signal:
                message = (
                    f"🔴 *SHORT SIGNAL: {symbol}*\n"
                    f"Price: ${s15['price']:.2f}\n"
                    f"15m EMA: ${s15['ema']:.2f}, Vol: {s15['volume']:.2f}\n"
                    f"30m EMA: ${s30['ema']:.2f}, Vol: {s30['volume']:.2f}\n"
                    f"Time: {s15['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
                )
                alerts.append(message)
                logger.info(f"SHORT signal for {symbol}")
            else:
                logger.info(f"No clear signal for {symbol}")

        if alerts:
            self.send_telegram_alert("\n\n".join(alerts))
        else:
            self.send_telegram_alert("📊 No long or short signals detected.")

        logger.info("Analysis complete.")

def main():
    bot = MEXCBot()
    bot.run_analysis()

if __name__ == "__main__":
    main()
