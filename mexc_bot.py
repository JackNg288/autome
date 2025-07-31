#!/usr/bin/env python3
"""
Modified MEXC Bot - Integrated with Signal Analyzer
This version sends signals to the analyzer for performance tracking
"""

import json
from datetime import datetime
from typing import Dict
import time
import os
import logging

# Setup logging (add if missing)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SignalOutput:
    """Output signals for the analyzer"""
    def __init__(self):
        self.signal_file = "active_signals.json"

    def save_signal(self, symbol: str, signal_type: str, signal_data: Dict):
        """Save signal for analyzer to process"""
        try:
            signal_record = {
                'signal_id': f"{symbol}_{signal_type}_{int(time.time())}",
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'direction': signal_type,
                'entry_price': signal_data.get('price', 0),
                'base_price': signal_data.get('ema15', 0),
                'confidence': self._calculate_confidence(signal_data),
                'source': 'BOT',
                'timeframe': '5m',
                'indicators': {
                    'ema5': signal_data.get('ema5', 0),
                    'ema10': signal_data.get('ema10', 0),
                    'ema15': signal_data.get('ema15', 0),
                    'rsi': signal_data.get('rsi', 50),
                    'volume': signal_data.get('volume', 0)
                },
                'atr_percent': 2.0
            }
            signals = []
            if os.path.exists(self.signal_file):
                try:
                    with open(self.signal_file, 'r') as f:
                        signals = json.load(f)
                except:
                    signals = []
            signals.append(signal_record)
            with open(self.signal_file, 'w') as f:
                json.dump(signals, f, indent=2)
            logger.info(f"Signal saved for analyzer: {signal_record['signal_id']}")
            return True
        except Exception as e:
            logger.error(f"Error saving signal for analyzer: {e}")
            return False

    def _calculate_confidence(self, signal_data: Dict) -> float:
        confidence = 50.0
        rsi = signal_data.get('rsi', 50)
        if 55 < rsi < 70:
            confidence += 10
        elif 30 < rsi < 45:
            confidence += 10
        if signal_data.get('volume', 0) > signal_data.get('volume_ma', 0) * 1.2:
            confidence += 10
        if signal_data.get('ema5', 0) > signal_data.get('ema10', 0) > signal_data.get('ema15', 0):
            confidence += 10
        elif signal_data.get('ema5', 0) < signal_data.get('ema10', 0) < signal_data.get('ema15', 0):
            confidence += 10
        return min(confidence, 90)

# ----- Template MEXCBot class with 24/7 logic -----
class MEXCBot:
    def __init__(self):
        # Example: replace with your symbols/logic
        self.symbols = ['BTCUSDT', 'ETHUSDT']
        self.running = True
        self.signal_output = SignalOutput()
        # Add any other needed initialization

    def check_telegram_updates(self):
        pass  # Fill in as needed

    def fetch_klines(self, symbol, timeframe, limit):
        # Dummy for demo; replace with real implementation
        import pandas as pd
        import numpy as np
        idx = range(limit)
        return pd.DataFrame({
            'close': np.random.uniform(20000, 60000, limit),
            'ema5': np.random.uniform(20000, 60000, limit),
            'ema10': np.random.uniform(20000, 60000, limit),
            'ema15': np.random.uniform(20000, 60000, limit),
            'rsi': np.random.uniform(30, 70, limit),
            'volume': np.random.uniform(10, 100, limit),
            'volume_ma': np.random.uniform(10, 100, limit),
        }, index=idx)

    def check_price_alerts(self, symbol, price):
        return []

    def format_price_alert(self, alert):
        return "Price alert!"

    def check_ema_crossover(self, df):
        # Dummy logic, replace with your strategy
        return {"signal": "LONG", "ema5": df['ema5'].iloc[-1], "ema10": df['ema10'].iloc[-1],
                "ema15": df['ema15'].iloc[-1], "rsi": df['rsi'].iloc[-1], "volume": df['volume'].iloc[-1],
                "volume_ma": df['volume_ma'].iloc[-1], "price": df['close'].iloc[-1]}

    def get_high_low_levels(self, df, n):
        return {"highest": df['close'].max(), "lowest": df['close'].min()}

    def format_trading_info(self, symbol, signal_5m, signal_15m, signal_detected, levels_15m):
        return f"{symbol} {signal_detected} at {signal_5m['price']}"

    def send_telegram_alert(self, message):
        logger.info(f"Telegram alert: {message}")

    def add_price_alert(self, symbol, base_price, mode):
        return True

    def run_single_analysis(self):
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
                # Example rsi threshold, customize as needed
                if signal_15m["rsi"] > 55:
                    signal_detected = "LONG"
            elif signal_5m["signal"] == "SHORT" and signal_15m["signal"] != "LONG":
                if signal_15m["rsi"] < 45:
                    signal_detected = "SHORT"

            if signal_detected:
                message = self.format_trading_info(symbol, signal_5m, signal_15m, signal_detected, levels_15m)
                alerts.append(message)
                self.signal_output.save_signal(symbol, signal_detected, signal_5m)
                logger.info(f"{signal_detected} signal detected for {symbol}")
                base_price = signal_5m['ema15']
                if self.add_price_alert(symbol, base_price, "auto"):
                    logger.info(f"Auto-alert set for {symbol} at base price ${base_price:.4f}")
                logger.info(f"{symbol} 5m - EMA5: {signal_5m['ema5']:.4f}, EMA10: {signal_5m['ema10']:.4f}, RSI: {signal_5m['rsi']:.1f}")
                logger.info(f"{symbol} 15m - EMA5: {signal_15m['ema5']:.4f}, EMA10: {signal_15m['ema10']:.4f}, RSI: {signal_15m['rsi']:.1f}")
                logger.info(f"{symbol} 15m Levels - High: {levels_15m['highest']:.4f}, Low: {levels_15m['lowest']:.4f}")
            else:
                logger.info(f"No clear signal for {symbol}")

        if alerts:
            for alert in alerts:
                self.send_telegram_alert(alert)

        if price_alerts_triggered:
            for price_alert in price_alerts_triggered:
                self.send_telegram_alert(price_alert)

        logger.info(f"Analysis complete. {len(alerts)} signals detected, {len(price_alerts_triggered)} price alerts triggered.")

    def run_forever(self, interval=60):
        """Run the bot in a loop forever, checking signals every `interval` seconds."""
        logger.info(f"🤖 MEXCBot started 24/7 monitoring, interval {interval}s")
        try:
            while True:
                try:
                    self.run_single_analysis()
                except Exception as e:
                    logger.error(f"Exception in analysis: {e}")
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("MEXCBot stopped by user.")

# --- Main execution ---
if __name__ == "__main__":
    bot = MEXCBot()
    bot.run_forever(interval=30)
