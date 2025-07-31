# integrated_bot_analyzer.py

from mexc_bot import MEXCBot  # Your MEXCBot class
from signal_analyzer import SignalAnalyzer  # Your SignalAnalyzer class
import time
import logging

logging.basicConfig(level=logging.INFO)

class IntegratedBot(MEXCBot):
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer  # Direct reference to analyzer

    def run_single_analysis(self):
        logging.info("Running integrated analysis...")
        self.check_telegram_updates()
        if not self.running:
            return

        alerts = []
        price_alerts_triggered = []

        for symbol in self.symbols:
            df_5m = self.fetch_klines(symbol, "5m", 100)
            df_15m = self.fetch_klines(symbol, "15m", 100)
            if df_5m is None or df_15m is None:
                logging.warning(f"Could not fetch data for {symbol}")
                continue

            current_price = float(df_5m.iloc[-1]['close'])

            triggered = self.check_price_alerts(symbol, current_price)
            for alert in triggered:
                alert_message = self.format_price_alert(alert)
                price_alerts_triggered.append(alert_message)
                logging.info(f"Price alert triggered for {symbol} at ${current_price:.4f}")

            signal_5m = self.check_ema_crossover(df_5m)
            signal_15m = self.check_ema_crossover(df_15m)
            levels_15m = self.get_high_low_levels(df_15m, 5)

            signal_detected = None
            if (signal_5m["signal"] == "LONG" and signal_15m["signal"] == "LONG"):
                signal_detected = "LONG"
            elif (signal_5m["signal"] == "SHORT" and signal_15m["signal"] == "SHORT"):
                signal_detected = "SHORT"
            elif signal_5m["signal"] == "LONG" and signal_15m["signal"] != "SHORT":
                if signal_15m["rsi"] > 55:
                    signal_detected = "LONG"
            elif signal_5m["signal"] == "SHORT" and signal_15m["signal"] != "LONG":
                if signal_15m["rsi"] < 45:
                    signal_detected = "SHORT"

            if signal_detected:
                message = self.format_trading_info(symbol, signal_5m, signal_15m, signal_detected, levels_15m)
                alerts.append(message)
                # 🔴 DIRECTLY send signal to analyzer instead of save_signal
                signal_data = signal_5m.copy()
                signal_data['symbol'] = symbol
                signal_data['direction'] = signal_detected
                signal_data['entry_price'] = signal_data['price']
                signal_data['timeframe'] = "5m"
                signal_data['indicators'] = {
                    'ema5': signal_data['ema5'],
                    'ema10': signal_data['ema10'],
                    'ema15': signal_data['ema15'],
                    'rsi': signal_data['rsi'],
                    'volume': signal_data['volume'],
                }
                result = self.analyzer.capture_bot_signal(signal_data)
                logging.info(f"Analyzer result: {result}")

                base_price = signal_5m['ema15']
                if self.add_price_alert(symbol, base_price, "auto"):
                    logging.info(f"Auto-alert set for {symbol} at base price ${base_price:.4f}")
                logging.info(f"{symbol} 5m - EMA5: {signal_5m['ema5']:.4f}, EMA10: {signal_5m['ema10']:.4f}, RSI: {signal_5m['rsi']:.1f}")
                logging.info(f"{symbol} 15m - EMA5: {signal_15m['ema5']:.4f}, EMA10: {signal_15m['ema10']:.4f}, RSI: {signal_15m['rsi']:.1f}")
                logging.info(f"{symbol} 15m Levels - High: {levels_15m['highest']:.4f}, Low: {levels_15m['lowest']:.4f}")
            else:
                logging.info(f"No clear signal for {symbol}")

        if alerts:
            for alert in alerts:
                self.send_telegram_alert(alert)

        if price_alerts_triggered:
            for price_alert in price_alerts_triggered:
                self.send_telegram_alert(price_alert)

        logging.info(f"Analysis complete. {len(alerts)} signals detected, {len(price_alerts_triggered)} price alerts triggered.")

    def run_forever(self, interval=60):
        logging.info(f"🤖 IntegratedBot started 24/7 monitoring, interval {interval}s")
        try:
            while True:
                try:
                    self.run_single_analysis()
                except Exception as e:
                    logging.error(f"Exception in analysis: {e}")
                time.sleep(interval)
        except KeyboardInterrupt:
            logging.info("Bot stopped by user.")

if __name__ == "__main__":
    analyzer = SignalAnalyzer()
    bot = IntegratedBot(analyzer)
    bot.run_forever(interval=30)
