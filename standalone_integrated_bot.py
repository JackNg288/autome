#!/usr/bin/env python3
"""
Standalone Integrated MEXC Bot + Multi-Strategy Signal Analyzer + Verification
"""

import requests
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
import logging
import time
from datetime import datetime
import sqlite3

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MiniSignalAnalyzer:
    def __init__(self):
        self.db_file = "signals_database.db"
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT UNIQUE,
                timestamp DATETIME,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                current_price REAL,
                target_price REAL,
                stop_loss REAL,
                status TEXT,
                pnl_percent REAL,
                source TEXT,
                confidence REAL,
                reason TEXT
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized")

    def capture_signal(self, signal_data):
        """Capture and store signal"""
        try:
            symbol = signal_data.get('symbol')
            direction = signal_data.get('direction', '').upper()
            entry_price = float(signal_data.get('entry_price', 0))
            reason = signal_data.get('reason', '')

            # Calculate TP/SL (2% SL, 4% TP for now)
            atr_percent = 2.0
            if direction == "LONG":
                stop_loss = entry_price * (1 - atr_percent / 100)
                target_price = entry_price * (1 + atr_percent * 2 / 100)
            else:
                stop_loss = entry_price * (1 + atr_percent / 100)
                target_price = entry_price * (1 - atr_percent * 2 / 100)

            signal_id = f"{symbol}_{direction}_{int(time.time())}"

            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signals (signal_id, timestamp, symbol, direction, 
                entry_price, target_price, stop_loss, status, source, confidence,
                current_price, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_id,
                datetime.now(),
                symbol,
                direction,
                entry_price,
                target_price,
                stop_loss,
                'ACTIVE',
                'BOT',
                signal_data.get('confidence', 70),
                entry_price,
                reason
            ))
            conn.commit()
            conn.close()
            logger.info(f"Signal captured: {signal_id}")
            return f"✅ Signal captured: {symbol} {direction} @ ${entry_price:.4f}"

        except Exception as e:
            logger.error(f"Error capturing signal: {e}")
            return f"❌ Error: {str(e)}"

    def update_active_signals(self, price_fetcher, telegram_sender=None):
        """Update active signals with current prices and send Telegram on close"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT signal_id, symbol, direction, entry_price, target_price, stop_loss, current_price
            FROM signals WHERE status = 'ACTIVE'
        ''')
        active_signals = cursor.fetchall()
        for signal in active_signals:
            signal_id, symbol, direction, entry_price, target_price, stop_loss, prev_price = signal
            current_price = price_fetcher(symbol)
            if not current_price:
                continue
            # Calculate PnL
            if direction == "LONG":
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
                hit_target = current_price >= target_price
                hit_stop = current_price <= stop_loss
            else:
                pnl_percent = ((entry_price - current_price) / entry_price) * 100
                hit_target = current_price <= target_price
                hit_stop = current_price >= stop_loss

            new_status = 'ACTIVE'
            message = None
            if hit_target:
                new_status = 'WIN'
                message = (
                    f"✅ *SIGNAL CLOSED: WIN*\n"
                    f"Symbol: `{symbol}`\n"
                    f"Direction: *{direction}*\n"
                    f"Entry: `${entry_price:.4f}`\n"
                    f"Exit: `${current_price:.4f}`\n"
                    f"Target: `${target_price:.4f}`\n"
                    f"Stop Loss: `${stop_loss:.4f}`\n"
                    f"PnL: `{pnl_percent:.2f}%`"
                )
            elif hit_stop:
                new_status = 'LOSS'
                message = (
                    f"❌ *SIGNAL CLOSED: LOSS*\n"
                    f"Symbol: `{symbol}`\n"
                    f"Direction: *{direction}*\n"
                    f"Entry: `${entry_price:.4f}`\n"
                    f"Exit: `${current_price:.4f}`\n"
                    f"Target: `${target_price:.4f}`\n"
                    f"Stop Loss: `${stop_loss:.4f}`\n"
                    f"PnL: `{pnl_percent:.2f}%`"
                )
            cursor.execute('''
                UPDATE signals 
                SET current_price = ?, pnl_percent = ?, status = ?
                WHERE signal_id = ?
            ''', (current_price, pnl_percent, new_status, signal_id))
            if new_status != 'ACTIVE' and telegram_sender and message:
                try:
                    telegram_sender(message)
                    logger.info(f"Telegram alert sent for {signal_id} ({new_status})")
                except Exception as te:
                    logger.error(f"Error sending telegram for {signal_id}: {te}")
        conn.commit()
        conn.close()

    def get_stats(self):
        """Get basic statistics"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM signals')
        total = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM signals WHERE status = "WIN"')
        wins = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM signals WHERE status = "LOSS"')
        losses = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM signals WHERE status = "ACTIVE"')
        active = cursor.fetchone()[0]
        conn.close()
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        return {
            'total': total,
            'wins': wins,
            'losses': losses,
            'active': active,
            'win_rate': win_rate
        }

class StandaloneIntegratedBot:
    def __init__(self):
        self.analyzer = MiniSignalAnalyzer()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'StandaloneBot/1.0'})
        self.symbols = self.load_symbols()
        self.base_url = "https://api.mexc.com"
        self.binance_url = "https://api.binance.com"
        self.ema5_period = 5
        self.ema10_period = 10
        self.ema15_period = 15
        self.rsi_period = 14
        self.rsi_long_threshold = 55
        self.rsi_short_threshold = 45
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.running = True
        logger.info("Standalone Integrated Bot initialized")

    def load_symbols(self):
        default_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
        if os.path.exists('symbols.txt'):
            try:
                with open('symbols.txt', 'r') as f:
                    symbols = [line.strip().upper() for line in f.readlines() if line.strip()]
                    if symbols:
                        return symbols
            except:
                pass
        return default_symbols

    def fetch_price(self, symbol):
        # Try MEXC
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            response = self.session.get(url, params={"symbol": symbol}, timeout=5)
            if response.status_code == 200:
                return float(response.json().get('price', 0))
        except:
            pass
        # Try Binance
        try:
            url = f"{self.binance_url}/api/v3/ticker/price"
            response = self.session.get(url, params={"symbol": symbol}, timeout=5)
            if response.status_code == 200:
                return float(response.json().get('price', 0))
        except:
            pass
        return None

    def fetch_klines(self, symbol, interval, limit=100):
        try:
            url = f"{self.base_url}/api/v3/klines"
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code != 200:
                url = f"{self.binance_url}/api/v3/klines"
                response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data, columns=[
                        "timestamp", "open", "high", "low", "close", "volume", 
                        "close_time", "quote_volume"
                    ])
                    numeric_cols = ["open", "high", "low", "close", "volume"]
                    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
                    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                    return df.sort_values("timestamp").reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
        return None

    def calculate_rsi(self, prices):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    # --- Verification Methods ---

    def check_confluence(self, signal_list, direction):
        """Return True if more than one strategy gave this signal."""
        count = sum(1 for sig in signal_list if sig['signal'] == direction)
        return count > 1

    def check_higher_tf_trend(self, symbol, direction, tf='1h'):
        """Check higher timeframe trend direction using EMA10/20."""
        df = self.fetch_klines(symbol, tf, limit=30)
        if df is None or len(df) < 25:
            return False
        df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        latest = df.iloc[-1]
        if direction == 'LONG':
            return latest['ema10'] > latest['ema20']
        else:
            return latest['ema10'] < latest['ema20']

    def check_support_resistance(self, df, entry_price, direction, threshold_pct=1.0):
        """Check if entry is too close to support (LONG) or resistance (SHORT)."""
        closes = df['close'].iloc[-30:]
        support = closes.min()
        resistance = closes.max()
        if direction == 'LONG':
            dist = ((entry_price - support) / entry_price) * 100
            return dist > threshold_pct  # 🟢 if far from support
        else:
            dist = ((resistance - entry_price) / entry_price) * 100
            return dist > threshold_pct  # 🟢 if far from resistance

    # --- Signal Detection ---
    def check_signals(self, df):
        signals = []
        if df is None or len(df) < 50:
            return signals
        try:
            # Compute all indicators
            df["ema5"] = df["close"].ewm(span=self.ema5_period, adjust=False).mean()
            df["ema10"] = df["close"].ewm(span=self.ema10_period, adjust=False).mean()
            df["ema15"] = df["close"].ewm(span=self.ema15_period, adjust=False).mean()
            df["rsi"] = self.calculate_rsi(df["close"])
            df["macd"], df["macd_signal"] = self.calculate_macd(df["close"])
            df["bb_mid"] = df["close"].rolling(window=20).mean()
            df["bb_std"] = df["close"].rolling(window=20).std()
            df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
            df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
            df["vol_avg"] = df["volume"].rolling(window=20).mean()

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            # 1. EMA5/10 Crossover + RSI
            bullish_cross = (latest["ema5"] > latest["ema10"]) and (prev["ema5"] <= prev["ema10"])
            bearish_cross = (latest["ema5"] < latest["ema10"]) and (prev["ema5"] >= prev["ema10"])
            if bullish_cross and latest["rsi"] > self.rsi_long_threshold:
                signals.append({
                    "signal": "LONG",
                    "reason": "EMA5/10 bullish crossover & RSI confirmation",
                })
            if bearish_cross and latest["rsi"] < self.rsi_short_threshold:
                signals.append({
                    "signal": "SHORT",
                    "reason": "EMA5/10 bearish crossover & RSI confirmation",
                })

            # 2. MACD Crossover + RSI Filter
            macd_bull = (latest["macd"] > latest["macd_signal"]) and (prev["macd"] <= prev["macd_signal"]) and latest["rsi"] > 50
            macd_bear = (latest["macd"] < latest["macd_signal"]) and (prev["macd"] >= prev["macd_signal"]) and latest["rsi"] < 50
            if macd_bull:
                signals.append({
                    "signal": "LONG",
                    "reason": "MACD bullish crossover & RSI>50"
                })
            if macd_bear:
                signals.append({
                    "signal": "SHORT",
                    "reason": "MACD bearish crossover & RSI<50"
                })

            # 3. RSI Oversold/Overbought Reversal
            rsi_oversold = (prev["rsi"] < 30 and latest["rsi"] >= 30)
            rsi_overbought = (prev["rsi"] > 70 and latest["rsi"] <= 70)
            if rsi_oversold:
                signals.append({
                    "signal": "LONG",
                    "reason": "RSI oversold reversal (crossed up 30)"
                })
            if rsi_overbought:
                signals.append({
                    "signal": "SHORT",
                    "reason": "RSI overbought reversal (crossed down 70)"
                })

            # 4. Bollinger Band Reversal
            bb_long = (prev["close"] < prev["bb_lower"] and latest["close"] > latest["bb_lower"])
            bb_short = (prev["close"] > prev["bb_upper"] and latest["close"] < latest["bb_upper"])
            if bb_long:
                signals.append({
                    "signal": "LONG",
                    "reason": "Bollinger Band lower reversal"
                })
            if bb_short:
                signals.append({
                    "signal": "SHORT",
                    "reason": "Bollinger Band upper reversal"
                })

            # 5. Price/Volume Breakout
            if (latest["close"] > df["close"].iloc[-21:-1].max()) and (latest["volume"] > 2 * latest["vol_avg"]):
                signals.append({
                    "signal": "LONG",
                    "reason": "Price breakout new high with volume spike"
                })
            if (latest["close"] < df["close"].iloc[-21:-1].min()) and (latest["volume"] > 2 * latest["vol_avg"]):
                signals.append({
                    "signal": "SHORT",
                    "reason": "Price breakout new low with volume spike"
                })

            for sig in signals:
                sig.update({
                    "price": float(latest["close"]),
                    "ema5": float(latest["ema5"]),
                    "ema10": float(latest["ema10"]),
                    "ema15": float(latest["ema15"]),
                    "rsi": float(latest["rsi"]),
                    "macd": float(latest["macd"]),
                    "macd_signal": float(latest["macd_signal"]),
                    "bb_upper": float(latest["bb_upper"]),
                    "bb_lower": float(latest["bb_lower"]),
                    "volume": float(latest["volume"]),
                })

        except Exception as e:
            logger.error(f"Signal calculation error: {e}")
        return signals

    def send_telegram(self, message):
        token = self.telegram_token
        chat_id = self.chat_id
        if not token or not chat_id:
            return
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            requests.post(url, data=payload, timeout=10)
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    def run_analysis(self):
        logger.info("Running analysis...")
        for symbol in self.symbols:
            try:
                for tf in ["5m", "15m"]:
                    df = self.fetch_klines(symbol, tf)
                    if df is None:
                        continue
                    signals = self.check_signals(df)
                    for sig in signals:
                        # Avoid duplicate ACTIVE signal of the same type in a row
                        conn = sqlite3.connect(self.analyzer.db_file)
                        cursor = conn.cursor()
                        cursor.execute('''
                            SELECT COUNT(*) FROM signals WHERE symbol=? AND direction=? AND status='ACTIVE'
                        ''', (symbol, sig['signal']))
                        already_active = cursor.fetchone()[0]
                        conn.close()
                        if already_active:
                            continue

                        # --- VERIFICATION FLAGS ---
                        confluence_flag = self.check_confluence(signals, sig['signal'])
                        confluence_str = '🟢' if confluence_flag else '🔴'
                        higher_tf_flag = self.check_higher_tf_trend(symbol, sig['signal'], tf='1h')
                        higher_tf_str = '🟢' if higher_tf_flag else '🔴'
                        sr_flag = self.check_support_resistance(df, sig['price'], sig['signal'])
                        sr_str = '🟢' if sr_flag else '🔴'

                        # Store and notify
                        signal_data = {
                            'symbol': symbol,
                            'direction': sig['signal'],
                            'entry_price': sig['price'],
                            'confidence': 70,
                            'reason': sig.get('reason', '')
                        }
                        self.analyzer.capture_signal(signal_data)
                        # TP/SL lookup
                        conn = sqlite3.connect(self.analyzer.db_file)
                        cursor = conn.cursor()
                        cursor.execute('''
                            SELECT entry_price, target_price, stop_loss 
                            FROM signals 
                            WHERE symbol=? AND direction=?
                            ORDER BY timestamp DESC LIMIT 1
                        ''', (symbol, sig['signal']))
                        row = cursor.fetchone()
                        conn.close()
                        if row:
                            entry, tp, sl = row
                        else:
                            entry, tp, sl = sig['price'], 0, 0
                        message = (
                            f"{'🟢' if sig['signal']=='LONG' else '🔴'} *{sig['signal']} SIGNAL: {symbol} [{tf}]*\n"
                            f"{sig.get('reason','')}\n"
                            f"Entry: `${entry:.4f}`\n"
                            f"Target: `${tp:.4f}`\n"
                            f"Stop Loss: `${sl:.4f}`\n"
                            f"RSI: `{sig['rsi']:.1f}`\n"
                            f"MACD: `{sig['macd']:.4f}` | MACD Signal: `{sig['macd_signal']:.4f}`\n"
                            f"Volume: `{sig['volume']:.2f}`\n"
                            f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n"
                            f"Multi-strategy confluence: {confluence_str}\n"
                            f"Higher timeframe trend: {higher_tf_str}\n"
                            f"S/R-Price action filter: {sr_str}"
                        )
                        self.send_telegram(message)

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

        # Update active signals and notify on close
        self.analyzer.update_active_signals(self.fetch_price, self.send_telegram)
        # Log stats
        stats = self.analyzer.get_stats()
        logger.info(f"Stats - Total: {stats['total']}, Active: {stats['active']}, Win Rate: {stats['win_rate']:.1f}%")

    def send_telegram(self, message):
        token = self.telegram_token
        chat_id = self.chat_id
        if not token or not chat_id:
            return
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            requests.post(url, data=payload, timeout=10)
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    def run(self):
        self.send_telegram("🤖 *Standalone Integrated Bot Started*\n\nMonitoring signals and tracking performance...")
        while self.running:
            try:
                self.run_analysis()
                time.sleep(30)  # 30 seconds
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)
        stats = self.analyzer.get_stats()
        self.send_telegram(
            f"🛑 *Bot Stopped*\n\n"
            f"Total Signals: {stats['total']}\n"
            f"Win Rate: {stats['win_rate']:.1f}%"
        )

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════╗
    ║    STANDALONE INTEGRATED BOT SYSTEM      ║
    ╚═══════════════════════════════════════════╝

    Starting bot with integrated signal analyzer...
    """)
    bot = StandaloneIntegratedBot()
    bot.run()
