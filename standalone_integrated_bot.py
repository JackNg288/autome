#!/usr/bin/env python3
"""
Standalone Integrated MEXC Bot + Signal Analyzer
All-in-one file for easier deployment
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
from datetime import datetime, timedelta
import threading
from typing import Optional, Dict, Any, List
import sqlite3
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MiniSignalAnalyzer:
    """Simplified Signal Analyzer integrated into bot"""
    
    def __init__(self):
        self.db_file = "signals_database.db"
        self.init_database()
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
    def init_database(self):
        """Initialize SQLite database"""
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
                confidence REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized")
    
    def capture_signal(self, signal_data: Dict) -> str:
        """Capture and store signal"""
        try:
            symbol = signal_data.get('symbol')
            direction = signal_data.get('direction', '').upper()
            entry_price = float(signal_data.get('entry_price', 0))
            
            # Calculate TP/SL
            atr_percent = 2.0  # 2% default
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
                current_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                entry_price
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Signal captured: {signal_id}")
            return f"✅ Signal captured: {symbol} {direction} @ ${entry_price:.4f}"
            
        except Exception as e:
            logger.error(f"Error capturing signal: {e}")
            return f"❌ Error: {str(e)}"
    
    def update_active_signals(self, price_fetcher):
        """Update active signals with current prices"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT signal_id, symbol, direction, entry_price, target_price, stop_loss
            FROM signals WHERE status = 'ACTIVE'
        ''')
        
        active_signals = cursor.fetchall()
        
        for signal in active_signals:
            signal_id, symbol, direction, entry_price, target_price, stop_loss = signal
            
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
            if hit_target:
                new_status = 'WIN'
            elif hit_stop:
                new_status = 'LOSS'
            
            cursor.execute('''
                UPDATE signals 
                SET current_price = ?, pnl_percent = ?, status = ?
                WHERE signal_id = ?
            ''', (current_price, pnl_percent, new_status, signal_id))
            
            if new_status != 'ACTIVE':
                logger.info(f"Signal {signal_id} closed: {new_status}")
        
        conn.commit()
        conn.close()
    
    def get_stats(self) -> Dict:
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
    """All-in-one MEXC Bot with integrated analyzer"""
    
    def __init__(self):
        self.analyzer = MiniSignalAnalyzer()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'StandaloneBot/1.0'})
        
        # Configuration
        self.symbols = self.load_symbols()
        self.base_url = "https://api.mexc.com"
        self.binance_url = "https://api.binance.com"
        
        # Strategy parameters
        self.ema5_period = 5
        self.ema10_period = 10
        self.ema15_period = 15
        self.rsi_period = 14
        self.rsi_long_threshold = 55
        self.rsi_short_threshold = 45
        
        # Telegram
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        self.running = True
        
        logger.info("Standalone Integrated Bot initialized")
    
    def load_symbols(self) -> List[str]:
        """Load symbols from file or use defaults"""
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
    
    def fetch_price(self, symbol: str) -> Optional[float]:
        """Fetch current price"""
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
    
    def fetch_klines(self, symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch kline data"""
        try:
            url = f"{self.base_url}/api/v3/klines"
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                # Try Binance
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
    
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def check_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for EMA crossover signal"""
        if df is None or len(df) < 50:
            return {"signal": None}
        
        try:
            # Calculate indicators
            df["ema5"] = df["close"].ewm(span=self.ema5_period, adjust=False).mean()
            df["ema10"] = df["close"].ewm(span=self.ema10_period, adjust=False).mean()
            df["ema15"] = df["close"].ewm(span=self.ema15_period, adjust=False).mean()
            df["rsi"] = self.calculate_rsi(df["close"])
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Check crossovers
            bullish_cross = (latest["ema5"] > latest["ema10"]) and (prev["ema5"] <= prev["ema10"])
            bearish_cross = (latest["ema5"] < latest["ema10"]) and (prev["ema5"] >= prev["ema10"])
            
            signal = None
            if bullish_cross and latest["rsi"] > self.rsi_long_threshold:
                signal = "LONG"
            elif bearish_cross and latest["rsi"] < self.rsi_short_threshold:
                signal = "SHORT"
            
            return {
                "signal": signal,
                "price": float(latest["close"]),
                "ema5": float(latest["ema5"]),
                "ema10": float(latest["ema10"]),
                "ema15": float(latest["ema15"]),
                "rsi": float(latest["rsi"]),
                "volume": float(latest["volume"])
            }
            
        except Exception as e:
            logger.error(f"Signal calculation error: {e}")
            return {"signal": None}
    
    def send_telegram(self, message: str):
        """Send Telegram message"""
        if not self.telegram_token or not self.chat_id:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            requests.post(url, data=payload, timeout=10)
        except Exception as e:
            logger.error(f"Telegram error: {e}")
    
    def run_analysis(self):
        """Run single analysis cycle"""
        logger.info("Running analysis...")
        
        for symbol in self.symbols:
            try:
                # Fetch data
                df_5m = self.fetch_klines(symbol, "5m")
                df_15m = self.fetch_klines(symbol, "15m")
                
                if df_5m is None or df_15m is None:
                    continue
                
                # Check signals
                signal_5m = self.check_signal(df_5m)
                signal_15m = self.check_signal(df_15m)
                
                # Determine final signal
                signal = None
                if signal_5m["signal"] == "LONG" and signal_15m["signal"] == "LONG":
                    signal = "LONG"
                elif signal_5m["signal"] == "SHORT" and signal_15m["signal"] == "SHORT":
                    signal = "SHORT"
                
                if signal:
                    # Capture signal
                    signal_data = {
                        'symbol': symbol,
                        'direction': signal,
                        'entry_price': signal_5m['price'],
                        'confidence': 70
                    }
                    
                    result = self.analyzer.capture_signal(signal_data)
                    logger.info(result)
                    
                    # Send Telegram notification
                    message = (
                        f"{'🟢' if signal == 'LONG' else '🔴'} *{signal} SIGNAL: {symbol}*\n"
                        f"💰 Price: ${signal_5m['price']:.4f}\n"
                        f"📊 RSI: {signal_5m['rsi']:.1f}\n"
                        f"📈 Signal captured for tracking"
                    )
                    self.send_telegram(message)
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        # Update active signals
        self.analyzer.update_active_signals(self.fetch_price)
        
        # Log stats
        stats = self.analyzer.get_stats()
        logger.info(f"Stats - Total: {stats['total']}, Active: {stats['active']}, Win Rate: {stats['win_rate']:.1f}%")
    
    def run(self):
        """Main run loop"""
        self.send_telegram("🤖 *Standalone Integrated Bot Started*\n\nMonitoring signals and tracking performance...")
        
        while self.running:
            try:
                self.run_analysis()
                time.sleep(30)  # 30 second intervals
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)
        
        # Final stats
        stats = self.analyzer.get_stats()
        self.send_telegram(
            f"🛑 *Bot Stopped*\n\n"
            f"Total Signals: {stats['total']}\n"
            f"Win Rate: {stats['win_rate']:.1f}%"
        )


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════╗
    ║    STANDALONE INTEGRATED BOT SYSTEM       ║
    ╚═══════════════════════════════════════════╝
    
    Starting bot with integrated signal analyzer...
    """)
    
    bot = StandaloneIntegratedBot()
    bot.run()
