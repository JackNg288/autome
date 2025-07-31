#!/usr/bin/env python3
"""
Enhanced Standalone Integrated MEXC Bot + Signal Analyzer
With improved debugging and signal detection
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

# Setup enhanced logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more info
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_bot.log'),
        logging.StreamHandler()
    ]
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
            
            logger.info(f"✅ Signal captured: {signal_id}")
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
        updated = 0
        
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
                logger.info(f"🎯 Signal {signal_id} hit target!")
            elif hit_stop:
                new_status = 'LOSS'
                logger.info(f"🛑 Signal {signal_id} hit stop loss")
            
            cursor.execute('''
                UPDATE signals 
                SET current_price = ?, pnl_percent = ?, status = ?
                WHERE signal_id = ?
            ''', (current_price, pnl_percent, new_status, signal_id))
            
            updated += 1
        
        conn.commit()
        conn.close()
        
        if updated > 0:
            logger.debug(f"Updated {updated} active signals")
    
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


class EnhancedStandaloneBot:
    """Enhanced MEXC Bot with better signal detection"""
    
    def __init__(self):
        self.analyzer = MiniSignalAnalyzer()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'EnhancedBot/1.0'})
        
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
        
        # Tracking
        self.last_signals = {}  # Track last signal time per symbol
        self.signal_cooldown = 300  # 5 minutes between signals per symbol
        
        self.running = True
        
        logger.info(f"Enhanced Bot initialized with {len(self.symbols)} symbols")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
    
    def load_symbols(self) -> List[str]:
        """Load symbols from file or use defaults"""
        default_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
        
        if os.path.exists('symbols.txt'):
            try:
                with open('symbols.txt', 'r') as f:
                    symbols = [line.strip().upper() for line in f.readlines() if line.strip()]
                    if symbols:
                        logger.info(f"Loaded {len(symbols)} symbols from file")
                        return symbols
            except Exception as e:
                logger.error(f"Error loading symbols: {e}")
        
        logger.info("Using default symbols")
        return default_symbols
    
    def fetch_price(self, symbol: str) -> Optional[float]:
        """Fetch current price"""
        # Try MEXC
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            response = self.session.get(url, params={"symbol": symbol}, timeout=5)
            if response.status_code == 200:
                price = float(response.json().get('price', 0))
                logger.debug(f"MEXC price for {symbol}: ${price}")
                return price
        except Exception as e:
            logger.debug(f"MEXC price fetch failed for {symbol}: {e}")
        
        # Try Binance
        try:
            url = f"{self.binance_url}/api/v3/ticker/price"
            response = self.session.get(url, params={"symbol": symbol}, timeout=5)
            if response.status_code == 200:
                price = float(response.json().get('price', 0))
                logger.debug(f"Binance price for {symbol}: ${price}")
                return price
        except Exception as e:
            logger.debug(f"Binance price fetch failed for {symbol}: {e}")
        
        logger.warning(f"Failed to fetch price for {symbol}")
        return None
    
    def fetch_klines(self, symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch kline data with enhanced error handling"""
        apis = [
            (self.base_url, "MEXC"),
            (self.binance_url, "Binance")
        ]
        
        for base_url, api_name in apis:
            try:
                url = f"{base_url}/api/v3/klines"
                params = {"symbol": symbol, "interval": interval, "limit": limit}
                response = self.session.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        df = pd.DataFrame(data, columns=[
                            "timestamp", "open", "high", "low", "close", "volume", 
                            "close_time", "quote_volume"
                        ])
                        
                        numeric_cols = ["open", "high", "low", "close", "volume"]
                        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
                        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                        
                        logger.debug(f"✅ {api_name} klines for {symbol} {interval}: {len(df)} candles")
                        return df.sort_values("timestamp").reset_index(drop=True)
                else:
                    logger.debug(f"{api_name} returned {response.status_code} for {symbol}")
                    
            except Exception as e:
                logger.debug(f"{api_name} klines error for {symbol}: {e}")
        
        logger.warning(f"Failed to fetch klines for {symbol} {interval}")
        return None
    
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def check_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for EMA crossover signal with debug info"""
        if df is None or len(df) < 50:
            return {"signal": None, "reason": "Insufficient data"}
        
        try:
            # Calculate indicators
            df["ema5"] = df["close"].ewm(span=self.ema5_period, adjust=False).mean()
            df["ema10"] = df["close"].ewm(span=self.ema10_period, adjust=False).mean()
            df["ema15"] = df["close"].ewm(span=self.ema15_period, adjust=False).mean()
            df["rsi"] = self.calculate_rsi(df["close"])
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Debug info
            logger.debug(f"Latest: EMA5={latest['ema5']:.2f}, EMA10={latest['ema10']:.2f}, RSI={latest['rsi']:.1f}")
            logger.debug(f"Previous: EMA5={prev['ema5']:.2f}, EMA10={prev['ema10']:.2f}")
            
            # Check crossovers
            bullish_cross = (latest["ema5"] > latest["ema10"]) and (prev["ema5"] <= prev["ema10"])
            bearish_cross = (latest["ema5"] < latest["ema10"]) and (prev["ema5"] >= prev["ema10"])
            
            # RSI conditions
            rsi_long_ok = latest["rsi"] > self.rsi_long_threshold
            rsi_short_ok = latest["rsi"] < self.rsi_short_threshold
            
            signal = None
            reason = "No signal"
            
            if bullish_cross:
                if rsi_long_ok:
                    signal = "LONG"
                    reason = f"Bullish cross + RSI {latest['rsi']:.1f} > {self.rsi_long_threshold}"
                else:
                    reason = f"Bullish cross but RSI {latest['rsi']:.1f} too low"
            elif bearish_cross:
                if rsi_short_ok:
                    signal = "SHORT"
                    reason = f"Bearish cross + RSI {latest['rsi']:.1f} < {self.rsi_short_threshold}"
                else:
                    reason = f"Bearish cross but RSI {latest['rsi']:.1f} too high"
            
            logger.debug(f"Signal check: {reason}")
            
            return {
                "signal": signal,
                "price": float(latest["close"]),
                "ema5": float(latest["ema5"]),
                "ema10": float(latest["ema10"]),
                "ema15": float(latest["ema15"]),
                "rsi": float(latest["rsi"]),
                "volume": float(latest["volume"]),
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"Signal calculation error: {e}")
            return {"signal": None, "reason": f"Error: {str(e)}"}
    
    def send_telegram(self, message: str):
        """Send Telegram message"""
        if not self.telegram_token or not self.chat_id:
            logger.warning("Telegram credentials not set")
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            response = requests.post(url, data=payload, timeout=10)
            if response.status_code == 200:
                logger.info("✅ Telegram message sent")
            else:
                logger.error(f"Telegram error: {response.status_code}")
        except Exception as e:
            logger.error(f"Telegram error: {e}")
    
    def check_cooldown(self, symbol: str) -> bool:
        """Check if enough time has passed since last signal"""
        if symbol not in self.last_signals:
            return True
        
        time_since_last = time.time() - self.last_signals[symbol]
        return time_since_last >= self.signal_cooldown
    
    def run_analysis(self):
        """Run single analysis cycle with enhanced debugging"""
        logger.info("🔍 Running analysis cycle...")
        signals_checked = 0
        
        for symbol in self.symbols:
            try:
                # Skip if in cooldown
                if not self.check_cooldown(symbol):
                    logger.debug(f"⏳ {symbol} in cooldown period")
                    continue
                
                # Fetch data
                logger.debug(f"Analyzing {symbol}...")
                df_5m = self.fetch_klines(symbol, "5m", 100)
                df_15m = self.fetch_klines(symbol, "15m", 100)
                
                if df_5m is None or df_15m is None:
                    logger.warning(f"❌ No data for {symbol}")
                    continue
                
                # Check signals
                signal_5m = self.check_signal(df_5m)
                signal_15m = self.check_signal(df_15m)
                
                logger.debug(f"{symbol} 5m signal: {signal_5m.get('signal')} - {signal_5m.get('reason')}")
                logger.debug(f"{symbol} 15m signal: {signal_15m.get('signal')} - {signal_15m.get('reason')}")
                
                # Determine final signal (both timeframes must agree)
                signal = None
                if signal_5m["signal"] == "LONG" and signal_15m["signal"] == "LONG":
                    signal = "LONG"
                    logger.info(f"🟢 LONG signal confirmed for {symbol}!")
                elif signal_5m["signal"] == "SHORT" and signal_15m["signal"] == "SHORT":
                    signal = "SHORT"
                    logger.info(f"🔴 SHORT signal confirmed for {symbol}!")
                elif signal_5m["signal"] == "LONG" and signal_15m.get("rsi", 50) > self.rsi_long_threshold:
                    # Relaxed condition: 5m signal + 15m RSI confirmation
                    signal = "LONG"
                    logger.info(f"🟢 LONG signal (5m + RSI) for {symbol}!")
                elif signal_5m["signal"] == "SHORT" and signal_15m.get("rsi", 50) < self.rsi_short_threshold:
                    # Relaxed condition: 5m signal + 15m RSI confirmation
                    signal = "SHORT"
                    logger.info(f"🔴 SHORT signal (5m + RSI) for {symbol}!")
                
                if signal:
                    # Capture signal
                    signal_data = {
                        'symbol': symbol,
                        'direction': signal,
                        'entry_price': signal_5m['price'],
                        'confidence': 70 if signal_5m["signal"] == signal_15m["signal"] else 60
                    }
                    
                    result = self.analyzer.capture_signal(signal_data)
                    logger.info(result)
                    
                    # Update cooldown
                    self.last_signals[symbol] = time.time()
                    
                    # Send Telegram notification
                    message = (
                        f"{'🟢' if signal == 'LONG' else '🔴'} *{signal} SIGNAL: {symbol}*\n\n"
                        f"💰 Entry Price: ${signal_5m['price']:.4f}\n"
                        f"📊 Indicators:\n"
                        f"• EMA5: ${signal_5m['ema5']:.4f}\n"
                        f"• EMA10: ${signal_5m['ema10']:.4f}\n"
                        f"• RSI: {signal_5m['rsi']:.1f}\n\n"
                        f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}\n"
                        f"📈 Signal captured for tracking"
                    )
                    self.send_telegram(message)
                
                signals_checked += 1
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        # Update active signals
        self.analyzer.update_active_signals(self.fetch_price)
        
        # Log stats
        stats = self.analyzer.get_stats()
        logger.info(
            f"📊 Stats - Checked: {signals_checked}, "
            f"Total: {stats['total']}, Active: {stats['active']}, "
            f"Wins: {stats['wins']}, Losses: {stats['losses']}, "
            f"Win Rate: {stats['win_rate']:.1f}%"
        )
    
    def run(self):
        """Main run loop"""
        startup_msg = (
            "🤖 *Enhanced Standalone Bot Started*\n\n"
            f"📊 Monitoring {len(self.symbols)} symbols\n"
            f"⚙️ Strategy: EMA5/10 crossover + RSI\n"
            f"🎯 RSI: Long >{self.rsi_long_threshold}, Short <{self.rsi_short_threshold}\n"
            f"⏱ Scan interval: 30s\n\n"
            "Signals are captured and tracked automatically!"
        )
        self.send_telegram(startup_msg)
        
        # Run test analysis
        logger.info("Running initial test...")
        self.run_analysis()
        
        while self.running:
            try:
                time.sleep(30)  # Wait before next cycle
                self.run_analysis()
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)
        
        # Final stats
        stats = self.analyzer.get_stats()
        final_msg = (
            f"🛑 *Bot Stopped*\n\n"
            f"📊 Final Statistics:\n"
            f"• Total Signals: {stats['total']}\n"
            f"• Wins: {stats['wins']}\n"
            f"• Losses: {stats['losses']}\n"
            f"• Win Rate: {stats['win_rate']:.1f}%"
        )
        self.send_telegram(final_msg)


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║    ENHANCED STANDALONE BOT WITH SIGNAL ANALYZER       ║
    ╚═══════════════════════════════════════════════════════╝
    
    Features:
    ✅ Enhanced debugging and logging
    ✅ Relaxed signal conditions
    ✅ API fallback (MEXC → Binance)
    ✅ Signal cooldown to prevent spam
    ✅ Detailed signal analysis
    
    Starting enhanced bot...
    """)
    
    bot = EnhancedStandaloneBot()
    bot.run()
