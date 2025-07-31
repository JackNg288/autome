#!/usr/bin/env python3
"""
Verbose Standalone Bot - Shows detailed analysis results
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
import sqlite3
from typing import Optional, Dict, Any, List

# Setup logging to show everything
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('verbose_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SignalDatabase:
    """Simple signal storage"""
    
    def __init__(self):
        self.db_file = "signals_database.db"
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                rsi REAL,
                status TEXT
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized")
    
    def save_signal(self, symbol, direction, price, rsi):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO signals (timestamp, symbol, direction, entry_price, rsi, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now(), symbol, direction, price, rsi, 'ACTIVE'))
        conn.commit()
        conn.close()
        logger.info(f"💾 Signal saved to database: {symbol} {direction} @ ${price:.4f}")


class VerboseBot:
    """Bot that shows everything that's happening"""
    
    def __init__(self):
        self.db = SignalDatabase()
        self.session = requests.Session()
        
        # Load symbols
        self.symbols = self.load_symbols()
        print(f"\n🎯 SYMBOLS TO MONITOR: {self.symbols}\n")
        
        # Strategy settings
        self.ema5_period = 5
        self.ema10_period = 10
        self.ema15_period = 15
        self.rsi_period = 14
        self.rsi_long_threshold = 55
        self.rsi_short_threshold = 45
        
        # Telegram
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        # Statistics
        self.cycles_run = 0
        self.signals_found = 0
        
        print("=" * 60)
        print("VERBOSE BOT INITIALIZED")
        print(f"Strategy: EMA5/10 Crossover + RSI Filter")
        print(f"RSI Thresholds: Long > {self.rsi_long_threshold}, Short < {self.rsi_short_threshold}")
        print("=" * 60)
        print()
    
    def load_symbols(self):
        """Load symbols with verbose output"""
        default_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        if os.path.exists('symbols.txt'):
            try:
                with open('symbols.txt', 'r') as f:
                    symbols = [line.strip().upper() for line in f.readlines() if line.strip()]
                    if symbols:
                        print(f"📄 Loaded {len(symbols)} symbols from symbols.txt")
                        return symbols[:5]  # Limit to 5 for faster analysis
            except Exception as e:
                print(f"⚠️ Error loading symbols: {e}")
        
        print(f"📌 Using default symbols: {default_symbols}")
        return default_symbols
    
    def fetch_klines(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch klines with detailed output"""
        print(f"  📡 Fetching {interval} data for {symbol}...", end="")
        
        # Try MEXC first
        try:
            url = "https://api.mexc.com/api/v3/klines"
            params = {"symbol": symbol, "interval": interval, "limit": 100}
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
                    
                    print(f" ✅ Success! ({len(df)} candles)")
                    return df
        except:
            pass
        
        # Try Binance
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": symbol, "interval": interval, "limit": 100}
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
                    
                    print(f" ✅ Success from Binance! ({len(df)} candles)")
                    return df
        except:
            pass
        
        print(f" ❌ Failed!")
        return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators with output"""
        # EMAs
        df["ema5"] = df["close"].ewm(span=self.ema5_period, adjust=False).mean()
        df["ema10"] = df["close"].ewm(span=self.ema10_period, adjust=False).mean()
        df["ema15"] = df["close"].ewm(span=self.ema15_period, adjust=False).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss.replace(0, 0.0001)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi"] = df["rsi"].fillna(50)
        
        return df
    
    def check_for_signal(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Check for signals and show the analysis"""
        if df is None or len(df) < 20:
            return {"signal": None, "reason": "Not enough data"}
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Show current values
        print(f"    📊 {timeframe} Analysis:")
        print(f"       Price: ${latest['close']:.2f}")
        print(f"       EMA5:  ${latest['ema5']:.2f}")
        print(f"       EMA10: ${latest['ema10']:.2f}")
        print(f"       RSI:   {latest['rsi']:.1f}")
        
        # Check for crossover
        ema5_above = latest['ema5'] > latest['ema10']
        ema5_was_below = prev['ema5'] <= prev['ema10']
        ema5_below = latest['ema5'] < latest['ema10']
        ema5_was_above = prev['ema5'] >= prev['ema10']
        
        bullish_cross = ema5_above and ema5_was_below
        bearish_cross = ema5_below and ema5_was_above
        
        # Check signal conditions
        signal = None
        reason = ""
        
        if bullish_cross:
            print(f"       ✨ BULLISH CROSSOVER DETECTED!")
            if latest['rsi'] > self.rsi_long_threshold:
                signal = "LONG"
                reason = f"Bullish cross + RSI {latest['rsi']:.1f} > {self.rsi_long_threshold}"
                print(f"       🟢 LONG SIGNAL CONFIRMED!")
            else:
                reason = f"Bullish cross but RSI {latest['rsi']:.1f} ≤ {self.rsi_long_threshold}"
                print(f"       ⚠️ RSI too low for LONG signal")
        elif bearish_cross:
            print(f"       ✨ BEARISH CROSSOVER DETECTED!")
            if latest['rsi'] < self.rsi_short_threshold:
                signal = "SHORT"
                reason = f"Bearish cross + RSI {latest['rsi']:.1f} < {self.rsi_short_threshold}"
                print(f"       🔴 SHORT SIGNAL CONFIRMED!")
            else:
                reason = f"Bearish cross but RSI {latest['rsi']:.1f} ≥ {self.rsi_short_threshold}"
                print(f"       ⚠️ RSI too high for SHORT signal")
        else:
            if ema5_above:
                reason = "EMA5 above EMA10 (no crossover)"
            else:
                reason = "EMA5 below EMA10 (no crossover)"
            print(f"       → {reason}")
        
        return {
            "signal": signal,
            "price": float(latest['close']),
            "ema5": float(latest['ema5']),
            "ema10": float(latest['ema10']),
            "ema15": float(latest['ema15']),
            "rsi": float(latest['rsi']),
            "reason": reason
        }
    
    def send_telegram(self, message: str):
        """Send telegram message"""
        if not self.telegram_token or not self.chat_id:
            print("  ⚠️ Telegram not configured")
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                print("  ✅ Telegram notification sent!")
            else:
                print(f"  ❌ Telegram error: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Telegram error: {e}")
    
    def analyze_symbol(self, symbol: str):
        """Analyze one symbol with full output"""
        print(f"\n🔍 ANALYZING {symbol}")
        print("-" * 40)
        
        # Fetch data
        df_5m = self.fetch_klines(symbol, "5m")
        df_15m = self.fetch_klines(symbol, "15m")
        
        if df_5m is None or df_15m is None:
            print(f"  ❌ Failed to fetch data for {symbol}")
            return
        
        # Check signals
        signal_5m = self.check_for_signal(df_5m, "5M")
        signal_15m = self.check_for_signal(df_15m, "15M")
        
        # Determine final signal
        print(f"\n  📋 SIGNAL DECISION:")
        
        final_signal = None
        if signal_5m['signal'] == 'LONG' and signal_15m['signal'] == 'LONG':
            final_signal = 'LONG'
            print(f"  🟢🟢 STRONG LONG SIGNAL (both timeframes agree)!")
        elif signal_5m['signal'] == 'SHORT' and signal_15m['signal'] == 'SHORT':
            final_signal = 'SHORT'
            print(f"  🔴🔴 STRONG SHORT SIGNAL (both timeframes agree)!")
        elif signal_5m['signal'] == 'LONG' and signal_15m['rsi'] > self.rsi_long_threshold:
            final_signal = 'LONG'
            print(f"  🟢 LONG SIGNAL (5m signal + 15m RSI confirmation)!")
        elif signal_5m['signal'] == 'SHORT' and signal_15m['rsi'] < self.rsi_short_threshold:
            final_signal = 'SHORT'
            print(f"  🔴 SHORT SIGNAL (5m signal + 15m RSI confirmation)!")
        else:
            print(f"  → No signal: Timeframes don't agree")
            print(f"    5M: {signal_5m['signal'] or 'None'}")
            print(f"    15M: {signal_15m['signal'] or 'None'}")
        
        # If signal found, save and notify
        if final_signal:
            self.signals_found += 1
            
            # Save to database
            self.db.save_signal(symbol, final_signal, signal_5m['price'], signal_5m['rsi'])
            
            # Send Telegram
            message = (
                f"{'🟢' if final_signal == 'LONG' else '🔴'} *{final_signal} SIGNAL: {symbol}*\n\n"
                f"💰 Price: ${signal_5m['price']:.4f}\n"
                f"📊 5M RSI: {signal_5m['rsi']:.1f}\n"
                f"📊 15M RSI: {signal_15m['rsi']:.1f}\n"
                f"⏰ {datetime.now().strftime('%H:%M:%S')}"
            )
            self.send_telegram(message)
            
            print(f"\n  🎉 SIGNAL CAPTURED AND SAVED!")
    
    def run_analysis_cycle(self):
        """Run one complete analysis cycle"""
        self.cycles_run += 1
        
        print("\n" + "=" * 60)
        print(f"🔄 ANALYSIS CYCLE #{self.cycles_run} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        for symbol in self.symbols:
            self.analyze_symbol(symbol)
            time.sleep(1)  # Small delay between symbols
        
        print("\n" + "-" * 60)
        print(f"✅ CYCLE COMPLETE - Signals found in this session: {self.signals_found}")
        print("-" * 60)
    
    def run(self):
        """Main run loop"""
        print("\n🚀 STARTING VERBOSE BOT\n")
        
        # Send startup message
        self.send_telegram(
            "🤖 *Verbose Bot Started*\n\n"
            f"Monitoring: {', '.join(self.symbols)}\n"
            "I will show you detailed analysis!"
        )
        
        # Run first analysis immediately
        self.run_analysis_cycle()
        
        # Continue running
        while True:
            try:
                print(f"\n⏳ Waiting 30 seconds until next analysis...\n")
                time.sleep(30)
                self.run_analysis_cycle()
            except KeyboardInterrupt:
                print("\n\n🛑 Bot stopped by user")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Waiting 60 seconds before retry...")
                time.sleep(60)
        
        # Final message
        print(f"\n📊 FINAL STATISTICS:")
        print(f"  • Cycles run: {self.cycles_run}")
        print(f"  • Signals found: {self.signals_found}")
        
        self.send_telegram(
            f"🛑 *Bot Stopped*\n\n"
            f"Cycles: {self.cycles_run}\n"
            f"Signals: {self.signals_found}"
        )


if __name__ == "__main__":
    bot = VerboseBot()
    bot.run()
