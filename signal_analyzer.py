#!/usr/bin/env python3
"""
Signal Analysis System - Professional Win Rate Calculator
Captures signals from MEXC bot and manual inputs, analyzes performance
Features:
- Capture signals from bot via file/webhook
- Manual signal input via Telegram
- Win rate calculation with multiple metrics
- Performance tracking per symbol
- Risk/reward analysis
- Time-based performance metrics
- Export reports
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
from typing import Optional, Dict, Any, List, Tuple
import threading
from collections import defaultdict
import sqlite3
from urllib.parse import urlencode

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('signal_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SignalAnalyzer:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'SignalAnalyzer/1.0'})
        
        # Database for signal storage
        self.db_file = "signals_database.db"
        self.init_database()
        
        # Signal sources
        self.signal_file = "active_signals.json"
        self.manual_signals_file = "manual_signals.json"
        
        # Analysis parameters
        self.target_multipliers = {
            'conservative': {'tp': 1.5, 'sl': 1.0},  # 1.5:1 RR
            'balanced': {'tp': 2.0, 'sl': 1.0},      # 2:1 RR
            'aggressive': {'tp': 3.0, 'sl': 1.0}     # 3:1 RR
        }
        self.default_risk_percent = 1.0  # 1% risk per trade
        self.analysis_timeframes = [5, 15, 30, 60, 240, 1440]  # minutes
        
        # API endpoints
        self.base_url = "https://api.mexc.com"
        self.binance_url = "https://api.binance.com"
        
        # Telegram
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.last_update_id = self.load_last_update_id()
        
        # Performance tracking
        self.performance_cache = {}
        self.symbol_performance = defaultdict(lambda: {
            'total': 0, 'wins': 0, 'losses': 0, 'pending': 0,
            'total_pnl': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0
        })
        
        # Analysis settings
        self.min_signals_for_stats = 5
        self.confidence_thresholds = {
            'very_high': 80,
            'high': 70,
            'medium': 60,
            'low': 50
        }
        
        logger.info("Signal Analyzer initialized")

    def init_database(self):
        """Initialize SQLite database for signal storage"""
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
                pnl_amount REAL,
                source TEXT,
                confidence REAL,
                timeframe TEXT,
                indicators TEXT,
                close_time DATETIME,
                close_price REAL,
                max_profit REAL,
                max_loss REAL,
                hold_time INTEGER,
                notes TEXT
            )
        ''')
    
        # FIXED: split each CREATE INDEX into its own execute!
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON signals(symbol);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON signals(timestamp);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON signals(status);')
    
        conn.commit()
        conn.close()
        logger.info("Database initialized")

    def load_last_update_id(self) -> int:
        """Load last Telegram update ID"""
        try:
            if os.path.exists("analyzer_last_update.txt"):
                with open("analyzer_last_update.txt", 'r') as f:
                    return int(f.read().strip())
        except:
            pass
        return 0

    def save_last_update_id(self, update_id: int):
        """Save last Telegram update ID"""
        try:
            with open("analyzer_last_update.txt", 'w') as f:
                f.write(str(update_id))
        except Exception as e:
            logger.error(f"Error saving update ID: {e}")

    def fetch_price(self, symbol: str) -> Optional[float]:
        """Fetch current price with multiple API fallback"""
        # Try MEXC first
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            params = {"symbol": symbol}
            response = self.session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data.get('price', 0))
        except:
            pass
        
        # Try Binance as fallback
        try:
            url = f"{self.binance_url}/api/v3/ticker/price"
            params = {"symbol": symbol}
            response = self.session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data.get('price', 0))
        except:
            pass
        
        logger.error(f"Failed to fetch price for {symbol}")
        return None

    def capture_bot_signal(self, signal_data: Dict) -> str:
        """Capture signal from bot and store in database"""
        try:
            symbol = signal_data.get('symbol')
            direction = signal_data.get('direction', '').upper()
            entry_price = float(signal_data.get('entry_price', 0))
            confidence = float(signal_data.get('confidence', 50))
            
            if not all([symbol, direction, entry_price]):
                return "Invalid signal data"
            
            # Calculate TP/SL based on ATR or fixed percentage
            atr_percent = signal_data.get('atr_percent', 2.0)  # Default 2%
            
            if direction == "LONG":
                stop_loss = entry_price * (1 - atr_percent / 100)
                target_price = entry_price * (1 + atr_percent * 2 / 100)  # 2:1 RR
            else:
                stop_loss = entry_price * (1 + atr_percent / 100)
                target_price = entry_price * (1 - atr_percent * 2 / 100)
            
            signal_id = f"{symbol}_{direction}_{int(time.time())}"
            
            # Store in database
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO signals (signal_id, timestamp, symbol, direction, 
                entry_price, target_price, stop_loss, status, source, confidence,
                timeframe, indicators, current_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                confidence,
                signal_data.get('timeframe', '5m'),
                json.dumps(signal_data.get('indicators', {})),
                entry_price
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Captured bot signal: {signal_id}")
            return f"✅ Signal captured: {symbol} {direction} @ ${entry_price:.4f}"
            
        except Exception as e:
            logger.error(f"Error capturing bot signal: {e}")
            return f"❌ Error capturing signal: {str(e)}"

    def parse_manual_signal(self, text: str) -> Dict:
        """Parse manual signal input"""
        # Format: LONG/SHORT SYMBOL PRICE [TP] [SL]
        # Example: "LONG BTCUSDT 116000" or "LONG BTCUSDT 116000 118000 115000"
        try:
            parts = text.strip().upper().split()
            if len(parts) < 3:
                return {'error': 'Format: LONG/SHORT SYMBOL PRICE [TP] [SL]'}
            
            direction = parts[0]
            if direction not in ['LONG', 'SHORT']:
                return {'error': 'Direction must be LONG or SHORT'}
            
            symbol = parts[1]
            entry_price = float(parts[2])
            
            # Optional TP and SL
            target_price = float(parts[3]) if len(parts) > 3 else None
            stop_loss = float(parts[4]) if len(parts) > 4 else None
            
            # Auto-calculate TP/SL if not provided
            if not target_price or not stop_loss:
                atr_percent = 2.0  # Default 2% move
                if direction == "LONG":
                    stop_loss = stop_loss or entry_price * (1 - atr_percent / 100)
                    target_price = target_price or entry_price * (1 + atr_percent * 2 / 100)
                else:
                    stop_loss = stop_loss or entry_price * (1 + atr_percent / 100)
                    target_price = target_price or entry_price * (1 - atr_percent * 2 / 100)
            
            return {
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'source': 'MANUAL'
            }
            
        except Exception as e:
            return {'error': f'Parse error: {str(e)}'}

    def add_manual_signal(self, signal_text: str) -> str:
        """Add manual signal from Telegram"""
        parsed = self.parse_manual_signal(signal_text)
        
        if 'error' in parsed:
            return f"❌ {parsed['error']}"
        
        try:
            signal_id = f"{parsed['symbol']}_{parsed['direction']}_{int(time.time())}"
            
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
                parsed['symbol'],
                parsed['direction'],
                parsed['entry_price'],
                parsed['target_price'],
                parsed['stop_loss'],
                'ACTIVE',
                'MANUAL',
                50.0,  # Default confidence for manual signals
                parsed['entry_price']
            ))
            
            conn.commit()
            conn.close()
            
            # Get current price for immediate feedback
            current_price = self.fetch_price(parsed['symbol'])
            price_info = f"Current: ${current_price:.4f}" if current_price else ""
            
            return (
                f"✅ Manual signal added:\n"
                f"📊 {parsed['symbol']} {parsed['direction']}\n"
                f"💰 Entry: ${parsed['entry_price']:.4f}\n"
                f"🎯 Target: ${parsed['target_price']:.4f}\n"
                f"🛡 Stop: ${parsed['stop_loss']:.4f}\n"
                f"{price_info}"
            )
            
        except Exception as e:
            logger.error(f"Error adding manual signal: {e}")
            return f"❌ Error adding signal: {str(e)}"

    def update_active_signals(self):
        """Update all active signals with current prices"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Get all active signals
        cursor.execute('''
            SELECT signal_id, symbol, direction, entry_price, target_price, stop_loss
            FROM signals WHERE status = 'ACTIVE'
        ''')
        
        active_signals = cursor.fetchall()
        updated_count = 0
        
        for signal in active_signals:
            signal_id, symbol, direction, entry_price, target_price, stop_loss = signal
            
            current_price = self.fetch_price(symbol)
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
            
            # Update signal
            new_status = 'ACTIVE'
            close_price = None
            close_time = None
            
            if hit_target:
                new_status = 'WIN'
                close_price = target_price
                close_time = datetime.now()
            elif hit_stop:
                new_status = 'LOSS'
                close_price = stop_loss
                close_time = datetime.now()
            
            # Update database
            cursor.execute('''
                UPDATE signals 
                SET current_price = ?, pnl_percent = ?, status = ?,
                    close_price = ?, close_time = ?
                WHERE signal_id = ?
            ''', (current_price, pnl_percent, new_status, close_price, close_time, signal_id))
            
            updated_count += 1
            
            if new_status != 'ACTIVE':
                logger.info(f"Signal {signal_id} closed: {new_status} @ {close_price:.4f}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated {updated_count} active signals")

    def calculate_win_rate(self, symbol: Optional[str] = None, 
                          days: int = 30, source: Optional[str] = None) -> Dict:
        """Calculate comprehensive win rate statistics"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Build query
        query = '''
            SELECT symbol, direction, status, pnl_percent, source, confidence,
                   timestamp, close_time, entry_price, close_price
            FROM signals 
            WHERE status IN ('WIN', 'LOSS')
            AND timestamp >= ?
        '''
        params = [datetime.now() - timedelta(days=days)]
        
        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
        
        if source:
            query += ' AND source = ?'
            params.append(source)
        
        cursor.execute(query, params)
        signals = cursor.fetchall()
        conn.close()
        
        if not signals:
            return {
                'error': 'No completed signals found',
                'total_signals': 0
            }
        
        # Calculate statistics
        df = pd.DataFrame(signals, columns=[
            'symbol', 'direction', 'status', 'pnl_percent', 'source', 
            'confidence', 'timestamp', 'close_time', 'entry_price', 'close_price'
        ])
        
        total_signals = len(df)
        wins = len(df[df['status'] == 'WIN'])
        losses = len(df[df['status'] == 'LOSS'])
        win_rate = (wins / total_signals) * 100 if total_signals > 0 else 0
        
        # Calculate average PnL
        avg_win_pnl = df[df['status'] == 'WIN']['pnl_percent'].mean() if wins > 0 else 0
        avg_loss_pnl = df[df['status'] == 'LOSS']['pnl_percent'].mean() if losses > 0 else 0
        total_pnl = df['pnl_percent'].sum()
        
        # Calculate profit factor
        total_win_pnl = df[df['status'] == 'WIN']['pnl_percent'].sum() if wins > 0 else 0
        total_loss_pnl = abs(df[df['status'] == 'LOSS']['pnl_percent'].sum()) if losses > 0 else 0
        profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float('inf')
        
        # Calculate expectancy
        expectancy = (win_rate / 100 * avg_win_pnl) - ((100 - win_rate) / 100 * abs(avg_loss_pnl))
        
        # Direction analysis
        long_stats = self._calculate_direction_stats(df[df['direction'] == 'LONG'])
        short_stats = self._calculate_direction_stats(df[df['direction'] == 'SHORT'])
        
        # Symbol performance
        symbol_performance = {}
        for sym in df['symbol'].unique():
            sym_df = df[df['symbol'] == sym]
            symbol_performance[sym] = {
                'total': len(sym_df),
                'win_rate': (len(sym_df[sym_df['status'] == 'WIN']) / len(sym_df)) * 100,
                'avg_pnl': sym_df['pnl_percent'].mean()
            }
        
        # Time-based analysis
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_performance = df.groupby('hour').agg({
            'status': lambda x: (x == 'WIN').sum() / len(x) * 100,
            'pnl_percent': 'mean'
        }).to_dict()
        
        # Confidence correlation
        if 'confidence' in df.columns:
            confidence_bins = pd.cut(df['confidence'], bins=[0, 50, 60, 70, 80, 100])
            confidence_performance = df.groupby(confidence_bins).agg({
                'status': lambda x: (x == 'WIN').sum() / len(x) * 100 if len(x) > 0 else 0
            })
        
        return {
            'period_days': days,
            'total_signals': total_signals,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 2),
            'avg_win_pnl': round(avg_win_pnl, 2),
            'avg_loss_pnl': round(avg_loss_pnl, 2),
            'total_pnl': round(total_pnl, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'N/A',
            'expectancy': round(expectancy, 2),
            'long_performance': long_stats,
            'short_performance': short_stats,
            'symbol_performance': symbol_performance,
            'best_hours': self._get_best_hours(hourly_performance),
            'source_breakdown': {
                'bot': len(df[df['source'] == 'BOT']),
                'manual': len(df[df['source'] == 'MANUAL'])
            }
        }

    def _calculate_direction_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate statistics for a specific direction"""
        if len(df) == 0:
            return {'total': 0, 'win_rate': 0, 'avg_pnl': 0}
        
        wins = len(df[df['status'] == 'WIN'])
        total = len(df)
        
        return {
            'total': total,
            'win_rate': round((wins / total) * 100, 2) if total > 0 else 0,
            'avg_pnl': round(df['pnl_percent'].mean(), 2)
        }

    def _get_best_hours(self, hourly_performance: Dict) -> List[int]:
        """Get best trading hours based on win rate"""
        if not hourly_performance.get('status'):
            return []
        
        sorted_hours = sorted(
            hourly_performance['status'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [hour for hour, win_rate in sorted_hours[:3] if win_rate > 60]

    def generate_performance_report(self, period_days: int = 30) -> str:
        """Generate comprehensive performance report"""
        stats = self.calculate_win_rate(days=period_days)
        
        if stats.get('error'):
            return f"📊 *Performance Report*\n\n{stats['error']}"
        
        report = f"""📊 *Performance Report ({period_days} days)*

📈 *Overall Statistics:*
• Total Signals: {stats['total_signals']}
• Wins: {stats['wins']} | Losses: {stats['losses']}
• Win Rate: {stats['win_rate']}%
• Profit Factor: {stats['profit_factor']}
• Expectancy: {stats['expectancy']}%

💰 *PnL Analysis:*
• Total PnL: {stats['total_pnl']:.2f}%
• Avg Win: +{stats['avg_win_pnl']:.2f}%
• Avg Loss: {stats['avg_loss_pnl']:.2f}%

📊 *Direction Performance:*
🟢 LONG: {stats['long_performance']['win_rate']}% ({stats['long_performance']['total']} trades)
🔴 SHORT: {stats['short_performance']['win_rate']}% ({stats['short_performance']['total']} trades)

🏆 *Top Performing Symbols:*"""

        # Add top 5 symbols
        sorted_symbols = sorted(
            stats['symbol_performance'].items(),
            key=lambda x: (x[1]['win_rate'], x[1]['total']),
            reverse=True
        )[:5]
        
        for symbol, perf in sorted_symbols:
            report += f"\n• {symbol}: {perf['win_rate']:.1f}% ({perf['total']} trades)"
        
        # Add best trading hours if available
        if stats['best_hours']:
            report += f"\n\n⏰ *Best Trading Hours:* {', '.join(map(str, stats['best_hours']))}:00"
        
        # Add source breakdown
        report += f"\n\n📱 *Signal Sources:*"
        report += f"\n• Bot Signals: {stats['source_breakdown']['bot']}"
        report += f"\n• Manual Signals: {stats['source_breakdown']['manual']}"
        
        # Add recommendations
        report += "\n\n💡 *Recommendations:*"
        
        if stats['win_rate'] < 50:
            report += "\n⚠️ Win rate below 50% - Review strategy"
        elif stats['win_rate'] > 65:
            report += "\n✅ Excellent win rate - Maintain discipline"
        
        if stats['profit_factor'] and stats['profit_factor'] != 'N/A' and stats['profit_factor'] < 1.5:
            report += "\n⚠️ Low profit factor - Improve risk/reward ratio"
        
        if abs(stats['avg_loss_pnl']) > stats['avg_win_pnl'] * 1.5:
            report += "\n⚠️ Losses too large - Tighten stop losses"
        
        return report

    def analyze_specific_signal(self, symbol: str, entry_price: float, 
                               direction: str, current_price: Optional[float] = None) -> str:
        """Analyze a specific signal and provide win probability"""
        if not current_price:
            current_price = self.fetch_price(symbol)
            if not current_price:
                return "❌ Unable to fetch current price"
        
        # Get historical performance for this symbol
        symbol_stats = self.calculate_win_rate(symbol=symbol, days=90)
        
        # Calculate current PnL
        if direction.upper() == "LONG":
            current_pnl = ((current_price - entry_price) / entry_price) * 100
        else:
            current_pnl = ((entry_price - current_price) / entry_price) * 100
        
        # Estimate win probability based on historical data
        base_win_rate = symbol_stats.get('win_rate', 50)
        
        # Adjust based on current momentum
        momentum_factor = 1.0
        if current_pnl > 0:
            momentum_factor = 1.1  # Positive momentum
        elif current_pnl < -1:
            momentum_factor = 0.9  # Negative momentum
        
        estimated_win_prob = min(base_win_rate * momentum_factor, 95)
        
        # Generate analysis
        analysis = f"""🔍 *Signal Analysis: {symbol} {direction}*

📊 *Current Status:*
• Entry: ${entry_price:.4f}
• Current: ${current_price:.4f}
• PnL: {current_pnl:+.2f}%

📈 *Historical Performance:*
• Symbol Win Rate: {symbol_stats.get('win_rate', 'N/A')}%
• Total Signals: {symbol_stats.get('total_signals', 0)}
• Avg Win: {symbol_stats.get('avg_win_pnl', 0):.2f}%
• Avg Loss: {symbol_stats.get('avg_loss_pnl', 0):.2f}%

🎯 *Win Probability Estimate: {estimated_win_prob:.1f}%*

💡 *Recommendation:*"""

        if current_pnl > 2:
            analysis += "\n✅ Consider taking partial profits"
        elif current_pnl < -2:
            analysis += "\n⚠️ Review stop loss placement"
        
        if estimated_win_prob > 70:
            analysis += "\n✅ High probability setup"
        elif estimated_win_prob < 40:
            analysis += "\n⚠️ Low probability - manage risk carefully"
        
        return analysis

    def export_signals_csv(self, filename: str = "signals_export.csv") -> str:
        """Export all signals to CSV"""
        try:
            conn = sqlite3.connect(self.db_file)
            df = pd.read_sql_query("SELECT * FROM signals ORDER BY timestamp DESC", conn)
            conn.close()
            
            df.to_csv(filename, index=False)
            return f"✅ Exported {len(df)} signals to {filename}"
        except Exception as e:
            logger.error(f"Error exporting signals: {e}")
            return f"❌ Export failed: {str(e)}"

    def process_telegram_command(self, message_text: str) -> str:
        """Process Telegram commands"""
        try:
            parts = message_text.strip().split()
            if not parts:
                return "Invalid command"
            
            command = parts[0].lower()
            
            # Signal input commands
            if command in ['/long', '/short']:
                signal_text = ' '.join([command[1:].upper()] + parts[1:])
                return self.add_manual_signal(signal_text)
            
            # Analysis commands
            elif command == '/analyze' and len(parts) >= 4:
                # /analyze BTCUSDT LONG 116000
                symbol = parts[1].upper()
                direction = parts[2].upper()
                entry_price = float(parts[3])
                return self.analyze_specific_signal(symbol, entry_price, direction)
            
            elif command == '/performance':
                days = int(parts[1]) if len(parts) > 1 else 30
                return self.generate_performance_report(days)
            
            elif command == '/winrate':
                if len(parts) > 1:
                    symbol = parts[1].upper()
                    stats = self.calculate_win_rate(symbol=symbol)
                else:
                    stats = self.calculate_win_rate()
                
                if stats.get('error'):
                    return f"❌ {stats['error']}"
                
                return f"""📊 *Win Rate Analysis*
Total: {stats['total_signals']} signals
Win Rate: {stats['win_rate']}%
Profit Factor: {stats['profit_factor']}
Expectancy: {stats['expectancy']}%"""
            
            elif command == '/active':
                conn = sqlite3.connect(self.db_file)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT symbol, direction, entry_price, current_price, pnl_percent
                    FROM signals WHERE status = 'ACTIVE'
                    ORDER BY timestamp DESC LIMIT 10
                ''')
                active = cursor.fetchall()
                conn.close()
                
                if not active:
                    return "📭 No active signals"
                
                msg = "📊 *Active Signals:*\n"
                for sig in active:
                    symbol, direction, entry, current, pnl = sig
                    emoji = "🟢" if direction == "LONG" else "🔴"
                    msg += f"\n{emoji} {symbol}: ${entry:.4f} → ${current:.4f} ({pnl:+.2f}%)"
                
                return msg
            
            elif command == '/export':
                return self.export_signals_csv()
            
            elif command == '/help':
                return """📋 *Signal Analyzer Commands:*
*Add Signals:*
/long SYMBOL PRICE [TP] [SL] - Add long signal
/short SYMBOL PRICE [TP] [SL] - Add short signal

*Analysis:*
/analyze SYMBOL DIRECTION PRICE - Analyze specific signal
/performance [DAYS] - Performance report
/winrate [SYMBOL] - Win rate statistics
/active - Show active signals

*Management:*
/export - Export signals to CSV
/clear_old - Clear signals older than 90 days

*Examples:*
/long BTCUSDT 116000
/analyze BTCUSDT LONG 116000
"""
        except Exception as e:
            logger.error(f"Error in process_telegram_command: {e}")
            return f"❌ Error: {str(e)}"
