#!/usr/bin/env python3
"""
Integration Setup Guide for MEXC Bot + Signal Analyzer
This script helps you integrate the signal analyzer with your existing bot
"""

import os
import shutil
import json

def create_integration_files():
    """Create necessary integration files"""
    
    # 1. Create signal output directory
    os.makedirs('signal_output', exist_ok=True)
    
    # 2. Create initial signal files
    for filename in ['active_signals.json', 'manual_signals.json']:
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                json.dump([], f)
    
    # 3. Create modified mexc_bot.py with integration
    integration_code = '''
# Add this to your mexc_bot.py after imports:

import json
from datetime import datetime

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
                'confidence': 70 if signal_data.get('rsi', 50) > 55 else 60,
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

# In MEXCBot.__init__, add:
# self.signal_output = SignalOutput()

# In run_single_analysis, after detecting signal, add:
# self.signal_output.save_signal(symbol, signal_detected, signal_5m)
'''
    
    with open('integration_guide.txt', 'w') as f:
        f.write(integration_code)
    
    print("✅ Integration files created")

def create_runner_script():
    """Create a script to run both bots"""
    
    runner_content = '''#!/usr/bin/env python3
"""
Run both MEXC Bot and Signal Analyzer
"""

import subprocess
import time
import os
from threading import Thread

def run_analyzer():
    """Run the signal analyzer"""
    print("🚀 Starting Signal Analyzer...")
    subprocess.run(["python", "signal_analyzer.py"])

def run_bot():
    """Run the MEXC bot"""
    print("⏰ Waiting 5 seconds for analyzer to initialize...")
    time.sleep(5)
    print("🚀 Starting MEXC Bot...")
    subprocess.run(["python", "mexc_bot.py"])

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════╗
    ║   MEXC BOT + SIGNAL ANALYZER SYSTEM       ║
    ╚═══════════════════════════════════════════╝
    
    Starting integrated system...
    """)
    
    # Start analyzer in background thread
    analyzer_thread = Thread(target=run_analyzer)
    analyzer_thread.daemon = True
    analyzer_thread.start()
    
    # Run bot in main thread
    run_bot()
'''
    
    with open('run_both.py', 'w') as f:
        f.write(runner_content)
    
    os.chmod('run_both.py', 0o755)  # Make executable
    print("✅ Runner script created: run_both.py")

def show_integration_steps():
    """Show step-by-step integration guide"""
    
    guide = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                    INTEGRATION GUIDE                              ║
    ╚═══════════════════════════════════════════════════════════════════╝

    STEP 1: MODIFY YOUR EXISTING mexc_bot.py
    ----------------------------------------
    1. Open your mexc_bot.py file
    2. Add the SignalOutput class (see integration_guide.txt)
    3. In MEXCBot.__init__, add:
       self.signal_output = SignalOutput()
    
    4. In run_single_analysis method, find this section:
       if signal_detected:
           message = self.format_trading_info(...)
           alerts.append(message)
    
    5. Add this line after alerts.append(message):
       self.signal_output.save_signal(symbol, signal_detected, signal_5m)

    STEP 2: TEST THE INTEGRATION
    ----------------------------
    1. Run the analyzer first:
       python signal_analyzer.py
    
    2. In another terminal, run your bot:
       python mexc_bot.py
    
    3. Or use the combined runner:
       python run_both.py

    STEP 3: USE THE SYSTEM
    ----------------------
    Bot Signals (Automatic):
    - Your bot will automatically send signals to analyzer
    - Analyzer tracks performance in real-time
    
    Manual Signals (Telegram):
    - /long BTCUSDT 116000
    - /short ETHUSDT 4200
    - /analyze BTCUSDT LONG 116000
    - /performance 7
    - /winrate BTCUSDT
    
    STEP 4: MONITOR PERFORMANCE
    ---------------------------
    - Check win rates: /winrate
    - View active signals: /active
    - Get performance reports: /performance
    - Export data: /export

    FEATURES:
    ---------
    ✅ Automatic signal capture from bot
    ✅ Manual signal input via Telegram
    ✅ Real-time win rate calculation
    ✅ Performance tracking per symbol
    ✅ Risk metrics (Sharpe ratio, drawdown)
    ✅ Time-based analysis
    ✅ Confidence scoring
    ✅ Multi-source tracking (bot vs manual)
    
    TELEGRAM COMMANDS:
    -----------------
    /long SYMBOL PRICE [TP] [SL] - Add long signal
    /short SYMBOL PRICE [TP] [SL] - Add short signal
    /analyze SYMBOL DIRECTION PRICE - Analyze signal
    /performance [DAYS] - Performance report
    /winrate [SYMBOL] - Win rate stats
    /active - Show active signals
    /export - Export to CSV
    /help - Show all commands
    
    DATABASE STRUCTURE:
    ------------------
    signals_database.db contains:
    - signal_id: Unique identifier
    - symbol: Trading pair
    - direction: LONG/SHORT
    - entry_price: Entry price
    - target_price: Take profit
    - stop_loss: Stop loss
    - status: ACTIVE/WIN/LOSS
    - pnl_percent: Profit/Loss %
    - source: BOT/MANUAL
    - confidence: Signal confidence score
    
    PERFORMANCE METRICS:
    -------------------
    - Win Rate: % of winning trades
    - Profit Factor: Total wins / Total losses
    - Expectancy: Average expected return
    - Sharpe Ratio: Risk-adjusted returns
    - Max Drawdown: Largest peak-to-trough decline
    - Calmar Ratio: Annual return / Max drawdown
    """
    
    print(guide)

def create_example_usage():
    """Create example usage scenarios"""
    
    examples = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                    EXAMPLE USAGE SCENARIOS                        ║
    ╚═══════════════════════════════════════════════════════════════════╝

    SCENARIO 1: TESTING A MANUAL SIGNAL
    -----------------------------------
    You see BTCUSDT at 116,000 and think it will go up.
    
    1. Add signal: /long BTCUSDT 116000
    2. Analyzer automatically sets:
       - Target: 118,320 (2% up)
       - Stop: 113,680 (2% down)
    3. Check analysis: /analyze BTCUSDT LONG 116000
    4. Monitor: /active

    SCENARIO 2: CUSTOM TARGETS
    -------------------------
    You want specific TP/SL levels:
    
    /long BTCUSDT 116000 118000 115000
    - Entry: 116,000
    - Target: 118,000
    - Stop: 115,000

    SCENARIO 3: ANALYZING BOT PERFORMANCE
    ------------------------------------
    After running bot for a week:
    
    1. Overall stats: /performance 7
    2. Specific symbol: /winrate BTCUSDT
    3. Export data: /export
    4. View report with:
       - Win rate by symbol
       - Best trading hours
       - Risk metrics

    SCENARIO 4: COMPARING STRATEGIES
    --------------------------------
    Track bot vs manual signals:
    
    - Bot signals: Automatic from your EMA strategy
    - Manual signals: Your discretionary trades
    - Compare performance in reports

    SCENARIO 5: RISK MANAGEMENT
    --------------------------
    Monitor risk metrics:
    
    /performance 30
    Shows:
    - Sharpe Ratio (risk-adjusted returns)
    - Maximum Drawdown
    - Win/Loss streaks
    - Volatility

    PRO TIPS:
    ---------
    1. Let analyzer run 24/7 to track all signals
    2. Use manual signals to test ideas
    3. Review weekly reports to improve strategy
    4. Export data for deeper analysis in Excel
    5. Monitor win rate by symbol to find best pairs
    """
    
    with open('usage_examples.txt', 'w') as f:
        f.write(examples)
    
    print("✅ Usage examples created: usage_examples.txt")

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║          MEXC BOT + SIGNAL ANALYZER INTEGRATION SETUP             ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    This script will help you integrate the Signal Analyzer with your
    existing MEXC bot for comprehensive performance tracking.
    """)
    
    # Create necessary files
    create_integration_files()
    create_runner_script()
    create_example_usage()
    
    # Show integration guide
    show_integration_steps()
    
    print("\n✅ Setup complete! Follow the integration guide above.")
    print("📁 Files created:")
    print("   - integration_guide.txt (code to add to your bot)")
    print("   - run_both.py (script to run both systems)")
    print("   - usage_examples.txt (example scenarios)")
    print("   - active_signals.json (signal storage)")
    print("\n🚀 Ready to start tracking your trading performance!")
