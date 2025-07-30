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
    
    with open('run_both.py', 'w')
