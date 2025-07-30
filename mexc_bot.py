#!/usr/bin/env python3
"""
Modified MEXC Bot - Integrated with Signal Analyzer
This version sends signals to the analyzer for performance tracking
"""

# Add this import at the top of your existing mexc_bot.py
import json
from datetime import datetime
from typing import Dict

# Add this class to your existing mexc_bot.py after the imports
class SignalOutput:
    """Output signals for the analyzer"""
    
    def __init__(self):
        self.signal_file = "active_signals.json"
        
    def save_signal(self, symbol: str, signal_type: str, signal_data: Dict):
        """Save signal for analyzer to process"""
        try:
            # Create signal record
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
                'atr_percent': 2.0  # Default 2% for TP/SL calculation
            }
            
            # Load existing signals
            signals = []
            if os.path.exists(self.signal_file):
                try:
                    with open(self.signal_file, 'r') as f:
                        signals = json.load(f)
                except:
                    signals = []
            
            # Add new signal
            signals.append(signal_record)
            
            # Save back
            with open(self.signal_file, 'w') as f:
                json.dump(signals, f, indent=2)
                
            logger.info(f"Signal saved for analyzer: {signal_record['signal_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving signal for analyzer: {e}")
            return False
    
    def _calculate_confidence(self, signal_data: Dict) -> float:
        """Calculate signal confidence based on indicators"""
        confidence = 50.0  # Base confidence
        
        # RSI confirmation
        rsi = signal_data.get('rsi', 50)
        if 55 < rsi < 70:  # Good RSI range for long
            confidence += 10
        elif 30 < rsi < 45:  # Good RSI range for short
            confidence += 10
        
        # Volume confirmation
        if signal_data.get('volume', 0) > signal_data.get('volume_ma', 0) * 1.2:
            confidence += 10
        
        # EMA alignment
        if signal_data.get('ema5', 0) > signal_data.get('ema10', 0) > signal_data.get('ema15', 0):
            confidence += 10  # Bullish alignment
        elif signal_data.get('ema5', 0) < signal_data.get('ema10', 0) < signal_data.get('ema15', 0):
            confidence += 10  # Bearish alignment
        
        return min(confidence, 90)  # Cap at 90%


# Modify your existing MEXCBot class - add this to __init__:
# self.signal_output = SignalOutput()

# Modify your run_single_analysis method in MEXCBot class:
# Add this code block after detecting a signal (around line 850 in your original code)
"""
# After this line: if signal_detected:
#     message = self.format_trading_info(symbol, signal_5m, signal_15m, signal_detected, levels_15m)
#     alerts.append(message)
#     
#     # ADD THIS: Save signal for analyzer
#     self.signal_output.save_signal(symbol, signal_detected, signal_5m)
#     
#     logger.info(f"{signal_detected} signal detected for {symbol}")
"""

# Here's the complete modified run_single_analysis method:
def run_single_analysis_with_output(bot_instance):
    """Modified run_single_analysis that outputs to analyzer"""
    logger.info("Running EMA Crossover + RSI analysis...")
    bot_instance.check_telegram_updates()
    if not bot_instance.running:
        return
    
    alerts = []
    price_alerts_triggered = []
    
    # Initialize signal output if not exists
    if not hasattr(bot_instance, 'signal_output'):
        bot_instance.signal_output = SignalOutput()
    
    for symbol in bot_instance.symbols:
        df_5m = bot_instance.fetch_klines(symbol, "5m", 100)
        df_15m = bot_instance.fetch_klines(symbol, "15m", 100)
        if df_5m is None or df_15m is None:
            logger.warning(f"Could not fetch data for {symbol}")
            continue
        
        current_price = float(df_5m.iloc[-1]['close'])
        
        # Check price alerts first
        triggered = bot_instance.check_price_alerts(symbol, current_price)
        for alert in triggered:
            alert_message = bot_instance.format_price_alert(alert)
            price_alerts_triggered.append(alert_message)
            logger.info(f"Price alert triggered for {symbol} at ${current_price:.4f}")
        
        signal_5m = bot_instance.check_ema_crossover(df_5m)
        signal_15m = bot_instance.check_ema_crossover(df_15m)
        levels_15m = bot_instance.get_high_low_levels(df_15m, 5)
        
        signal_detected = None
        if (signal_5m["signal"] == "LONG" and signal_15m["signal"] == "LONG"):
            signal_detected = "LONG"
        elif (signal_5m["signal"] == "SHORT" and signal_15m["signal"] == "SHORT"):
            signal_detected = "SHORT"
        elif signal_5m["signal"] == "LONG" and signal_15m["signal"] != "SHORT":
            if signal_15m["rsi"] > bot_instance.rsi_long_threshold:
                signal_detected = "LONG"
        elif signal_5m["signal"] == "SHORT" and signal_15m["signal"] != "LONG":
            if signal_15m["rsi"] < bot_instance.rsi_short_threshold:
                signal_detected = "SHORT"
        
        if signal_detected:
            message = bot_instance.format_trading_info(symbol, signal_5m, signal_15m, signal_detected, levels_15m)
            alerts.append(message)
            
            # SAVE SIGNAL FOR ANALYZER
            bot_instance.signal_output.save_signal(symbol, signal_detected, signal_5m)
            
            logger.info(f"{signal_detected} signal detected for {symbol}")
            
            # Auto-set alert for base price (EMA15)
            base_price = signal_5m['ema15']
            if bot_instance.add_price_alert(symbol, base_price, "auto"):
                logger.info(f"Auto-alert set for {symbol} at base price ${base_price:.4f}")
            
            logger.info(f"{symbol} 5m - EMA5: {signal_5m['ema5']:.4f}, EMA10: {signal_5m['ema10']:.4f}, RSI: {signal_5m['rsi']:.1f}")
            logger.info(f"{symbol} 15m - EMA5: {signal_15m['ema5']:.4f}, EMA10: {signal_15m['ema10']:.4f}, RSI: {signal_15m['rsi']:.1f}")
            logger.info(f"{symbol} 15m Levels - High: {levels_15m['highest']:.4f}, Low: {levels_15m['lowest']:.4f}")
        else:
            logger.info(f"No clear signal for {symbol}")
    
    # Send all alerts
    if alerts:
        for alert in alerts:
            bot_instance.send_telegram_alert(alert)
    
    if price_alerts_triggered:
        for price_alert in price_alerts_triggered:
            bot_instance.send_telegram_alert(price_alert)
    
    logger.info(f"Analysis complete. {len(alerts)} signals detected, {len(price_alerts_triggered)} price alerts triggered.")


# Integration Instructions:
print("""
INTEGRATION INSTRUCTIONS FOR YOUR EXISTING MEXC BOT:

1. Add the SignalOutput class to your mexc_bot.py file

2. In your MEXCBot.__init__ method, add:
   self.signal_output = SignalOutput()

3. In your run_single_analysis method, after detecting a signal, add:
   self.signal_output.save_signal(symbol, signal_detected, signal_5m)

4. The signals will be saved to 'active_signals.json' file

5. Run both bots simultaneously:
   Terminal 1: python mexc_bot.py
   Terminal 2: python signal_analyzer.py

The analyzer will automatically pick up signals from the bot!
""")
