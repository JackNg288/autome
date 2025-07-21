#!/usr/bin/env python3
"""
MEXC Trading Bot - Improved Version
Monitors multiple crypto pairs for buy signals based on EMA and volume analysis.
"""

import requests
import pandas as pd
import time
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MEXCBot:
    def __init__(self):
        """Initialize the MEXC trading bot."""
        # Configuration - Use environment variables for sensitive data
        self.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self.ema_period = 20
        self.base_url = "https://api.mexc.com"
        
        # Get Telegram credentials from environment variables
        self.telegram_token = os.getenv('TELEGRAM_TOKEN', 'YOUR_TELEGRAM_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID')
        
        # Request session for better connection management
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MEXC-Bot/1.0'
        })
    
    def fetch_klines(self, symbol: str, interval: str, limit: int = 50) -> Optional[pd.DataFrame]:
        """
        Fetch kline/candlestick data from MEXC API.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval ('1m', '5m', '1h', '4h', '1d', etc.)
            limit: Number of klines to retrieve (max 1000)
        
        Returns:
            DataFrame with kline data or None if error
        """
        url = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        try:
            logger.debug(f"Fetching {interval} klines for {symbol}")
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if not data:
                logger.warning(f"No data returned for {symbol} {interval}")
                return None
            
            # Create DataFrame with proper column names
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "buy_base_volume", 
                "buy_quote_volume", "ignore"
            ])
            
            # Convert to numeric types
            numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert timestamp to datetime for easier handling
            df["datetime"] = pd.to_datetime(df["timestamp"], unit='ms')
            
            logger.debug(f"Successfully fetched {len(df)} klines for {symbol} {interval}")
            return df
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching klines for {symbol} {interval}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching klines for {symbol} {interval}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching klines for {symbol} {interval}: {e}")
            return None
    
    def check_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for buy signal based on EMA and volume analysis.
        
        Args:
            df: DataFrame with kline data
        
        Returns:
            Dictionary with signal information
        """
        if df is None or len(df) < self.ema_period:
            return {"signal": False, "reason": "Insufficient data"}
        
        try:
            # Calculate EMA
            df_copy = df.copy()
            df_copy["ema"] = df_copy["close"].ewm(span=self.ema_period, adjust=False).mean()
            
            # Calculate volume average
            vol_avg = df_copy["volume"].rolling(window=self.ema_period).mean()
            
            # Get latest values
            latest = df_copy.iloc[-1]
            prev_vol_avg = vol_avg.iloc[-2] if len(vol_avg) > 1 else vol_avg.iloc[-1]
            
            # Signal conditions
            price_above_ema = latest["close"] > latest["ema"]
            volume_spike = latest["volume"] > prev_vol_avg
            
            signal = price_above_ema and volume_spike
            
            return {
                "signal": signal,
                "price": latest["close"],
                "ema": latest["ema"],
                "volume": latest["volume"],
                "vol_avg": prev_vol_avg,
                "price_above_ema": price_above_ema,
                "volume_spike": volume_spike,
                "timestamp": latest["datetime"]
            }
            
        except Exception as e:
            logger.error(f"Error calculating signal: {e}")
            return {"signal": False, "reason": f"Calculation error: {e}"}
    
    def send_telegram_alert(self, message: str) -> bool:
        """
        Send alert message to Telegram.
        
        Args:
            message: Message to send
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.telegram_token or self.telegram_token == 'YOUR_TELEGRAM_TOKEN':
            logger.warning("Telegram token not configured, skipping alert")
            return False
        
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"  # Enable markdown formatting
        }
        
        try:
            response = self.session.post(url, data=payload, timeout=10)
            response.raise_for_status()
            logger.info("Telegram alert sent successfully")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
    
    def format_signal_message(self, symbol: str, signal_1h: Dict, signal_4h: Dict) -> str:
        """Format a buy signal message."""
        message = f"🚀 *BUY SIGNAL: {symbol}*\n"
        message += f"Price: ${signal_1h['price']:.4f}\n"
        message += f"1H EMA20: ${signal_1h['ema']:.4f}\n"
        message += f"4H EMA20: ${signal_4h['ema']:.4f}\n"
        message += f"Time: {signal_1h['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
        return message
    
    def run_analysis(self) -> None:
        """Run the main analysis for all symbols."""
        logger.info("Starting MEXC bot analysis")
        messages = []
        
        for symbol in self.symbols:
            try:
                logger.info(f"Analyzing {symbol}")
                
                # Fetch 1H and 4H data
                df_1h = self.fetch_klines(symbol, "1h")
                df_4h = self.fetch_klines(symbol, "4h")
                
                if df_1h is None or df_4h is None:
                    messages.append(f"⚠️ Error: Could not fetch data for {symbol}")
                    continue
                
                # Check signals
                signal_1h = self.check_signal(df_1h)
                signal_4h = self.check_signal(df_4h)
                
                # Both timeframes must show buy signal
                if signal_1h["signal"] and signal_4h["signal"]:
                    message = self.format_signal_message(symbol, signal_1h, signal_4h)
                    messages.append(message)
                    logger.info(f"Buy signal detected for {symbol}")
                else:
                    logger.debug(f"No signal for {symbol}")
                    logger.debug(f"1H: {signal_1h}")
                    logger.debug(f"4H: {signal_4h}")
                
            except Exception as e:
                error_msg = f"⚠️ Error analyzing {symbol}: {str(e)}"
                messages.append(error_msg)
                logger.error(error_msg)
        
        # Send alerts if any signals found
        if messages:
            combined_message = "\n\n".join(messages)
            self.send_telegram_alert(combined_message)
        
        # Send status update
        status_msg = f"✅ Bot analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        if not messages:
            status_msg += "\n📊 No buy signals detected"
        
        self.send_telegram_alert(status_msg)
        logger.info("Analysis completed")
    
    def run_continuous(self, interval_minutes: int = 60) -> None:
        """
        Run the bot continuously with specified interval.
        
        Args:
            interval_minutes: Minutes between analysis runs
        """
        logger.info(f"Starting continuous mode with {interval_minutes} minute intervals")
        
        while True:
            try:
                self.run_analysis()
                logger.info(f"Sleeping for {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in continuous mode: {e}")
                time.sleep(60)  # Wait 1 minute before retrying


def main():
    """Main entry point."""
    bot = MEXCBot()
    
    # Run single analysis
    bot.run_analysis()
    
    # Uncomment below to run continuously
    bot.run_continuous(interval_minutes=5)


if __name__ == "__main__":
    main()
