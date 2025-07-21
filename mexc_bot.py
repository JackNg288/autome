# mexc_bot.py
import requests, pandas as pd, time

# === CONFIG ===
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
EMA_PERIOD = 20
BASE_URL = "https://api.mexc.com"
TELEGRAM_TOKEN = "7596862485:AAGNlV893IdMVRVhx07UZjgZf51fKefUNAg"
CHAT_ID = "1465742044"

# === FUNCTIONS ===
def fetch_klines(symbol, interval, limit=50):
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    res = requests.get(url, params=params)
    res.raise_for_status()
    df = pd.DataFrame(res.json(), columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote", "trades", "buy_base", "buy_quote", "ignore"
    ])
    df["close"] = pd.to_numeric(df["close"])
    df["volume"] = pd.to_numeric(df["volume"])
    return df

def check_signal(df):
    df["EMA20"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    vol_avg = df["volume"].rolling(EMA_PERIOD).mean()
    last = df.iloc[-1]
    return last["close"] > last["EMA20"] and last["volume"] > vol_avg.iloc[-2]

def send_alert(message):
    url = f"https://api.telegram.org/bot7596862485:AAGNlV893IdMVRVhx07UZjgZf51fKefUNAg/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=payload)

# === MAIN LOOP ===
messages = []
for symbol in SYMBOLS:
    try:
        one = fetch_klines(symbol, "1h")
        four = fetch_klines(symbol, "4h")
        if check_signal(one) and check_signal(four):
            messages.append(f"🚀 BUY SIGNAL: {symbol}")
    except Exception as e:
        messages.append(f"⚠️ Error for {symbol}: {str(e)}")

if messages:
    send_alert("\n".join(messages))
send_alert("✅ Bot ran successfully at " + time.strftime("%Y-%m-%d %H:%M:%S"))
