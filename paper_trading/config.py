import os
API_KEY = os.getenv("DELTA_API_KEY")
API_SECRET = os.getenv("DELTA_API_SECRET")
BASE_URL = os.getenv("DELTA_BASE_URL")

SYMBOL = os.getenv("DELTA_SYMBOL", "BTCUSD")   # Use 'BTCUSDT' for global testnet
TRADE_SIZE = int(os.getenv("TRADE_SIZE", "1"))
FETCH_INTERVAL = int(os.getenv("FETCH_INTERVAL", "60"))      # in seconds
RESOLUTION = os.getenv("RESOLUTION", "1m")

LOG_FILE = os.getenv("LOG_FILE", "paper_trading_log.csv")
USER_AGENT = os.getenv("USER_AGENT", "delta-forward-tester/1.0")
