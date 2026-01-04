import requests
import time
import hmac
import hashlib
import json
from urllib.parse import urlencode

from config import API_KEY, API_SECRET, BASE_URL, USER_AGENT

# Use prod API for market data (candles), testnet BASE_URL for trading
MARKET_DATA_BASE_URL = "https://api.delta.exchange"


def _json_minify(obj):
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    # use default json.dumps (with spaces) so it matches what we send
    return json.dumps(obj, ensure_ascii=False)

def generate_signature(timestamp, method, endpoint, query=None, body=None):
    query_str = _encode_query(query)
    body_str = _json_minify(body)
    prehash = f"{method.upper()}{timestamp}{endpoint}{query_str}{body_str}"
    return hmac.new(API_SECRET.encode(), prehash.encode(), hashlib.sha256).hexdigest()




def _encode_query(query: dict | None) -> str:
    if not query:
        return ""
    return "?" + urlencode(query, doseq=True)


def get_headers(method, endpoint, query=None, body=None):
    timestamp = str(int(time.time()))
    signature = generate_signature(timestamp, method, endpoint, query, body)
    return {
        "api-key": API_KEY,
        "timestamp": timestamp,
        "signature": signature,
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }


def get_product_id(symbol):
    endpoint = f"/v2/products/{symbol}"
    url = BASE_URL + endpoint
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
        return data["result"]["id"]
    except Exception as e:
        print("Error fetching product_id:", e)
        try:
            print("Response text:", resp.text)
        except Exception:
            pass
        return None


def get_ticker(symbol):
    endpoint = f"/v2/tickers/{symbol}"
    url = BASE_URL + endpoint
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        return resp.json().get("result", None)
    except Exception as e:
        print("Error fetching ticker:", e)
        try:
            print("Response text:", resp.text)
        except Exception:
            pass
        return None


def get_candles(symbol, resolution="1h", window=50):
    """
    Fetch last `window` candles for given symbol and resolution.

    resolution examples: "1m", "5m", "15m", "1h"
    window is number of candles you want
    """
    endpoint = "/v2/history/candles"
    end_ts = int(time.time())

    # Convert resolution string to seconds per candle
    if resolution.endswith("m"):
        step = int(resolution[:-1]) * 60
    elif resolution.endswith("h"):
        step = int(resolution[:-1]) * 3600
    else:
        step = 60  # default 1 minute

    start_ts = end_ts - window * step

    params = {
        "symbol": symbol,
        "resolution": resolution,
        "start": start_ts,
        "end": end_ts,
    }

    url = MARKET_DATA_BASE_URL + endpoint
    try:
        resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
        candles = data.get("result", [])

        if not candles:
            print("Warning: empty candles result for", symbol, "params:", params)

        return candles
    except Exception as e:
        print("Error fetching candles:", e)
        try:
            print("Response text:", resp.text)
        except Exception:
            pass
        return None


def place_order(product_id, side, size):
    endpoint = "/v2/orders"
    url = BASE_URL + endpoint

    # Build body exactly like docs example, but with market_order
    body_dict = {
        "order_type": "market_order",   # or "limit_order" if you later add limit_price
        "size": size,
        "side": side,
        "product_id": product_id,
    }
    # Single canonical string for both signature and HTTP body
    payload = json.dumps(body_dict, ensure_ascii=False)

    # DEBUG
    print("DEBUG payload:", payload)

    # Use payload string when generating signature
    timestamp = str(int(time.time()))
    signature = generate_signature(timestamp, "POST", endpoint, body=payload)

    headers = {
        "api-key": API_KEY,
        "timestamp": timestamp,
        "signature": signature,
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }

    try:
        # IMPORTANT: use data=payload (raw JSON string), not json=body_dict
        resp = requests.post(url, headers=headers, data=payload)

        print("DEBUG status:", resp.status_code, "body:", resp.text)

        if resp.status_code == 401:
            print("Unauthorized: Check API key, environment, IP whitelist, or permissions.")
            return None

        resp.raise_for_status()
        return resp.json().get("result", {"status": "submitted"})
    except Exception as e:
        print("Error placing order:", e)
        return None
