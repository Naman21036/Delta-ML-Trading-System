import time
import csv
from datetime import datetime
import os
import json

from delta_api1 import get_ticker, place_order, get_product_id
from config import SYMBOL, TRADE_SIZE, FETCH_INTERVAL, LOG_FILE
from model_inference import predict_signal

POSITION_STATE_FILE = "position_state.json"


def load_position_state(path: str = POSITION_STATE_FILE) -> int:
    """
    Load current position (in contracts) from disk.
    Returns zero if file does not exist or is invalid.
    """
    if not os.path.exists(path):
        return 0

    try:
        with open(path, "r") as f:
            data = json.load(f)
        pos = int(data.get("current_position", 0))
        return pos
    except Exception as e:
        print("Warning could not load position state from file:", e)
        return 0


def save_position_state(current_position: int, path: str = POSITION_STATE_FILE) -> None:
    """
    Save current position to disk as json.
    """
    try:
        data = {
            "current_position": int(current_position),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print("Warning could not save position state:", e)


def log_trade(timestamp, price, signal, order_response, position_after):
    """
    Append one row to the log file.
    """
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            price,
            signal,
            order_response.get("status", "hold") if order_response else "hold",
            order_response.get("side", "") if order_response else "",
            order_response.get("product_id", "") if order_response else "",
            position_after,
        ])


def main():
    print("✅ Starting Paper Trading on Delta Exchange Testnet...")

    # create log file header if it does not exist
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp",
                "Price",
                "Signal",
                "OrderStatus",
                "Side",
                "ProductID",
                "PositionAfter",
            ])

    product_id = get_product_id(SYMBOL)
    print(f"Using SYMBOL = {SYMBOL}, product_id = {product_id}, TRADE_SIZE = {TRADE_SIZE}")
    if not product_id:
        print("Error: product_id not found. Check symbol and API connectivity.")
        return

    # load position from disk so restart is safe
    current_position = load_position_state()
    print(f"Loaded position from disk: {current_position} contracts")

    while True:
        try:
            ticker = get_ticker(SYMBOL)
            if not ticker:
                print("Warning: No ticker returned, retrying...")
                time.sleep(FETCH_INTERVAL)
                continue

            price = float(
                ticker.get("mark_price", 0)
                or ticker.get("close", 0)
                or 0
            )
            if price <= 0:
                print("Warning: Invalid price, retrying...")
                time.sleep(FETCH_INTERVAL)
                continue

            signal = predict_signal()
            now = datetime.now()
            print(f"[{now}] Price: {price} | Signal: {signal} | Position: {current_position}")

            # decide whether we actually want to trade
            order_side = None

                        # decide whether we actually want to trade
            order_side = None

            # open a new long only if flat or short (practically flat in your case)
            if signal == "buy" and current_position <= 0:
                order_side = "buy"

            # close long if we get a sell signal and we are currently long
            elif signal == "sell" and current_position > 0:
                order_side = "sell"


            order_response = None

            if order_side is not None:
                print(f"Placing order: side={order_side}, size={TRADE_SIZE}, product_id={product_id}")
                order_response = place_order(product_id, order_side, TRADE_SIZE)

                if order_response:
                    # update in memory position
                    filled_size = float(order_response.get("size", 0))
                    side = order_response.get("side", "")

                    if side == "buy":
                        current_position += int(filled_size)
                    elif side == "sell":
                        current_position -= int(filled_size)

                    # persist position state
                    save_position_state(current_position)

                    print(f"Order API response: {order_response}")
                    print(f"New position: {current_position}")
                else:
                    print("Warning: No response from order, logging as hold.")
                    order_response = {"status": "hold"}
            else:
                # no structural change to position required
                order_response = {"status": "hold"}

            log_trade(now, price, signal, order_response, current_position)
            time.sleep(FETCH_INTERVAL)

        except Exception as e:
            print("Error in loop:", e)
            time.sleep(10)


if __name__ == "__main__":
    main()
