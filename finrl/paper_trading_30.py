from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from finrl.config import INDICATORS
from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca

ALPACA_API_KEY = ""
ALPACA_API_SECRET = ""
ALPACA_API_BASE_URL = ""

try:
    from selected_etfs_and_stocks_30 import SELECTED_TICKERS_30, SELECTED_STOCKS_20, SELECTED_ETFS_10
    print(f"stocks:{len(SELECTED_STOCKS_20)}")
    print(f"ETF:{len(SELECTED_ETFS_10)} ")
    ticker_list = SELECTED_TICKERS_30
except ImportError:
    print("ERROR: selected_etfs_and_stocks_30.py not found")
    exit(1)

stock_dim = len(ticker_list)
tech_dim = len(INDICATORS) * stock_dim
state_dim = 1 + 2 + 3 * stock_dim + tech_dim
action_dim = stock_dim 

net_dim = [128, 64]
cwd = "./papertrading_backtest_v3_retrain" 
time_interval = "1Min"

print(f"model path: {cwd}")
print(f"time interval: {time_interval}")

import os
if not os.path.exists(f"{cwd}/actor.pth"):
    print(f"error: {cwd}/actor.pth not found")
    exit(1)

import json
config_file = f"{cwd}/training_config.json"
if os.path.exists(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
        saved_state_dim = config.get('model_params', {}).get('state_dim', 0)
        saved_action_dim = config.get('model_params', {}).get('action_dim', 0)
else:
    print(f"{config_file} not found")

import alpaca_trade_api as tradeapi
alpaca_api = tradeapi.REST(
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_API_BASE_URL,
    'v2'
)

try:
    positions = alpaca_api.list_positions()
    if len(positions) > 0:
        print(f" there is{len(positions)} stocks:")
        for pos in positions:
            symbol = pos.symbol
            qty = pos.qty
            print(f"{symbol}: {qty} ")
            
            if symbol not in ticker_list:
                print(f"{symbol} not in ticker_list, closing position")

        alpaca_api.close_all_positions()
    else:
        print("no positions found")
except Exception as e:
    print(f"error {e}")

# start
paper_trading = PaperTradingAlpaca(
    ticker_list=ticker_list,
    time_interval=time_interval,
    drl_lib="elegantrl",
    agent="ppo",
    cwd=cwd,
    net_dim=net_dim,
    state_dim=state_dim,
    action_dim=action_dim,
    API_KEY=ALPACA_API_KEY,
    API_SECRET=ALPACA_API_SECRET,
    API_BASE_URL=ALPACA_API_BASE_URL,
    tech_indicator_list=INDICATORS,
    turbulence_thresh=30,
    max_stock=1e2,
)

paper_trading.run()
