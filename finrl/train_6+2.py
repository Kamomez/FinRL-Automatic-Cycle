from __future__ import annotations
import datetime
from pandas.tseries.offsets import BDay
import torch
import json
import os
from selected_etfs_and_stocks_30 import SELECTED_TICKERS_30, SELECTED_STOCKS_20, SELECTED_ETFS_10

DATA_API_KEY = "PKKK5YRQJLXN7TXEZSIN5UIIOG"
DATA_API_SECRET = "EMPJeJMkB6TCpARCWpyF8mw4yYxdRxWM6UkaduT9TbRz"
DATA_API_BASE_URL = "https://data.alpaca.markets"

from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.paper_trading.common import train, test
from finrl.config import INDICATORS

ticker_list = SELECTED_TICKERS_30
env = StockTradingEnv
GPU_ID = 0

if GPU_ID >= 0 and torch.cuda.is_available():
    print(f"{torch.cuda.get_device_name(GPU_ID)}")
else:
    print("CPU training")

# ElegantRL
ERL_PARAMS = {
    "learning_rate": 3e-6,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": [128, 64],
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 1,
}

# 6+2
today = datetime.datetime.today()

TEST_END_DATE = (today - BDay(1)).to_pydatetime().date()
TEST_START_DATE = (TEST_END_DATE - BDay(1)).to_pydatetime().date()
TRAIN_END_DATE = (TEST_START_DATE - BDay(1)).to_pydatetime().date()
TRAIN_START_DATE = (TRAIN_END_DATE - BDay(5)).to_pydatetime().date()
TRAINFULL_START_DATE = TRAIN_START_DATE
TRAINFULL_END_DATE = TEST_END_DATE

TRAIN_START_DATE = str(TRAIN_START_DATE)
TRAIN_END_DATE = str(TRAIN_END_DATE)
TEST_START_DATE = str(TEST_START_DATE)
TEST_END_DATE = str(TEST_END_DATE)
TRAINFULL_START_DATE = str(TRAINFULL_START_DATE)
TRAINFULL_END_DATE = str(TRAINFULL_END_DATE)

print(f"  training: {TRAIN_START_DATE} to {TRAIN_END_DATE} ")
print(f"   testing: {TEST_START_DATE} to {TEST_END_DATE} ")
print(f"   full training: {TRAINFULL_START_DATE} to {TRAINFULL_END_DATE} ")
print(f"   asset count: {len(ticker_list)} ")

stock_dim = len(ticker_list)
tech_dim = len(INDICATORS) * stock_dim
state_dim = 1 + 2 + 3 * stock_dim + tech_dim
action_dim = stock_dim

trained_model_dir = "./papertrading_backtest_30"
retrain_model_dir = "./papertrading_backtest_30_retrain"

_ = train(
    start_date=TRAIN_START_DATE,
    end_date=TRAIN_END_DATE,
    ticker_list=ticker_list,
    data_source="alpaca",
    time_interval="1D",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    if_vix=True,
    API_KEY=DATA_API_KEY,
    API_SECRET=DATA_API_SECRET,
    API_BASE_URL=DATA_API_BASE_URL,
    erl_params=ERL_PARAMS,
    cwd=trained_model_dir,
    break_step=1e5,
    stock_dim=stock_dim,
    state_dim=state_dim,
    action_dim=action_dim,
)

print(f"model saved to: {trained_model_dir}")

account_value_erl = test(
    start_date=TEST_START_DATE,
    end_date=TEST_END_DATE,
    ticker_list=ticker_list,
    data_source="alpaca",
    time_interval="1D",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    if_vix=True,
    API_KEY=DATA_API_KEY,
    API_SECRET=DATA_API_SECRET,
    API_BASE_URL=DATA_API_BASE_URL,
    cwd=trained_model_dir,
    net_dimension=ERL_PARAMS["net_dimension"],
    stock_dim=stock_dim,
    state_dim=state_dim,
    action_dim=action_dim,
)

# retrain with full data
_ = train(
    start_date=TRAINFULL_START_DATE,
    end_date=TRAINFULL_END_DATE,
    ticker_list=ticker_list,
    data_source="alpaca",
    time_interval="1D",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    if_vix=True,
    API_KEY=DATA_API_KEY,
    API_SECRET=DATA_API_SECRET,
    API_BASE_URL=DATA_API_BASE_URL,
    erl_params=ERL_PARAMS,
    cwd=retrain_model_dir,
    break_step=1e5,
    stock_dim=stock_dim,
    state_dim=state_dim,
    action_dim=action_dim,
)

print(f"retrained model saved to: {retrain_model_dir}")

config = {
    "train_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "assets": {
        "total": len(ticker_list),
        "stocks": len(SELECTED_STOCKS_20),
        "etfs": len(SELECTED_ETFS_10),
        "tickers": ticker_list
    },
    "model_params": {
        "stock_dim": stock_dim,
        "tech_dim": tech_dim,
        "state_dim": state_dim,
        "action_dim": action_dim,
    },
    "erl_params": ERL_PARAMS,
    "time_window": {
        "train_start": TRAIN_START_DATE,
        "train_end": TRAIN_END_DATE,
        "test_start": TEST_START_DATE,
        "test_end": TEST_END_DATE,
        "full_train_start": TRAINFULL_START_DATE,
        "full_train_end": TRAINFULL_END_DATE,
    }
}

config_file = f"{retrain_model_dir}/training_config.json"
with open(config_file, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)

print(f"retrained model config saved to: {config_file}")