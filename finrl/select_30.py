from __future__ import annotations
import datetime
from pandas.tseries.offsets import BDay
import pandas as pd
import numpy as np
import sys
import io
import time
import json
from typing import List, Dict
from tqdm import tqdm
from finrl.config_tickers import SP_500_TICKER
from finrl.config import INDICATORS
from yahoo_etfs import YAHOO_ETFS_519

DATA_API_KEY = "PKKK5YRQJLXN7TXEZSIN5UIIOG"
DATA_API_SECRET = "EMPJeJMkB6TCpARCWpyF8mw4yYxdRxWM6UkaduT9TbRz"
DATA_API_BASE_URL = "https://data.alpaca.markets"

ALL_ETFS = YAHOO_ETFS_519

today = datetime.datetime.today()
# backtest_end_date = (today - BDay(1)).to_pydatetime().date()
BACKTEST_END_DATE = (today - BDay(1)).to_pydatetime().strftime("%Y-%m-%d")
BACKTEST_START_DATE = (pd.Timestamp(BACKTEST_END_DATE) - BDay(60)).to_pydatetime().strftime("%Y-%m-%d")

BACKTEST_CONFIG = {
    "initial_amount": 100000,
    "hmax": 100,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "reward_scaling": 1e-4,
    "tech_indicators": INDICATORS,
}


def batch_download_data(ticker_list: List[str], start_date: str, end_date: str, batch_size: int = 50) -> Dict[str, pd.DataFrame]:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from stockstats import StockDataFrame as Sdf
    
    client = StockHistoricalDataClient(DATA_API_KEY, DATA_API_SECRET)
    result_dict = {}
    
    for i in range(0, len(ticker_list), batch_size):
        batch = ticker_list[i:i+batch_size]
        
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            bars = client.get_stock_bars(request_params)
            df_all = bars.df
            
            if not df_all.empty:
                df_all = df_all.reset_index()
                
                for ticker in batch:
                    if 'symbol' in df_all.columns:
                        ticker_df = df_all[df_all['symbol'] == ticker].copy()
                    else:
                        continue
                    
                    if len(ticker_df) >= 30:
                        ticker_df = ticker_df.rename(columns={
                            'symbol': 'tic', 
                            'timestamp': 'date'
                        })
                        
                        if 'tic' not in ticker_df.columns:
                            ticker_df['tic'] = ticker
                        if 'date' not in ticker_df.columns and 'timestamp' in ticker_df.columns:
                            ticker_df['date'] = ticker_df['timestamp']
                        
                        if 'close' not in ticker_df.columns:
                            continue

                        try:
                            stock_df = Sdf.retype(ticker_df.copy())
                            for indicator in BACKTEST_CONFIG["tech_indicators"]:
                                try:
                                    stock_df[indicator]
                                except:
                                    pass
                            ticker_df = stock_df.copy()

                            if ticker_df.isnull().values.any():
                                ticker_df = ticker_df.ffill().bfill()
                            
                            result_dict[ticker] = ticker_df
                        except:
                            pass
            
            time.sleep(0.2)
            
        except Exception as e:
            if "rate limit" in str(e).lower():
                time.sleep(60)
            continue
    
    return result_dict

def backtest_single_ticker(ticker: str, df: pd.DataFrame, initial_amount: float = 100000) -> dict:
    if df.empty or len(df) < 5:
        return None
    
    try:
        buy_price = df.iloc[0]['close']
        sell_price = df.iloc[-1]['close']
        
        buy_cost_pct = BACKTEST_CONFIG['buy_cost_pct']
        sell_cost_pct = BACKTEST_CONFIG['sell_cost_pct']
        
        shares = int(initial_amount / (buy_price * (1 + buy_cost_pct)))
        actual_cost = shares * buy_price * (1 + buy_cost_pct)
        remaining_cash = initial_amount - actual_cost
        
        sell_proceeds = shares * sell_price * (1 - sell_cost_pct)
        final_value = remaining_cash + sell_proceeds
        
        total_return = (final_value - initial_amount) / initial_amount
        
        days = len(df)
        annual_return = total_return * (252 / days) if days > 0 else 0
        
        df_copy = df.copy()
        df_copy['daily_return'] = df_copy['close'].pct_change()
        
        df_copy['cumulative'] = (1 + df_copy['daily_return']).cumprod()
        df_copy['running_max'] = df_copy['cumulative'].cummax()
        df_copy['drawdown'] = (df_copy['cumulative'] - df_copy['running_max']) / df_copy['running_max']
        max_drawdown = df_copy['drawdown'].min()
        
        mean_return = df_copy['daily_return'].mean()
        std_return = df_copy['daily_return'].std()
        sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        win_rate = (df_copy['daily_return'] > 0).sum() / len(df_copy)

        if 'volume' in df.columns:
            turnover_series = df['close'] * df['volume']
            avg_daily_turnover = turnover_series.replace([np.inf, -np.inf], np.nan).dropna().mean()
            avg_daily_turnover = float(avg_daily_turnover) if pd.notna(avg_daily_turnover) else 0.0
        else:
            avg_daily_turnover = 0.0
        
        return {
            'ticker': ticker,
            'annual_return': annual_return,
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'win_rate': win_rate,
            'avg_daily_turnover': avg_daily_turnover,
            'days': days
        }
        
    except Exception as e:
        return None

def calculate_backtest_score(result: dict) -> float:
    
    annual_return_score = result['annual_return'] * 100
    sharpe_score = result['sharpe'] * 10
    calmar_score = result['calmar'] * 10
    drawdown_score = (1 + result['max_drawdown']) * 100 
    win_rate_score = result['win_rate'] * 100
    turnover = max(result.get('avg_daily_turnover', 0.0), 0.0)
    liquidity_score = np.clip((np.log10(turnover + 1) - 4) * 20, 0, 100)

    total_score = (
        annual_return_score * 0.25 +
        sharpe_score * 0.25 +
        calmar_score * 0.20 +
        drawdown_score * 0.10 +
        win_rate_score * 0.10 +
        liquidity_score * 0.10
    )
    
    return total_score

# download data for all tickers
stock_data = batch_download_data(SP_500_TICKER, BACKTEST_START_DATE, BACKTEST_END_DATE)
etf_data = batch_download_data(ALL_ETFS, BACKTEST_START_DATE, BACKTEST_END_DATE)


stock_results = []

for ticker in tqdm(stock_data.keys()):
    result = backtest_single_ticker(ticker, stock_data[ticker])
    if result:
        score = calculate_backtest_score(result)
        result['score'] = score
        stock_results.append(result)
    time.sleep(0.1)

print(f"completed {len(stock_results)} backtests for stocks")

# sorted by score
stock_results_sorted = sorted(stock_results, key=lambda x: x['score'], reverse=True)

# first 20 stocks
SELECTED_STOCKS_20 = [r['ticker'] for r in stock_results_sorted[:20]]

if stock_results:
    avg_return = np.mean([r['annual_return'] for r in stock_results])
    avg_sharpe = np.mean([r['sharpe'] for r in stock_results])
    avg_drawdown = np.mean([r['max_drawdown'] for r in stock_results])
    print(f"average annual return: {avg_return*100:.2f}%")
    print(f"average Sharpe: {avg_sharpe:.2f}")
    print(f"average max drawdown: {avg_drawdown*100:.2f}%")

for i, ticker in enumerate(SELECTED_STOCKS_20[:10], 1):
    result = next(r for r in stock_results_sorted if r['ticker'] == ticker)
    print(f"  {i:2d}. {ticker:6s} - score: {result['score']:8.2f} | annual return: {result['annual_return']*100:6.2f}% | Sharpe: {result['sharpe']:5.2f} | avg daily turnover: ${result['avg_daily_turnover']:,.0f}")
if len(SELECTED_STOCKS_20) > 10:
    print(f" remaining {len(SELECTED_STOCKS_20)-10} stocks")

# etf backtest
etf_results = []

for ticker in tqdm(etf_data.keys()):
    result = backtest_single_ticker(ticker, etf_data[ticker])
    if result:
        score = calculate_backtest_score(result)
        result['score'] = score
        etf_results.append(result)
    time.sleep(0.1)

etf_results_sorted = sorted(etf_results, key=lambda x: x['score'], reverse=True)
SELECTED_ETFS_10 = [r['ticker'] for r in etf_results_sorted[:10]]

if etf_results:
    avg_return = np.mean([r['annual_return'] for r in etf_results])
    avg_sharpe = np.mean([r['sharpe'] for r in etf_results])
    avg_drawdown = np.mean([r['max_drawdown'] for r in etf_results])
    print(f"  average annual return: {avg_return*100:.2f}%")
    print(f"  average Sharpe:   {avg_sharpe:.2f}")
    print(f"  average max drawdown: {avg_drawdown*100:.2f}%")

for i, ticker in enumerate(SELECTED_ETFS_10, 1):
    result = next(r for r in etf_results_sorted if r['ticker'] == ticker)
    print(f"  {i:2d}. {ticker:6s} - score: {result['score']:8.2f} | annual return: {result['annual_return']*100:6.2f}% | Sharpe: {result['sharpe']:5.2f} | avg daily turnover: ${result['avg_daily_turnover']:,.0f}")

# result stored
SELECTED_TICKERS_30 = SELECTED_STOCKS_20 + SELECTED_ETFS_10
output_file = "selected_etfs_and_stocks_30.py"

stock_stats = {
    'avg_annual_return': np.mean([r['annual_return'] for r in stock_results]) if stock_results else 0,
    'avg_sharpe': np.mean([r['sharpe'] for r in stock_results]) if stock_results else 0,
    'avg_max_drawdown': np.mean([r['max_drawdown'] for r in stock_results]) if stock_results else 0,
}

etf_stats = {
    'avg_annual_return': np.mean([r['annual_return'] for r in etf_results]) if etf_results else 0,
    'avg_sharpe': np.mean([r['sharpe'] for r in etf_results]) if etf_results else 0,
    'avg_max_drawdown': np.mean([r['max_drawdown'] for r in etf_results]) if etf_results else 0,
}

content = f'''
SELECTED_STOCKS_20 = {SELECTED_STOCKS_20}
SELECTED_ETFS_10 = {SELECTED_ETFS_10}
SELECTED_TICKERS_30 = SELECTED_STOCKS_20 + SELECTED_ETFS_10
BACKTEST_STATS = {{
    'stocks': {{
        'avg_annual_return': {stock_stats['avg_annual_return']:.4f},
        'avg_sharpe': {stock_stats['avg_sharpe']:.4f},
        'avg_max_drawdown': {stock_stats['avg_max_drawdown']:.4f},
    }},
    'etfs': {{
        'avg_annual_return': {etf_stats['avg_annual_return']:.4f},
        'avg_sharpe': {etf_stats['avg_sharpe']:.4f},
        'avg_max_drawdown': {etf_stats['avg_max_drawdown']:.4f},
    }}
}}
'''

with open(output_file, 'w') as f:
    f.write(content)