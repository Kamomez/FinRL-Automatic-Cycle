# Automatic-Cycle-Version with etf

This version added etf to the portfolio, select 20 stocks from SP500 and 10 etfs from yahoo etf list.

Retrain it weekly or you can retrain it daily, it will use data from last 8 dates, 6 days for training and 2 days for testing.

Run it once, and it will continue running in the background indefinitely.

# What changed

added yahoo_etfs.py

added select_30.py

train.py --> train_6+2.py

trade.py --> paper_trading_30.py

