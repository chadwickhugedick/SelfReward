import pandas as pd
import numpy as np
from src.environment.trading_env import TradingEnvironment


def make_price_series(prices):
    idx = pd.date_range(start='2020-01-01', periods=len(prices), freq='D')
    df = pd.DataFrame({'Close': prices}, index=idx)
    return df


def test_open_short_and_cover():
    prices = [100.0, 95.0, 90.0, 92.0]
    df = make_price_series(prices)
    env = TradingEnvironment(df, window_size=1, initial_capital=1000.0, transaction_cost=0.0)

    obs, _ = env.reset()
    # Step 0: Sell to open short at price 100
    obs, reward, done, trunc, info = env.step(2)
    assert env.current_position == -1
    assert env.shares_held < 0
    # Capital after opening short should have increased by proceeds of short sale
    assert info['capital'] > 1000.0

    # Step 1: hold (do nothing), price drops to 95
    obs, reward, done, trunc, info = env.step(0)
    # unrealized PnL should be positive for short
    pv = info['portfolio_value']
    assert pv > env.initial_capital

    # Step 2: Buy to cover at price 90
    obs, reward, done, trunc, info = env.step(1)
    assert env.current_position == 0
    assert env.shares_held == 0
    # After covering, capital should reflect profit from short
    assert info['capital'] > 1000.0


def test_open_long_and_close():
    # Provide three prices so open (at t=1) and close (at t=2) are both valid
    prices = [95.0, 100.0, 110.0]
    df = make_price_series(prices)
    env = TradingEnvironment(df, window_size=1, initial_capital=1000.0, transaction_cost=0.0)

    obs, _ = env.reset()
    # Step 0: Buy to open long at price 100
    obs, reward, done, trunc, info = env.step(1)
    assert env.current_position == 1
    assert env.shares_held > 0

    # Step 1: Sell to close long at price 110
    obs, reward, done, trunc, info = env.step(2)
    assert env.current_position == 0
    assert env.shares_held == 0
    assert info['capital'] > 1000.0
