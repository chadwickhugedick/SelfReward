import pandas as pd
import numpy as np
from src.environment.trading_env import TradingEnvironment


def make_price_series(prices):
    idx = pd.date_range(start='2020-01-01', periods=len(prices), freq='D')
    df = pd.DataFrame({'Close': prices}, index=idx)
    return df


def test_transaction_costs_affect_pnl():
    prices = [100.0, 110.0]
    df = make_price_series(prices)
    env = TradingEnvironment(df, window_size=1, initial_capital=1000.0, transaction_cost=0.01)

    obs, _ = env.reset()
    # Open long at 100 (cost includes tx)
    obs, reward, done, trunc, info = env.step(1)
    entry_price = info['entry_price']
    # Close at 110
    obs, reward, done, trunc, info = env.step(2)
    realized = info['last_realized_pnl']
    # Realized PnL should be less than naive (110-100)*shares due to transaction costs
    expected_naive = (110.0 - entry_price) * env.entry_shares if env.entry_shares > 0 else 0.0
    assert realized <= expected_naive + 1e-6


def test_insufficient_capital_prevents_open():
    prices = [1000.0, 1000.0]
    df = make_price_series(prices)
    env = TradingEnvironment(df, window_size=1, initial_capital=10.0, transaction_cost=0.0)

    obs, _ = env.reset()
    # Try to open long but capital too low to buy even one share
    obs, reward, done, trunc, info = env.step(1)
    assert env.current_position == 0
    assert env.shares_held == 0

    # Try to open short but insufficient capital -> should still allow short if logic allows proceeds
    obs, reward, done, trunc, info = env.step(2)
    # With current implementation, shorts are allowed if max_shares > 0 based on capital; expect none
    assert env.current_position == 0 or env.shares_held == 0


def test_multi_step_holding_and_drawdown():
    # Prices go up then crash to create drawdown
    prices = [100.0, 120.0, 130.0, 80.0, 90.0]
    df = make_price_series(prices)
    env = TradingEnvironment(df, window_size=1, initial_capital=1000.0, transaction_cost=0.0)

    obs, _ = env.reset()
    # Open long
    obs, reward, done, trunc, info = env.step(1)
    pv_list = [info['portfolio_value']]
    # Hold for two steps
    obs, reward, done, trunc, info = env.step(0)
    pv_list.append(info['portfolio_value'])
    obs, reward, done, trunc, info = env.step(0)
    pv_list.append(info['portfolio_value'])
    # Price crashes -> update
    obs, reward, done, trunc, info = env.step(0)
    pv_list.append(info['portfolio_value'])

    # Ensure portfolio values show a peak before the crash and final value is lower than peak
    peak = max(pv_list[:-1])
    assert peak >= pv_list[0]
    assert pv_list[-1] < peak
    stats = env.get_portfolio_stats()
    assert 'MDD' in stats
    assert stats['MDD'] >= 0.0
