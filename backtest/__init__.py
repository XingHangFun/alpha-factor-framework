"""
Alpha Factor Backtesting Framework
===================================
独立的因子回测框架，通过 pickle 文件加载股票数据，对 Alpha 因子进行回测。

Usage:
    from backtest import BacktestEnv, FactorBacktestEngine

    env = BacktestEnv('/path/to/data.pkl')
    engine = FactorBacktestEngine(env, n_select=200, rebalance_freq='M')
    engine.add_factor(my_factor, params={'win1': 60, 'win2': 5}, name='MyFactor')
    engine.run(start_date='2024-01-01', end_date='2025-12-31')
    engine.summary()
"""

from .env import BacktestEnv
from .engine import FactorBacktestEngine
from .stability import ParamStabilityTester, param_range
from .metrics import (
    annual_return,
    annual_volatility,
    sharpe_ratio,
    calmar_ratio,
    max_drawdown,
    win_rate,
    information_ratio,
    excess_return,
    turnover,
    ic_summary,
)

__all__ = [
    'BacktestEnv',
    'FactorBacktestEngine',
    'ParamStabilityTester',
    'param_range',
    'annual_return',
    'annual_volatility',
    'sharpe_ratio',
    'calmar_ratio',
    'max_drawdown',
    'win_rate',
    'information_ratio',
    'excess_return',
    'turnover',
    'ic_summary',
]
