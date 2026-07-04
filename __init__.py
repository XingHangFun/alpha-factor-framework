"""
F0 — Alpha Factor Research Framework
=====================================
中证 1000 指数增强 — 双轨攻防量化因子体系。

Usage:
    from F0 import FunctionPool
    from F0.backtest import BacktestEnv, FactorBacktestEngine

    env = BacktestEnv('data.pkl')
    engine = FactorBacktestEngine(env, n_select=200, rebalance_freq='M')
"""

from .FunctionPool import FunctionPool

__all__ = ['FunctionPool']
