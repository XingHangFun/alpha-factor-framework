"""
绩效指标计算模块
=================
纯函数，接收日收益序列和基准收益序列，返回标量指标或结构化结果。
"""

from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd


# 交易日数量（近似）
TRADING_DAYS_PER_YEAR = 252


def _to_series(returns) -> pd.Series:
    """统一转换为 pd.Series。"""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    return returns.dropna()


# ================================================================
# 基础收益指标
# ================================================================

def annual_return(daily_returns) -> float:
    """
    年化收益率。

    参数
    ----
    daily_returns : array-like
        日收益序列。

    返回
    ----
    float : 年化收益率（小数，如 0.15 = 15%）。
    """
    r = _to_series(daily_returns)
    if len(r) == 0:
        return 0.0
    cumulative = (1 + r).prod()
    n_years = len(r) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return 0.0
    return cumulative ** (1 / n_years) - 1


def annual_volatility(daily_returns) -> float:
    """年化波动率。"""
    r = _to_series(daily_returns)
    if len(r) < 2:
        return 0.0
    return r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def cumulative_return(daily_returns) -> float:
    """累计收益率。"""
    r = _to_series(daily_returns)
    if len(r) == 0:
        return 0.0
    return (1 + r).prod() - 1


# ================================================================
# 风险调整指标
# ================================================================

def sharpe_ratio(daily_returns, rf: float = 0.02) -> float:
    """
    夏普比率。

    参数
    ----
    daily_returns : array-like
        日收益序列。
    rf : float
        无风险利率（年化，默认 2%）。

    返回
    ----
    float : 年化夏普比率。
    """
    r = _to_series(daily_returns)
    if len(r) < 2 or r.std() == 0:
        return 0.0
    excess = r.mean() * TRADING_DAYS_PER_YEAR - rf
    vol = r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return excess / vol if vol > 0 else 0.0


def calmar_ratio(daily_returns) -> float:
    """
    卡尔玛比率 = 年化收益率 / |最大回撤|。

    返回
    ----
    float : Calmar ratio。无回撤时返回 inf。
    """
    r = _to_series(daily_returns)
    ann_ret = annual_return(r)
    mdd = max_drawdown(r)
    if mdd == 0:
        return float('inf') if ann_ret > 0 else 0.0
    return ann_ret / abs(mdd)


def sortino_ratio(daily_returns, rf: float = 0.02) -> float:
    """
    索提诺比率 — 使用下行波动率替代总波动率。
    """
    r = _to_series(daily_returns)
    if len(r) < 2:
        return 0.0
    excess = r.mean() * TRADING_DAYS_PER_YEAR - rf
    downside = r[r < 0]
    if len(downside) < 2:
        return 0.0
    down_vol = downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return excess / down_vol if down_vol > 0 else 0.0


def max_drawdown(daily_returns) -> float:
    """
    最大回撤（小数，如 -0.25 = -25%）。

    返回
    ----
    float : 负值表示回撤幅度。
    """
    r = _to_series(daily_returns)
    if len(r) == 0:
        return 0.0
    cumulative = (1 + r).cumprod()
    peak = cumulative.expanding().max()
    dd = (cumulative / peak - 1)
    return dd.min()


def max_drawdown_detail(daily_returns) -> dict:
    """
    最大回撤详细信息。

    返回
    ----
    dict : keys = 'mdd', 'peak_date', 'trough_date', 'recovery_date', 'duration_days'
    """
    r = _to_series(daily_returns)
    if len(r) == 0:
        return {'mdd': 0.0, 'peak_date': None, 'trough_date': None,
                'recovery_date': None, 'duration_days': 0}

    cumulative = (1 + r).cumprod()
    peak = cumulative.expanding().max()
    dd = cumulative / peak - 1

    trough_idx = dd.idxmin()
    mdd_val = dd.min()

    # 找到峰顶日期（回撤开始前的高点）
    peak_idx = cumulative.loc[:trough_idx].idxmax()

    # 找到恢复日期（净值回到峰顶以上的第一个日期）
    peak_val = cumulative.loc[peak_idx]
    recovered = cumulative.loc[trough_idx:] >= peak_val
    recovery_idx = recovered[recovered].index[0] if recovered.any() else None
    duration = (recovery_idx - peak_idx).days if recovery_idx else (r.index[-1] - peak_idx).days

    return {
        'mdd': float(mdd_val),
        'peak_date': str(peak_idx.date()) if hasattr(peak_idx, 'date') else str(peak_idx),
        'trough_date': str(trough_idx.date()) if hasattr(trough_idx, 'date') else str(trough_idx),
        'recovery_date': str(recovery_idx.date()) if recovery_idx and hasattr(recovery_idx, 'date') else str(recovery_idx) if recovery_idx else '未恢复',
        'duration_days': duration,
    }


# ================================================================
# 胜率与概率指标
# ================================================================

def win_rate(daily_returns) -> float:
    """日胜率（正收益天数占比）。"""
    r = _to_series(daily_returns)
    if len(r) == 0:
        return 0.0
    return (r > 0).mean()


def profit_loss_ratio(daily_returns) -> float:
    """盈亏比 = 平均盈利 / |平均亏损|。"""
    r = _to_series(daily_returns)
    gains = r[r > 0]
    losses = r[r < 0]
    if len(losses) == 0:
        return float('inf') if len(gains) > 0 else 0.0
    if len(gains) == 0:
        return 0.0
    return gains.mean() / abs(losses.mean())


# ================================================================
# 相对基准指标
# ================================================================

def excess_return(daily_returns, benchmark_returns) -> float:
    """年化超额收益 vs 基准。"""
    r = _to_series(daily_returns)
    b = _to_series(benchmark_returns)
    common = r.index.intersection(b.index)
    if len(common) == 0:
        return 0.0
    return annual_return(r.loc[common]) - annual_return(b.loc[common])


def information_ratio(daily_returns, benchmark_returns) -> float:
    """信息比率 = 年化超额收益 / 跟踪误差。"""
    r = _to_series(daily_returns)
    b = _to_series(benchmark_returns)
    common = r.index.intersection(b.index)
    if len(common) < 2:
        return 0.0
    diff = r.loc[common] - b.loc[common]
    ann_excess = diff.mean() * TRADING_DAYS_PER_YEAR
    te = diff.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return ann_excess / te if te > 0 else 0.0


def tracking_error(daily_returns, benchmark_returns) -> float:
    """年化跟踪误差。"""
    r = _to_series(daily_returns)
    b = _to_series(benchmark_returns)
    common = r.index.intersection(b.index)
    if len(common) < 2:
        return 0.0
    diff = r.loc[common] - b.loc[common]
    return diff.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


# ================================================================
# 因子评价指标
# ================================================================

def ic_summary(factor_values: pd.DataFrame, forward_returns: pd.DataFrame) -> dict:
    """
    计算因子 IC（信息系数）相关指标。

    IC = 截面 Spearman rank correlation(factor_score, forward_return)

    参数
    ----
    factor_values : pd.DataFrame
        因子值 (index=date, columns=stock)。
    forward_returns : pd.DataFrame
        对应的未来收益 (同 shape)。

    返回
    ----
    dict : keys='ic_mean', 'ic_std', 'icir', 'rank_ic_mean', 'ic_win_rate', 'ic_series'
    """
    common_dates = factor_values.index.intersection(forward_returns.index)
    if len(common_dates) < 2:
        return {'ic_mean': 0, 'ic_std': 0, 'icir': 0, 'rank_ic_mean': 0,
                'ic_win_rate': 0, 'ic_series': pd.Series(dtype=float)}

    ic_list = []
    rank_ic_list = []

    for dt in common_dates:
        fv = factor_values.loc[dt].dropna()
        fr = forward_returns.loc[dt].dropna()
        common = fv.index.intersection(fr.index)
        if len(common) < 10:
            continue

        # Pearson IC
        ic = fv[common].corr(fr[common], method='pearson')
        # Spearman Rank IC
        rank_ic = fv[common].corr(fr[common], method='spearman')

        if not np.isnan(ic):
            ic_list.append(ic)
        if not np.isnan(rank_ic):
            rank_ic_list.append(rank_ic)

    ic_series = pd.Series(ic_list, index=common_dates[:len(ic_list)])

    ic_mean = np.mean(ic_list) if ic_list else 0.0
    ic_std = np.std(ic_list) if ic_list else 0.0
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    rank_ic_mean = np.mean(rank_ic_list) if rank_ic_list else 0.0
    ic_win_rate = (np.array(ic_list) > 0).mean() if ic_list else 0.0

    return {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'rank_ic_mean': rank_ic_mean,
        'ic_win_rate': ic_win_rate,
        'ic_series': ic_series,
    }


def turnover(holdings_history: Dict[str, set]) -> float:
    """
    平均单边换手率。

    参数
    ----
    holdings_history : dict
        {date_str: set(stock_codes)}，每个调仓日的持仓集合。

    返回
    ----
    float : 平均单边换手率（小数，如 0.30 = 30%）。
    """
    if len(holdings_history) < 2:
        return 0.0

    dates = sorted(holdings_history.keys())
    turnovers = []

    for i in range(1, len(dates)):
        prev = holdings_history[dates[i - 1]]
        curr = holdings_history[dates[i]]
        if len(prev) == 0:
            continue
        # 单边换手 = 卖出的比例
        sold = len(prev - curr)
        turnovers.append(sold / len(prev))

    return np.mean(turnovers) if turnovers else 0.0


# ================================================================
# 综合报告
# ================================================================

def full_report(daily_returns, benchmark_returns=None,
                holdings_history: Dict[str, set] = None,
                factor_values: pd.DataFrame = None,
                forward_returns: pd.DataFrame = None,
                rf: float = 0.02) -> dict:
    """
    生成完整的绩效报告。

    返回
    ----
    dict : 包含所有指标的字典。
    """
    r = _to_series(daily_returns)

    report = {
        # 基础指标
        'cumulative_return': cumulative_return(r),
        'annual_return': annual_return(r),
        'annual_volatility': annual_volatility(r),

        # 风险调整
        'sharpe_ratio': sharpe_ratio(r, rf),
        'calmar_ratio': calmar_ratio(r),
        'sortino_ratio': sortino_ratio(r, rf),
        'max_drawdown': max_drawdown(r),
        'mdd_detail': max_drawdown_detail(r),

        # 概率
        'win_rate': win_rate(r),
        'profit_loss_ratio': profit_loss_ratio(r),

        # 交易频率
        'n_trading_days': len(r),
    }

    # 相对基准
    if benchmark_returns is not None:
        b = _to_series(benchmark_returns)
        report['benchmark_annual_return'] = annual_return(b)
        report['excess_return'] = excess_return(r, b)
        report['information_ratio'] = information_ratio(r, b)
        report['tracking_error'] = tracking_error(r, b)

    # 换手率
    if holdings_history is not None:
        report['turnover'] = turnover(holdings_history)

    # IC
    if factor_values is not None and forward_returns is not None:
        report['ic'] = ic_summary(factor_values, forward_returns)

    return report
