import numpy as np
import pandas as pd

# ---- 辅助函数 ----
def _broadcast(gen, market_series: pd.Series) -> pd.DataFrame:
    """广播市场级Series到个股DataFrame."""
    return pd.DataFrame({c: market_series.values for c in gen.close.columns}, index=market_series.index)

def cs_zscore(gen, x: pd.DataFrame, clip: float=None) -> pd.DataFrame:
    """截面 z-score 标准化."""
    mu = x.mean(axis=1)
    sigma = x.std(axis=1).replace(0, np.nan)
    result = x.sub(mu, axis=0).div(sigma, axis=0)
    if clip is not None:
        result = result.clip(-clip, clip)
    return result

def regime_switch(gen, trend_factor: pd.DataFrame, reversal_factor: pd.DataFrame, switch_signal: pd.Series=None, window: int=60) -> pd.DataFrame:
    """
        MTR (连续版): 自动切换动量/反转模式.
        switch_signal: 趋势强度 (0~1), 连续权重混合.
        默认使用 trend_strength 作为切换信号.
        """
    if switch_signal is None:
        switch_signal = trend_strength(gen, window)
    sw = _broadcast(gen, switch_signal)
    raw = sw * cs_zscore(gen, trend_factor) + (1 - sw) * cs_zscore(gen, reversal_factor)
    return cs_zscore(gen, raw)

def trend_strength(gen, window: int=60) -> pd.Series:
    """
        趋势强度 (0~1, 市场级).
        高值 = 市场处于强趋势状态 (无论方向), 适合动量因子.
        低值 = 震荡市, 适合反转因子.
        """
    mkt_ret = gen.close.pct_change().mean(axis=1)
    cum = (1 + mkt_ret).rolling(window, min_periods=20).apply(lambda x: x.prod(), raw=True)
    path = mkt_ret.abs().rolling(window, min_periods=20).sum()
    efficiency = (abs(cum - 1) / path.replace(0, np.nan)).fillna(0)
    return efficiency.rank(pct=True).fillna(0.5)

# ---- 因子函数 ----
def compute(gen, params):
    """
    效率切换: 趋势效率高 → 市场方向明确 → 动量有效。
    趋势效率低 → 震荡/方向不明 → 反转有效。
    使用 regime_switch 连续混合，切换信号 = 趋势强度。
    """
    win1 = int(params.get('win1', 60))
    win2 = int(params.get('win2', 20))
    win3 = int(params.get('win3', 5))
    ret = gen.close.pct_change(1)
    min_p = max(30, win1 // 2)
    mom_short = ret.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    mom_long = ret.rolling(win1, min_periods=min_p).mean()
    trend_factor = 0.5 * cs_zscore(gen, mom_short) + 0.5 * cs_zscore(gen, mom_long)
    reversal_ret = -ret.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    range_pct = (gen.high - gen.low) / (gen.close + 1e-06)
    reversal_range = -range_pct.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    reversal_factor = 0.5 * cs_zscore(gen, reversal_ret) + 0.5 * cs_zscore(gen, reversal_range)
    return regime_switch(gen, trend_factor=trend_factor, reversal_factor=reversal_factor, window=win1)
