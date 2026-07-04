import numpy as np
import pandas as pd

# ---- 辅助函数 ----
def cs_zscore(gen, x: pd.DataFrame, clip: float=None) -> pd.DataFrame:
    """截面 z-score 标准化."""
    mu = x.mean(axis=1)
    sigma = x.std(axis=1).replace(0, np.nan)
    result = x.sub(mu, axis=0).div(sigma, axis=0)
    if clip is not None:
        result = result.clip(-clip, clip)
    return result

def vol_regime(gen, window: int=60) -> pd.Series:
    """
        波动率区间 (0~1, 市场级).
        高值 = 当前处于历史高波动区间.
        """
    mkt_ret = gen.close.pct_change().mean(axis=1)
    vol = mkt_ret.rolling(window, min_periods=20).std() * np.sqrt(252)
    return vol.rank(pct=True).fillna(0.5)

def _broadcast(gen, market_series: pd.Series) -> pd.DataFrame:
    """广播市场级Series到个股DataFrame."""
    return pd.DataFrame({c: market_series.values for c in gen.close.columns}, index=market_series.index)

# ---- 因子函数 ----
def compute(gen, params):
    """
    流动性自适应: 市场流动性充裕 → 动量因子有效(资金推动趋势)。
    流动性枯竭 → 反转因子有效(流动性冲击后的反弹)。
    流动性度量: Amihud非流动性(市场级) → 高 = 流动性差。
    使用 vol_regime 作为辅助信号: 高波动通常伴随流动性枯竭。
    """
    win1 = int(params.get('win1', 60))
    win2 = int(params.get('win2', 20))
    win3 = int(params.get('win3', 5))
    ret = gen.close.pct_change(1)
    min_p = max(30, win1 // 2)
    mom = ret.rolling(win1, min_periods=min_p).mean() / (ret.rolling(win1, min_periods=min_p).std() + 1e-06)
    ret_abs = ret.abs()
    dollar_vol = gen.close * gen.volume
    illiq = ret_abs / (dollar_vol + 1e-06)
    illiq_ma = illiq.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    reversal = cs_zscore(gen, illiq_ma)
    mkt_illiq = illiq.mean(axis=1).rolling(win1, min_periods=min_p).mean()
    mkt_illiq_pct = mkt_illiq.rank(pct=True).fillna(0.5)
    vol = vol_regime(gen, window=win1)
    switch_signal = 1 - 0.5 * mkt_illiq_pct - 0.5 * vol
    sw = _broadcast(gen, switch_signal.clip(0, 1))
    raw = sw * cs_zscore(gen, mom) + (1 - sw) * reversal
    return cs_zscore(gen, raw).ewm(span=win3, min_periods=max(3, win3 // 2)).mean()
