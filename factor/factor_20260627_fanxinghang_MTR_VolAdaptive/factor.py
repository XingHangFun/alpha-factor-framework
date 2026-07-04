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

def vol_regime(gen, window: int=60) -> pd.Series:
    """
        波动率区间 (0~1, 市场级).
        高值 = 当前处于历史高波动区间.
        """
    mkt_ret = gen.close.pct_change().mean(axis=1)
    vol = mkt_ret.rolling(window, min_periods=20).std() * np.sqrt(252)
    return vol.rank(pct=True).fillna(0.5)

# ---- 因子函数 ----
def compute(gen, params):
    """波动率自适应: 低波动时追趋势, 高波动时做反转. vol_regime 连续权重混合."""
    win1 = int(params.get('win1', 60))
    win2 = int(params.get('win2', 20))
    win3 = int(params.get('win3', 5))
    ret = gen.close.pct_change(1)
    mom = ret.rolling(win1, min_periods=max(30, win1 // 2)).mean()
    reversal = -ret.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    sw = vol_regime(gen, window=win1)
    sw_df = _broadcast(gen, sw)
    raw = (1 - sw_df) * cs_zscore(gen, mom) + sw_df * cs_zscore(gen, reversal)
    return raw.ewm(span=win3, min_periods=max(3, win3 // 2)).mean()
