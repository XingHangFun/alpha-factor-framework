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

def market_dispersion(gen, window: int=20) -> pd.Series:
    """
        截面离散度 (市场级).
        高值 = 个股分化大, Alpha机会多; 低值 = 同涨同跌.
        """
    ret = gen.close.pct_change()
    dispersion = ret.std(axis=1).rolling(window, min_periods=10).mean()
    return dispersion.rank(pct=True).fillna(0.5)

def _broadcast(gen, market_series: pd.Series) -> pd.DataFrame:
    """广播市场级Series到个股DataFrame."""
    return pd.DataFrame({c: market_series.values for c in gen.close.columns}, index=market_series.index)

# ---- 因子函数 ----
def compute(gen, params):
    """
    离散度自适应: 截面离散度高 → 个股分化大 → Alpha机会多 → 动量选股有效。
    离散度低 → 同涨同跌 → 个股动量无效 → 切换为反转(均值回复)。
    离散度作为 regime_switch 的连续权重信号。
    """
    win1 = int(params.get('win1', 60))
    win2 = int(params.get('win2', 20))
    win3 = int(params.get('win3', 5))
    ret = gen.close.pct_change(1)
    min_p = max(30, win1 // 2)
    mom = ret.rolling(win1, min_periods=min_p).mean() / (ret.rolling(win1, min_periods=min_p).std() + 1e-06)
    reversal = -ret.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    dispersion = market_dispersion(gen, window=win1)
    sw = dispersion
    sw_df = _broadcast(gen, sw)
    raw = sw_df * cs_zscore(gen, mom) + (1 - sw_df) * cs_zscore(gen, reversal)
    return raw.ewm(span=win3, min_periods=max(3, win3 // 2)).mean()
