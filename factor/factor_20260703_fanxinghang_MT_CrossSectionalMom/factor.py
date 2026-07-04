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

def cs_rank_uniform(gen, x: pd.DataFrame) -> pd.DataFrame:
    """截面 rank 映射到 [-1, 1]."""
    r = x.rank(axis=1, pct=True)
    return (r - 0.5) * 2.0

def cs_winsorize(gen, x: pd.DataFrame, sigma: float=3.0) -> pd.DataFrame:
    """截面 winsorize — 用 mean±sigma*std 截断, 比MAD简洁."""
    mu = x.mean(axis=1)
    std = x.std(axis=1).replace(0, np.nan)
    upper = mu + sigma * std
    lower = mu - sigma * std
    return x.clip(lower=lower, upper=upper, axis=0)

def cs_mad_clip(gen, x: pd.DataFrame, n_sigma: float=3.0) -> pd.DataFrame:
    """截面 MAD 去极值. 常数1.4826将MAD转为σ等价值."""
    median = x.median(axis=1)
    mad = x.sub(median, axis=0).abs().median(axis=1)
    upper = median + n_sigma * 1.4826 * mad
    lower = median - n_sigma * 1.4826 * mad
    return x.clip(lower=lower, upper=upper, axis=0)

def cs_neutralize(gen, y: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """截面OLS中性化: regress y on x, 取残差."""

    def _regress(col_y, col_x):
        mask = col_y.notna() & col_x.notna()
        n = mask.sum()
        if n < 10:
            return col_y
        A = np.column_stack([col_x[mask], np.ones(n)])
        coef = np.linalg.lstsq(A, col_y[mask], rcond=None)[0]
        resid = col_y.copy()
        resid.loc[mask] = col_y[mask] - (coef[0] * col_x[mask] + coef[1])
        return resid
    return y.combine(x, _regress)

def factor_process(gen, factor: pd.DataFrame, outlier: str='mad', outlier_sigma: float=3.0, neutralize: bool=True, neutralize_by: str='market_cap', industry: pd.DataFrame=None, zscore: bool=True, smooth: int=5, rank_output: bool=False, clip: float=3.0, fillna: bool=True) -> pd.DataFrame:
    """
        因子标准化流水线 — 串联 rule_v11 第二章的三步清洗.

        流水线顺序:
          NaN/Inf 清理 → 去极值(outlier) → 中性化(neutralize) → ZScore → 平滑 → Rank → 截断 → 填零

        每一步都可以通过参数关闭, 默认开启全部标准清洗.
        """
    factor = factor.replace([np.inf, -np.inf], np.nan)
    if outlier == 'mad':
        factor = cs_mad_clip(gen, factor, n_sigma=outlier_sigma)
    elif outlier == 'winsorize':
        factor = cs_winsorize(gen, factor, sigma=outlier_sigma)
    if neutralize:
        if neutralize_by == 'market_cap':
            cap = getattr(gen, 'market_cap', None)
            if cap is not None:
                cap_log = np.log(cap.replace(0, np.nan))
                factor = cs_neutralize(gen, factor, cap_log)
        elif neutralize_by == 'industry' and industry is not None:
            for ind in industry.stack().unique():
                mask = industry == ind
                if mask.sum().sum() < 20:
                    continue
                sector_mean = factor[mask].mean(axis=1)
                factor = factor.sub(sector_mean, axis=0).where(mask, factor)
    if zscore:
        factor = cs_zscore(gen, factor)
    if smooth > 0:
        factor = factor.ewm(span=smooth, min_periods=max(3, smooth // 2), adjust=False).mean()
    if rank_output:
        factor = cs_rank_uniform(gen, factor)
    if clip is not None and clip > 0:
        factor = factor.clip(-clip, clip)
    if fillna:
        factor = factor.fillna(0)
    return factor

# ---- 因子函数 ----
def compute(gen, params):
    """
    截面动量稳定性: 个股在截面上排名靠前且排名稳定 → 真龙头。
    不仅看当前排名，还要看排名的持续性。排名忽高忽低 = 噪声。
    度量: win2内每日截面rank的平均值 × (1 - rank波动率)。
    """
    win1 = int(params.get('win1', 60))
    win2 = int(params.get('win2', 20))
    win3 = int(params.get('win3', 5))
    ret = gen.close.pct_change(1)
    min_p = max(30, win1 // 2)
    long_ret = ret.rolling(win1, min_periods=min_p).mean()
    rank_mean = ret.rolling(win2, min_periods=max(10, win2 // 2)).apply(lambda s: s.rank(pct=True).mean(), raw=False)
    rank_std = ret.rolling(win2, min_periods=max(10, win2 // 2)).apply(lambda s: s.rank(pct=True).std(), raw=False)
    stability = rank_mean * (1 - rank_std / (rank_std.max() + 1e-06))
    raw = stability * np.sign(long_ret)
    return factor_process(gen, raw, outlier='mad', neutralize=True, zscore=True, smooth=win3)
