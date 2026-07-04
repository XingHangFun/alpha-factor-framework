"""
Auto-generated from 20 factor folders.
Date: 20260629
"""

import numpy as np
import pandas as pd

# ============================================================
# 辅助函数
# ============================================================

def _broadcast(self, market_series: pd.Series) -> pd.DataFrame:
    """广播市场级Series到个股DataFrame."""
    return pd.DataFrame({c: market_series.values for c in self.close.columns}, index=market_series.index)


def cs_mad_clip(self, x: pd.DataFrame, n_sigma: float=3.0) -> pd.DataFrame:
    """截面 MAD 去极值. 常数1.4826将MAD转为σ等价值."""
    median = x.median(axis=1)
    mad = x.sub(median, axis=0).abs().median(axis=1)
    upper = median + n_sigma * 1.4826 * mad
    lower = median - n_sigma * 1.4826 * mad
    return x.clip(lower=lower, upper=upper, axis=0)


def cs_neutralize(self, y: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
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


def cs_rank_uniform(self, x: pd.DataFrame) -> pd.DataFrame:
    """截面 rank 映射到 [-1, 1]."""
    r = x.rank(axis=1, pct=True)
    return (r - 0.5) * 2.0


def cs_winsorize(self, x: pd.DataFrame, sigma: float=3.0) -> pd.DataFrame:
    """截面 winsorize — 用 mean±sigma*std 截断, 比MAD简洁."""
    mu = x.mean(axis=1)
    std = x.std(axis=1).replace(0, np.nan)
    upper = mu + sigma * std
    lower = mu - sigma * std
    return x.clip(lower=lower, upper=upper, axis=0)


def cs_zscore(self, x: pd.DataFrame, clip: float=None) -> pd.DataFrame:
    """截面 z-score 标准化."""
    mu = x.mean(axis=1)
    sigma = x.std(axis=1).replace(0, np.nan)
    result = x.sub(mu, axis=0).div(sigma, axis=0)
    if clip is not None:
        result = result.clip(-clip, clip)
    return result


def discrete_switch(self, alpha_factor: pd.DataFrame, defense_factor: pd.DataFrame=None, enter_signal=None, exit_signal=None, confirm: int=1) -> pd.DataFrame:
    """
        离散状态机: enter_signal 持续 confirm 天后切到 defense,
        exit_signal 触发后回到 alpha. 不传 defense 时默认用大市值防守.

        regime_switch 做连续权重混合, 本方法做硬切换 —
        适合尾部危机保护场景, 牺牲连续性换取明确的风控边界.
        """
    if defense_factor is None:
        cap_log = np.log(self.market_cap.replace(0, np.nan))
        defense_factor = self.cs_zscore(cap_log)
    if isinstance(enter_signal, pd.Series):
        enter_signal = self._broadcast(enter_signal)
    if exit_signal is None:
        exit_signal = ~enter_signal
    elif isinstance(exit_signal, pd.Series):
        exit_signal = self._broadcast(exit_signal)
    cols = alpha_factor.columns.intersection(defense_factor.columns).intersection(enter_signal.columns).intersection(exit_signal.columns)
    enter = enter_signal[cols]
    exit_ = exit_signal[cols]
    if confirm > 1:
        enter = enter.rolling(confirm, min_periods=1).sum().ge(confirm)
        exit_ = exit_.rolling(confirm, min_periods=1).sum().ge(confirm)
    state = enter.where(enter, other=(~exit_).where(exit_)).ffill().fillna(False)
    return alpha_factor[cols].where(~state, defense_factor[cols])


def factor_process(self, factor: pd.DataFrame, outlier: str='mad', outlier_sigma: float=3.0, neutralize: bool=True, neutralize_by: str='market_cap', industry: pd.DataFrame=None, zscore: bool=True, smooth: int=5, rank_output: bool=False, clip: float=3.0, fillna: bool=True) -> pd.DataFrame:
    """
        因子标准化流水线 — 串联 rule_v11 第二章的三步清洗.

        流水线顺序:
          NaN/Inf 清理 → 去极值(outlier) → 中性化(neutralize) → ZScore → 平滑 → Rank → 截断 → 填零

        每一步都可以通过参数关闭, 默认开启全部标准清洗.
        """
    factor = factor.replace([np.inf, -np.inf], np.nan)
    if outlier == 'mad':
        factor = self.cs_mad_clip(factor, n_sigma=outlier_sigma)
    elif outlier == 'winsorize':
        factor = self.cs_winsorize(factor, sigma=outlier_sigma)
    if neutralize:
        if neutralize_by == 'market_cap':
            cap = getattr(self, 'market_cap', None)
            if cap is not None:
                cap_log = np.log(cap.replace(0, np.nan))
                factor = self.cs_neutralize(factor, cap_log)
        elif neutralize_by == 'industry' and industry is not None:
            for ind in industry.stack().unique():
                mask = industry == ind
                if mask.sum().sum() < 20:
                    continue
                sector_mean = factor[mask].mean(axis=1)
                factor = factor.sub(sector_mean, axis=0).where(mask, factor)
    if zscore:
        factor = self.cs_zscore(factor)
    if smooth > 0:
        factor = factor.ewm(span=smooth, min_periods=max(3, smooth // 2), adjust=False).mean()
    if rank_output:
        factor = self.cs_rank_uniform(factor)
    if clip is not None and clip > 0:
        factor = factor.clip(-clip, clip)
    if fillna:
        factor = factor.fillna(0)
    return factor


def panic_state(self, window: int=60) -> pd.Series:
    """
        恐慌状态 (0~1, 市场级).
        综合下行波动率占比 + 回撤深度 + Chopiness.
        """
    mkt_ret = self.close.pct_change().mean(axis=1)
    dn = mkt_ret.clip(upper=0)
    dn_vol = dn.rolling(window, min_periods=20).std()
    tv = mkt_ret.rolling(window, min_periods=20).std()
    vol_ratio = (dn_vol / tv.replace(0, np.nan)).fillna(0.5)
    mkt_price = self.close.mean(axis=1)
    dd = (mkt_price / mkt_price.rolling(window, min_periods=20).max() - 1).clip(upper=0)
    am = mkt_ret.abs().rolling(20, min_periods=10).mean()
    rm = mkt_ret.rolling(20, min_periods=10).mean()
    cp = (am / (rm.abs() + am * 0.01)).clip(1, 20)
    cn = 1 - 1 / cp
    p = 0.35 * vol_ratio.rank(pct=True) + 0.35 * (-dd).rank(pct=True) + 0.3 * cn.rank(pct=True)
    return p.fillna(0.5).clip(0, 1).ewm(span=5, min_periods=3).mean()


def regime_switch(self, trend_factor: pd.DataFrame, reversal_factor: pd.DataFrame, switch_signal: pd.Series=None, window: int=60) -> pd.DataFrame:
    """
        MTR (连续版): 自动切换动量/反转模式.
        switch_signal: 趋势强度 (0~1), 连续权重混合.
        默认使用 trend_strength 作为切换信号.
        """
    if switch_signal is None:
        switch_signal = self.trend_strength(window)
    sw = self._broadcast(switch_signal)
    raw = sw * self.cs_zscore(trend_factor) + (1 - sw) * self.cs_zscore(reversal_factor)
    return self.cs_zscore(raw)


def trend_strength(self, window: int=60) -> pd.Series:
    """
        趋势强度 (0~1, 市场级).
        高值 = 市场处于强趋势状态 (无论方向), 适合动量因子.
        低值 = 震荡市, 适合反转因子.
        """
    mkt_ret = self.close.pct_change().mean(axis=1)
    cum = (1 + mkt_ret).rolling(window, min_periods=20).apply(lambda x: x.prod(), raw=True)
    path = mkt_ret.abs().rolling(window, min_periods=20).sum()
    efficiency = (abs(cum - 1) / path.replace(0, np.nan)).fillna(0)
    return efficiency.rank(pct=True).fillna(0.5)


def vol_regime(self, window: int=60) -> pd.Series:
    """
        波动率区间 (0~1, 市场级).
        高值 = 当前处于历史高波动区间.
        """
    mkt_ret = self.close.pct_change().mean(axis=1)
    vol = mkt_ret.rolling(window, min_periods=20).std() * np.sqrt(252)
    return vol.rank(pct=True).fillna(0.5)


# ============================================================
# 因子函数 (20个)
# ============================================================

def factor_20260627_fanxinghang_MR_CrashRebound(self, win1=20, win2=5):
    """暴跌反弹: 窗口内最大回撤越深, 短期反弹动力越强. 恐慌性抛售后的均值回复."""
    peak = self.close.rolling(win1, min_periods=max(10, win1 // 2)).max()
    drawdown = self.close / (peak + 1e-06) - 1
    raw = -drawdown
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MR_ExtremeRetReversal(self, win1=5, win2=5):
    """极端收益反转: 短期赢家回撤, 输家反弹. A股散户追涨杀跌后均值回复."""
    ret = self.close.pct_change(1)
    short_ret = ret.rolling(win1, min_periods=max(3, win1 // 2)).mean()
    raw = -short_ret
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MR_HighLowReversal(self, win1=20, win2=5):
    """振幅反转: 近期振幅(高低价差)扩大 → 多空分歧加大 → 极端分歧后趋于收敛."""
    range_pct = (self.high - self.low) / (self.close + 1e-06)
    range_ma = range_pct.rolling(win1, min_periods=max(10, win1 // 2)).mean()
    range_chg = range_pct - range_ma
    raw = -range_chg
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MR_IdioVolReversal(self, win1=60, win2=5):
    """特质波动率反转: 高特质波动 → 彩票型股票 → 散户偏好 → 未来低收益 (Ang 2006)."""
    ret = self.close.pct_change(1)
    mkt_ret = ret.mean(axis=1)
    tracking_err = ret.sub(mkt_ret, axis=0).rolling(win1, min_periods=max(20, win1 // 2)).std()
    raw = -tracking_err
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MR_IlliquidityPremium(self, win1=20, win2=5):
    """非流动性溢价: Amihud 度量 — 单位成交额引起的价格冲击越大, 流动性补偿越高."""
    ret_abs = self.close.pct_change(1).abs()
    dollar_vol = self.close * self.volume
    illiq = ret_abs / (dollar_vol + 1e-06)
    illiq_ma = illiq.rolling(win1, min_periods=max(10, win1 // 2)).mean()
    raw = illiq_ma
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MR_MaxRetReversal(self, win1=20, win2=5):
    """MAX效应反转: 月内最大日收益越高, 未来收益越低 (Bali 2011 彩票偏好)."""
    ret = self.close.pct_change(1)
    max_ret = ret.rolling(win1, min_periods=max(10, win1 // 2)).max()
    raw = -max_ret
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MR_TurnoverShockReversal(self, win1=60, win2=5, win3=5):
    """换手率冲击反转: 换手率异常飙升 = 注意力驱动交易, 随关注度消退而反转."""
    turnover = self.total_turnover / (self.market_cap + 1e-06)
    ma = turnover.rolling(win1, min_periods=max(30, win1 // 2)).mean()
    std = turnover.rolling(win1, min_periods=max(30, win1 // 2)).std()
    shock = (turnover - ma) / (std + 1e-06)
    raw = -shock.rolling(win2, min_periods=3).mean()
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win3)


def factor_20260627_fanxinghang_MR_VolumePriceDivergence(self, win1=20, win2=5):
    """量价背离: 价涨量缩 → 上涨乏力; 价跌量增 → 下跌衰竭. 取背离方向的反转信号."""
    price_chg = self.close.pct_change(win1)
    vol_chg = self.volume.rolling(win1, min_periods=max(10, win1 // 2)).mean().pct_change(win1)
    raw = self.cs_zscore(vol_chg) - self.cs_zscore(price_chg)
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MTR_CrashProtected(self, win1=60, win2=5, panic_confirm=3):
    """带崩溃保护的动量: 恐慌状态离散切大市值防守, 其余时间追动量."""
    ret = self.close.pct_change(1)
    mom = ret.rolling(win1, min_periods=max(30, win1 // 2)).mean() / (ret.rolling(win1, min_periods=max(30, win1 // 2)).std() + 1e-06)
    mom = self.cs_zscore(mom)
    panic = self.panic_state(window=win1)
    enter_defense = panic > 0.7
    return self.discrete_switch(mom, enter_signal=enter_defense, confirm=panic_confirm)


def factor_20260627_fanxinghang_MTR_PanicDefense(self, win1=5, win2=40, win3=40, panic_confirm=3):
    """恐慌防守反转: 参考 CAPDownsideSemi 逻辑, 加 panic_state 离散防守切换."""
    mv = np.log(self.market_cap.replace(0, np.nan))
    mkt_price = (self.close.pct_change(win1) * mv).div(mv.sum(axis=1), axis=0).sum(axis=1)
    dn_ret = np.minimum(mkt_price, 0)
    semi = dn_ret.rolling(win2, min_periods=max(5, win2 // 2)).std()
    direction = semi - semi.rolling(win3, min_periods=max(5, win3 // 2)).mean()
    f = -self.market_cap.mul(direction, axis=0)
    f = self.cs_zscore(f)
    panic = self.panic_state(window=win2)
    return self.discrete_switch(f, enter_signal=panic > 0.7, confirm=panic_confirm)


def factor_20260627_fanxinghang_MTR_RegimeBalanced(self, win1=60, win2=20, win3=5):
    """状态平衡混合: 趋势强度高时偏动量, 趋势强度低时偏反转. 连续动态权重."""
    ret = self.close.pct_change(1)
    mom = ret.rolling(win1, min_periods=max(30, win1 // 2)).mean()
    rev = -ret.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    return self.regime_switch(trend_factor=mom, reversal_factor=rev, window=win1)


def factor_20260627_fanxinghang_MTR_VolAdaptive(self, win1=60, win2=20, win3=5):
    """波动率自适应: 低波动时追趋势, 高波动时做反转. vol_regime 连续权重混合."""
    ret = self.close.pct_change(1)
    mom = ret.rolling(win1, min_periods=max(30, win1 // 2)).mean()
    reversal = -ret.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    sw = self.vol_regime(window=win1)
    sw_df = self._broadcast(sw)
    raw = (1 - sw_df) * self.cs_zscore(mom) + sw_df * self.cs_zscore(reversal)
    return raw.ewm(span=win3, min_periods=max(3, win3 // 2)).mean()


def factor_20260627_fanxinghang_MT_52WeekHigh(self, win1=250, win2=5):
    """52周新高: 接近一年高点的股票趋势更强. A股趋势市中突破新高的动量延续."""
    peak = self.close.rolling(win1, min_periods=max(60, win1 // 2)).max()
    raw = self.close / (peak + 1e-06)
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MT_EarningsMom(self, win1=120, win2=20, win3=5):
    """盈利动量代理: 价格加速度 — 近期趋势 vs 远期趋势的差值, 捕捉基本面改善斜率."""
    ret = self.close.pct_change(1)
    short_mom = ret.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    long_mom = ret.rolling(win1, min_periods=max(60, win1 // 2)).mean()
    raw = short_mom - long_mom
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win3)


def factor_20260627_fanxinghang_MT_IndRelStrength(self, win1=20, win2=5):
    """行业相对强度: 个股收益 - 行业均值收益. 捕捉行业内相对领先者."""
    ret = self.close.pct_change(win1)
    ind_mean = ret.mean(axis=1)
    raw = ret.sub(ind_mean, axis=0)
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MT_InfoDiscreteness(self, win1=60, win2=5):
    """信息离散度动量: 信息以集中跳跃方式释放的股票, 市场消化不足 → 动量更强 (Da 2014)."""
    ret = self.close.pct_change(1)
    abs_ret = ret.abs()
    max_ret = abs_ret.rolling(win1, min_periods=max(30, win1 // 2)).max()
    sum_ret = abs_ret.rolling(win1, min_periods=max(30, win1 // 2)).sum()
    avg_ret = sum_ret / win1
    discreteness = max_ret / (avg_ret + 1e-06)
    mom_sign = np.sign(ret.rolling(win1, min_periods=max(30, win1 // 2)).mean())
    raw = discreteness * mom_sign
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MT_LowVolUptrend(self, win1=60, win2=5):
    """低波动上行: 收益 rank × 逆波动 rank, 高收益且低波动才能得分, 过滤高波动噪声."""
    ret = self.close.pct_change(1)
    mu = ret.rolling(win1, min_periods=max(30, win1 // 2)).mean()
    sigma = ret.rolling(win1, min_periods=max(30, win1 // 2)).std()
    rank_mu = mu.rank(axis=1, pct=True)
    rank_inv_vol = (1.0 / (sigma + 1e-06)).rank(axis=1, pct=True)
    raw = rank_mu * rank_inv_vol
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MT_RelStrength(self, win1=20, win2=5):
    """相对强度: RSI风格 — 上涨日平均涨幅 / (上涨+下跌平均幅度). 捕捉趋势持续性."""
    ret = self.close.pct_change(1)
    up = ret.clip(lower=0).rolling(win1, min_periods=max(10, win1 // 2)).mean()
    dn = ret.clip(upper=0).abs().rolling(win1, min_periods=max(10, win1 // 2)).mean()
    raw = up / (up + dn + 1e-06)
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MT_RiskAdjustedMom(self, win1=60, win2=5):
    """风险调整动量: 收益/波动 → Sharpe-like, 区分"稳步涨"和"高波动赌涨"."""
    ret = self.close.pct_change(1)
    mu = ret.rolling(win1, min_periods=max(30, win1 // 2)).mean()
    sigma = ret.rolling(win1, min_periods=max(30, win1 // 2)).std()
    raw = mu / (sigma + 1e-06)
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MT_VolumeConfirm(self, win1=20, win2=5, win3=5):
    """量能确认趋势: 放量上涨 + 缩量下跌 = 强趋势, 量能与价格方向一致的股票."""
    ret = self.close.pct_change(1)
    vol_chg = self.volume.pct_change(1)
    vol_ret_confirm = ret.rolling(win1, min_periods=max(10, win1 // 2)).mean() * vol_chg.rolling(win2, min_periods=max(5, win2 // 2)).mean()
    raw = vol_ret_confirm
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win3)


# ============================================================
# 因子注册
# ============================================================

FACTOR_LIST = [
    'factor_20260627_fanxinghang_MR_CrashRebound',
    'factor_20260627_fanxinghang_MR_ExtremeRetReversal',
    'factor_20260627_fanxinghang_MR_HighLowReversal',
    'factor_20260627_fanxinghang_MR_IdioVolReversal',
    'factor_20260627_fanxinghang_MR_IlliquidityPremium',
    'factor_20260627_fanxinghang_MR_MaxRetReversal',
    'factor_20260627_fanxinghang_MR_TurnoverShockReversal',
    'factor_20260627_fanxinghang_MR_VolumePriceDivergence',
    'factor_20260627_fanxinghang_MTR_CrashProtected',
    'factor_20260627_fanxinghang_MTR_PanicDefense',
    'factor_20260627_fanxinghang_MTR_RegimeBalanced',
    'factor_20260627_fanxinghang_MTR_VolAdaptive',
    'factor_20260627_fanxinghang_MT_52WeekHigh',
    'factor_20260627_fanxinghang_MT_EarningsMom',
    'factor_20260627_fanxinghang_MT_IndRelStrength',
    'factor_20260627_fanxinghang_MT_InfoDiscreteness',
    'factor_20260627_fanxinghang_MT_LowVolUptrend',
    'factor_20260627_fanxinghang_MT_RelStrength',
    'factor_20260627_fanxinghang_MT_RiskAdjustedMom',
    'factor_20260627_fanxinghang_MT_VolumeConfirm',
]
