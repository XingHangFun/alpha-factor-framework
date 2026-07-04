"""
Auto-generated 20 new factors based on rule_v11.txt V11.1.
Date: 20260703

因子结构:
  MR  (回归震荡类) x8  — 捕捉均值回复、过度反应后的价格修正
  MT  (动量趋势类) x8  — 捕捉趋势延续、价格/量能的方向性信号
  MTR (自动切换类) x4  — 根据市场状态动态切换动量/反转模式

双轨归属由外部配置文件决定，不写入函数名。
"""

import numpy as np
import pandas as pd

# ============================================================
# 辅助函数 (与现有 factors.py 一致, convert_factors.py 自动分析依赖)
# ============================================================

def _broadcast(self, market_series: pd.Series) -> pd.DataFrame:
    """广播市场级Series到个股DataFrame."""
    return pd.DataFrame({c: market_series.values for c in self.close.columns}, index=market_series.index)


def cs_mad_clip(self, x: pd.DataFrame, n_sigma: float=3.0) -> pd.DataFrame:
    """截面 MAD 去极值."""
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
    """截面 winsorize."""
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
    """离散状态机: enter_signal 持续 confirm 天后切到 defense, exit_signal 触发后回到 alpha."""
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
    """因子标准化流水线 — 串联 rule_v11 第二章的三步清洗."""
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
    """恐慌状态 (0~1, 市场级)."""
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
    """MTR (连续版): 自动切换动量/反转模式."""
    if switch_signal is None:
        switch_signal = self.trend_strength(window)
    sw = self._broadcast(switch_signal)
    raw = sw * self.cs_zscore(trend_factor) + (1 - sw) * self.cs_zscore(reversal_factor)
    return self.cs_zscore(raw)


def trend_strength(self, window: int=60) -> pd.Series:
    """趋势强度 (0~1, 市场级)."""
    mkt_ret = self.close.pct_change().mean(axis=1)
    cum = (1 + mkt_ret).rolling(window, min_periods=20).apply(lambda x: x.prod(), raw=True)
    path = mkt_ret.abs().rolling(window, min_periods=20).sum()
    efficiency = (abs(cum - 1) / path.replace(0, np.nan)).fillna(0)
    return efficiency.rank(pct=True).fillna(0.5)


def vol_regime(self, window: int=60) -> pd.Series:
    """波动率区间 (0~1, 市场级)."""
    mkt_ret = self.close.pct_change().mean(axis=1)
    vol = mkt_ret.rolling(window, min_periods=20).std() * np.sqrt(252)
    return vol.rank(pct=True).fillna(0.5)


def market_dispersion(self, window: int=20) -> pd.Series:
    """截面离散度 (市场级)."""
    ret = self.close.pct_change()
    dispersion = ret.std(axis=1).rolling(window, min_periods=10).mean()
    return dispersion.rank(pct=True).fillna(0.5)


# ============================================================
# MR 类因子 (回归震荡类) — 8 个
# ============================================================

def factor_20260703_fanxinghang_MR_SkewnessReversal(self, win1=60, win2=5):
    """
    收益偏度反转: 日收益偏度越高 → 彩票型股票 → 散户过度追捧 → 未来收益反转。
    卖空正偏度(赢家), 做多负偏度(被错杀的输家)。偏度比波动率更能刻画散户的"博彩偏好"。
    A股中正偏度与未来收益显著负相关 (Zheng 2020).
    """
    ret = self.close.pct_change(1)
    mu = ret.rolling(win1, min_periods=max(30, win1 // 2)).mean()
    sigma = ret.rolling(win1, min_periods=max(30, win1 // 2)).std()
    # 滚动偏度 = E[(r-μ)³] / σ³
    skew = ((ret - mu) ** 3).rolling(win1, min_periods=max(30, win1 // 2)).mean() / (sigma ** 3 + 1e-06)
    raw = -skew  # 正偏度 → 反转 → 负信号
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260703_fanxinghang_MR_GapReversal(self, win1=5, win2=5):
    """
    跳空缺口反转: 向上跳空缺口 = 隔夜信息冲击导致开盘价远超前收。
    A股散户倾向于开盘追涨 → 日内回落 → 缺口幅度越大反转概率越高。
    度量: (今开 - 昨收) / 昨收, 取负值做反转。
    """
    gap = (self.open - self.close.shift(1)) / (self.close.shift(1) + 1e-06)
    gap_ma = gap.rolling(win1, min_periods=max(3, win1 // 2)).mean()
    raw = -gap_ma  # 正缺口 → 反转做空
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260703_fanxinghang_MR_OvernightReversal(self, win1=20, win2=5):
    """
    隔夜收益反转: 将日收益拆为隔夜(close→next open)和日内(open→close)。
    隔夜涨幅过大反映散户盘后情绪过热 → 次日回落。隔夜与日内收益负相关
    是A股的稳健异象 (Liu 2021)。信号: 隔夜收益动量取负。
    """
    overnight_ret = self.open / self.close.shift(1) - 1
    intraday_ret = self.close / self.open - 1
    # 隔夜收益占比: 隔夜 / (隔夜绝对值 + 日内绝对值)
    on_mag = overnight_ret.abs().rolling(win1, min_periods=max(10, win1 // 2)).mean()
    id_mag = intraday_ret.abs().rolling(win1, min_periods=max(10, win1 // 2)).mean()
    on_ratio = on_mag / (on_mag + id_mag + 1e-06)
    # 隔夜占比高且隔夜正收益 → 过度乐观 → 反转
    on_sign = overnight_ret.rolling(win1, min_periods=max(10, win1 // 2)).mean()
    raw = -on_sign * on_ratio
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260703_fanxinghang_MR_CAPMAlphaReversal(self, win1=60, win2=5):
    """
    CAPM-Alpha反转: 滚动CAPM回归截距(alpha)。近期高alpha股票被过度追捧，
    市场对其定价过高 → Alpha均值回复。剔除Beta暴露后的纯特质alpha反转。
    """
    ret = self.close.pct_change(1)
    mkt_ret = ret.mean(axis=1)
    # 滚动CAPM: beta = cov(r, mkt) / var(mkt), alpha = mean(r) - beta * mean(mkt)
    min_p = max(30, win1 // 2)
    cov = ((ret.sub(mkt_ret, axis=0)).mul(mkt_ret.sub(mkt_ret.mean()), axis=0)
           .rolling(win1, min_periods=min_p).mean())
    var_mkt = mkt_ret.rolling(win1, min_periods=min_p).var()
    beta = cov.div(var_mkt + 1e-06, axis=0)
    mu_ret = ret.rolling(win1, min_periods=min_p).mean()
    mu_mkt = mkt_ret.rolling(win1, min_periods=min_p).mean()
    alpha = mu_ret.sub(beta.mul(mu_mkt, axis=0), axis=0)
    raw = -alpha  # 正alpha → 反转
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260703_fanxinghang_MR_CoskewnessReversal(self, win1=120, win2=5):
    """
    下行协偏度反转: 个股收益与市场下行波动的协方差。协偏度高 = 市场大跌时
    个股跌得更狠 → 投资者要求更高预期收益作为补偿 (Harvey 2000)。
    做多高协偏度(需补偿)、做空低协偏度。信号: 协偏度本身(不做反转)。
    """
    ret = self.close.pct_change(1)
    mkt_ret = ret.mean(axis=1)
    mkt_dn = mkt_ret.clip(upper=0)
    min_p = max(60, win1 // 2)
    # 协偏度近似: E[(r - μ_r) * (mkt_dn - μ_dn)²]
    mu_ret = ret.rolling(win1, min_periods=min_p).mean()
    mu_dn = mkt_dn.rolling(win1, min_periods=min_p).mean()
    mu_dn2 = (mkt_dn ** 2).rolling(win1, min_periods=min_p).mean()
    numer = ((ret.sub(mu_ret, axis=0)).mul((mkt_dn - mu_dn).abs() * mkt_dn.abs(), axis=0)
             .rolling(win1, min_periods=min_p).mean())
    denom = (ret.rolling(win1, min_periods=min_p).std()
             .mul(mkt_dn.rolling(win1, min_periods=min_p).std() ** 2 + 1e-06, axis=0))
    coskew = numer / (denom + 1e-06)
    raw = coskew  # 高协偏度 → 正信号 (风险补偿)
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260703_fanxinghang_MR_VolOfVolReversal(self, win1=60, win2=20, win3=5):
    """
    波动率之波动反转: 波动率的波动率(vol-of-vol)飙升 = 不确定性骤增 = 恐慌性抛售。
    vol-of-vol极端值后趋于回落，做多高vol-of-vol股票(超跌反弹)。
    二阶波动率比一阶波动率更能刻画尾部不确定性。
    """
    ret = self.close.pct_change(1)
    vol = ret.rolling(win2, min_periods=max(10, win2 // 2)).std()
    vol_of_vol = vol.rolling(win1, min_periods=max(30, win1 // 2)).std()
    # 近期vol-of-vol vs 长期均值 → 偏离越大反弹越强
    vol_of_vol_ma = vol_of_vol.rolling(win1, min_periods=max(30, win1 // 2)).mean()
    raw = vol_of_vol - vol_of_vol_ma  # 正向偏离 → 超跌 → 反弹
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win3)


def factor_20260703_fanxinghang_MR_BetaReversal(self, win1=60, win2=5):
    """
    滚动Beta反转: 高Beta股票在牛市中吸引追涨资金，Beta被推高后趋于均值回复。
    做空高Beta(近期被过度追捧)、做多低Beta(被忽视)。注意: 这里是短期Beta反转，
    与长期"低Beta异象"不同，本因子捕捉的是Beta的时序均值回复。
    """
    ret = self.close.pct_change(1)
    mkt_ret = ret.mean(axis=1)
    min_p = max(30, win1 // 2)
    cov = ((ret.sub(mkt_ret, axis=0)).mul(mkt_ret.sub(mkt_ret.mean()), axis=0)
           .rolling(win1, min_periods=min_p).mean())
    var_mkt = mkt_ret.rolling(win1, min_periods=min_p).var()
    beta = cov.div(var_mkt + 1e-06, axis=0)
    # Beta变化率: 近期beta - 远期beta → beta加速上升的股票做空
    beta_short = cov.rolling(max(20, win1 // 3), min_periods=10).mean().div(var_mkt.rolling(max(20, win1 // 3), min_periods=10).mean() + 1e-06, axis=0)
    beta_chg = beta_short - beta
    raw = -beta_chg  # beta上升过快 → 反转
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260703_fanxinghang_MR_DownsideCorrReversal(self, win1=60, win2=5):
    """
    下行相关性反转: 市场下跌日个股与市场的相关性 vs 上涨日的相关性。
    下跌相关性远高于上涨相关性 = 个股在市场下跌时被恐慌性联动抛售 → 超跌反转。
    区分"真弱"(涨跌都低相关)和"错杀"(仅跌时高相关)。
    """
    ret = self.close.pct_change(1)
    mkt_ret = ret.mean(axis=1)
    min_p = max(30, win1 // 2)
    # 下跌日mask
    dn_mask = mkt_ret < 0
    up_mask = mkt_ret > 0
    # 下跌相关性
    ret_dn = ret[dn_mask].reindex(ret.index)
    mkt_dn = mkt_ret[dn_mask].reindex(mkt_ret.index)
    corr_dn = (ret_dn.sub(ret_dn.rolling(win1, min_periods=min_p).mean(), axis=0)
               .mul(mkt_dn.sub(mkt_dn.rolling(win1, min_periods=min_p).mean()), axis=0)
               .rolling(win1, min_periods=min_p).mean()
               .div(ret_dn.rolling(win1, min_periods=min_p).std()
                    .mul(mkt_dn.rolling(win1, min_periods=min_p).std() + 1e-06, axis=0) + 1e-06))
    # 上涨相关性
    ret_up = ret[up_mask].reindex(ret.index)
    mkt_up = mkt_ret[up_mask].reindex(mkt_ret.index)
    corr_up = (ret_up.sub(ret_up.rolling(win1, min_periods=min_p).mean(), axis=0)
               .mul(mkt_up.sub(mkt_up.rolling(win1, min_periods=min_p).mean()), axis=0)
               .rolling(win1, min_periods=min_p).mean()
               .div(ret_up.rolling(win1, min_periods=min_p).std()
                    .mul(mkt_up.rolling(win1, min_periods=min_p).std() + 1e-06, axis=0) + 1e-06))
    # 下行-上行相关差 (不对称性), 差值越大→越多恐慌联动→反转潜力越大
    corr_asym = (corr_dn.fillna(0) - corr_up.fillna(0)).fillna(0)
    raw = corr_asym  # 正值 = 下跌时更相关 → 错杀 → 反弹
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


# ============================================================
# MT 类因子 (动量趋势类) — 8 个
# ============================================================

def factor_20260703_fanxinghang_MT_VolumePriceTrend(self, win1=20, win2=5, win3=5):
    """
    量价趋势共振: 量价配合是A股最朴素的趋势确认逻辑。
    放量上涨 = 资金真金白银推动 → 趋势可靠; 缩量上涨 = 无量空涨 → 不可持续。
    量价相关系数 × 价格方向 → 既涨且量配合的得分高。
    """
    ret = self.close.pct_change(1)
    vol_chg = self.volume.pct_change(1)
    min_p = max(10, win1 // 2)
    # 量价滚动相关
    mr = ret.rolling(win1, min_periods=min_p).mean()
    mv = vol_chg.rolling(win1, min_periods=min_p).mean()
    cov = ((ret - mr) * (vol_chg - mv)).rolling(win1, min_periods=min_p).mean()
    sr = ret.rolling(win1, min_periods=min_p).std()
    sv = vol_chg.rolling(win1, min_periods=min_p).std()
    corr = cov / (sr * sv + 1e-06)
    # 量价共振 = 相关系数 × 价格方向(sign)
    price_dir = ret.rolling(win1, min_periods=min_p).mean()
    raw = corr * np.sign(price_dir) * price_dir.abs()
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win3)


def factor_20260703_fanxinghang_MT_MomentumAccel(self, win1=60, win2=20, win3=5):
    """
    动量加速度: 动量正在加速的股票 → 趋势尚未衰竭 → 继续持有。
    动量减速(accel<0) → 趋势接近尾声 → 减仓。二阶导比一阶导更早捕捉拐点。
    度量: 短期动量 - 长期动量的变化率。
    """
    ret = self.close.pct_change(1)
    short_mom = ret.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    long_mom = ret.rolling(win1, min_periods=max(30, win1 // 2)).mean()
    # 加速度 = (short_mom / long_mom - 1) 近似, 正=加速
    raw = (short_mom - long_mom) / (long_mom.abs() + 1e-06)
    # 叠加方向: 加速 + 正收益
    direction = np.sign(long_mom)
    raw = raw * direction
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win3)


def factor_20260703_fanxinghang_MT_FractalEfficiency(self, win1=20, win2=60, win3=5):
    """
    分形效率: 两个时间尺度的路径效率交叉验证。
    短窗口效率高 + 长窗口效率也高 → 趋势在不同时间尺度上一贯 → 强趋势。
    仅短窗口效率高 → 可能是噪声假突破。多时间尺度一致性过滤假信号。
    """
    ret = self.close.pct_change(1)
    # 短窗口效率
    short_cum = (1 + ret).rolling(win1, min_periods=max(10, win1 // 2)).apply(lambda x: x.prod(), raw=True)
    short_path = ret.abs().rolling(win1, min_periods=max(10, win1 // 2)).sum()
    short_eff = (abs(short_cum - 1) / (short_path + 1e-06)).fillna(0)
    # 长窗口效率
    long_cum = (1 + ret).rolling(win2, min_periods=max(30, win2 // 2)).apply(lambda x: x.prod(), raw=True)
    long_path = ret.abs().rolling(win2, min_periods=max(30, win2 // 2)).sum()
    long_eff = (abs(long_cum - 1) / (long_path + 1e-06)).fillna(0)
    # 分形效率 = 短效率 × 长效率 → 两者都高才得高分
    direction = np.sign(ret.rolling(win2, min_periods=max(30, win2 // 2)).mean())
    raw = short_eff * long_eff * direction
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win3)


def factor_20260703_fanxinghang_MT_ResidualMomentum(self, win1=60, win2=5):
    """
    特质残差动量: 剔除市场Beta和市值因子后的残差收益动量。
    纯特质趋势不会被市场和风格Beta稀释，捕捉个股独立的基本面改善路径。
    残差 = 原始收益 - (beta * mkt + size_exposure)。只取残差部分的动量。
    """
    ret = self.close.pct_change(1)
    mkt_ret = ret.mean(axis=1)
    min_p = max(30, win1 // 2)
    # 双变量残差化: regress ret on [mkt, log_mcap]
    cap_log = np.log(self.market_cap.replace(0, np.nan))
    cap_ret = cap_log.diff()
    # 简化: 先去市值(截面中性化依赖factor_process), 再去市场Beta
    cov_mkt = ((ret.sub(mkt_ret, axis=0)).mul(mkt_ret.sub(mkt_ret.mean()), axis=0)
               .rolling(win1, min_periods=min_p).mean())
    var_mkt = mkt_ret.rolling(win1, min_periods=min_p).var()
    beta = cov_mkt.div(var_mkt + 1e-06, axis=0)
    residual = ret - beta.mul(mkt_ret, axis=0)
    # 残差动量
    raw = residual.rolling(win1, min_periods=min_p).mean()
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260703_fanxinghang_MT_TrendQuality(self, win1=60, win2=5):
    """
    趋势质量: 分离"稳步涨"(高质量)和"高波动赌涨"(低质量)。
    质量分 = 上涨日占比 × 平均涨幅 / 波动率。上涨天数多、涨幅均匀、波动低的股票
    才是真正的好趋势。单日暴涨拉动的动量 → 质量低 → 不可持续。
    """
    ret = self.close.pct_change(1)
    min_p = max(30, win1 // 2)
    # 上涨日占比
    up_days = (ret > 0).rolling(win1, min_periods=min_p).mean()
    # 平均涨幅 (仅上涨日)
    up_ret = ret.clip(lower=0).rolling(win1, min_periods=min_p).sum() / (
        (ret > 0).rolling(win1, min_periods=min_p).sum() + 1)
    # 下行波动
    dn_std = ret.clip(upper=0).rolling(win1, min_periods=min_p).std()
    # 趋势质量 = 上涨日占比 × 平均涨幅 / (1 + 下行波动)
    raw = up_days * up_ret / (1 + dn_std)
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260703_fanxinghang_MT_IntradayMom(self, win1=20, win2=5):
    """
    日内动量延续: 日内方向(open→close)延续到下一日。
    强势日内=买方主导全天，收盘在高位 → 次日大概率延续。
    度量: 日内涨幅 × 收盘位置 (close-low)/(high-low) → 收盘在高位的阳线。
    """
    intraday_ret = self.close / self.open - 1
    # 收盘位置: 0(收在最低) ~ 1(收在最高)
    hl_range = self.high - self.low
    close_pos = (self.close - self.low) / (hl_range + 1e-06)
    # 日内质量 = 日内收益 × 收盘位置
    intraday_quality = intraday_ret * close_pos
    raw = intraday_quality.rolling(win1, min_periods=max(10, win1 // 2)).mean()
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260703_fanxinghang_MT_CrossSectionalMom(self, win1=60, win2=20, win3=5):
    """
    截面动量稳定性: 个股在截面上排名靠前且排名稳定 → 真龙头。
    不仅看当前排名，还要看排名的持续性。排名忽高忽低 = 噪声。
    度量: win2内每日截面rank的平均值 × (1 - rank波动率)。
    """
    ret = self.close.pct_change(1)
    min_p = max(30, win1 // 2)
    # 长期收益方向
    long_ret = ret.rolling(win1, min_periods=min_p).mean()
    # 短期rank稳定性
    rank_ts = ret.rolling(win2, min_periods=max(10, win2 // 2)).apply(
        lambda s: s.rank(pct=True).iloc[-1], raw=False)
    rank_mean = ret.rolling(win2, min_periods=max(10, win2 // 2)).apply(
        lambda s: s.rank(pct=True).mean(), raw=False)
    rank_std = ret.rolling(win2, min_periods=max(10, win2 // 2)).apply(
        lambda s: s.rank(pct=True).std(), raw=False)
    # 排名稳定性 = rank_mean * (1 - rank_std)
    stability = rank_mean * (1 - rank_std / (rank_std.max() + 1e-06))
    raw = stability * np.sign(long_ret)
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win3)


def factor_20260703_fanxinghang_MT_HighLowBreakout(self, win1=20, win2=60, win3=5):
    """
    高低点突破: 当前价格在N日高低区间的相对位置。
    接近或突破高点 = 强势突破 → 动量延续; 在区间中部 = 盘整。
    度量: (close - low_N) / (high_N - low_N) → 0~1, 接近1=突破。
    """
    high_N = self.high.rolling(win1, min_periods=max(10, win1 // 2)).max()
    low_N = self.low.rolling(win1, min_periods=max(10, win1 // 2)).min()
    # 当前收盘在短期区间位置
    range_pos = (self.close - low_N) / (high_N - low_N + 1e-06)
    # 确认: 长期区间位置 (过滤假突破)
    high_L = self.high.rolling(win2, min_periods=max(30, win2 // 2)).max()
    low_L = self.low.rolling(win2, min_periods=max(30, win2 // 2)).min()
    range_pos_L = (self.close - low_L) / (high_L - low_L + 1e-06)
    # 双重确认: 短期突破 + 长期也在高位
    raw = range_pos * range_pos_L
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win3)


# ============================================================
# MTR 类因子 (自动切换类) — 4 个
# ============================================================

def factor_20260703_fanxinghang_MTR_DispersionAdaptive(self, win1=60, win2=20, win3=5):
    """
    离散度自适应: 截面离散度高 → 个股分化大 → Alpha机会多 → 动量选股有效。
    离散度低 → 同涨同跌 → 个股动量无效 → 切换为反转(均值回复)。
    离散度作为 regime_switch 的连续权重信号。
    """
    ret = self.close.pct_change(1)
    min_p = max(30, win1 // 2)
    # 动量端: 风险调整动量
    mom = ret.rolling(win1, min_periods=min_p).mean() / (ret.rolling(win1, min_periods=min_p).std() + 1e-06)
    # 反转端: 短期反转
    reversal = -ret.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    # 离散度信号: 高→动量, 低→反转
    dispersion = self.market_dispersion(window=win1)
    sw = dispersion  # 高离散度 → 动量权重高
    sw_df = self._broadcast(sw)
    raw = sw_df * self.cs_zscore(mom) + (1 - sw_df) * self.cs_zscore(reversal)
    return raw.ewm(span=win3, min_periods=max(3, win3 // 2)).mean()


def factor_20260703_fanxinghang_MTR_SentimentSwitch(self, win1=60, win2=5, panic_confirm=2):
    """
    情绪硬切换: 恐慌状态 → 切大市值防守(离散硬切换); 正常状态 → 动量进攻。
    与连续混合不同，恐慌时完全切换到底座资产，不留动量暴露。
    极端尾部保护 > 连续切换的平滑性。
    """
    ret = self.close.pct_change(1)
    min_p = max(30, win1 // 2)
    # 动量信号: 路径效率 × 收益方向
    cum = (1 + ret).rolling(win1, min_periods=min_p).apply(lambda x: x.prod(), raw=True)
    path = ret.abs().rolling(win1, min_periods=min_p).sum()
    eff = (abs(cum - 1) / (path + 1e-06)).fillna(0)
    mom = eff * np.sign(ret.rolling(win1, min_periods=min_p).mean())
    mom = self.cs_zscore(mom)
    # 防守端: 低波动大市值
    cap = np.log(self.market_cap.replace(0, np.nan))
    vol_inv = 1.0 / (ret.rolling(win1, min_periods=min_p).std() + 1e-06)
    defense_raw = self.cs_zscore(cap) * 0.5 + self.cs_zscore(vol_inv) * 0.5
    defense = self.cs_zscore(defense_raw)
    # 恐慌触发
    panic = self.panic_state(window=win1)
    return self.discrete_switch(mom, defense_factor=defense,
                                enter_signal=panic > 0.7, confirm=panic_confirm)


def factor_20260703_fanxinghang_MTR_EfficiencySwitch(self, win1=60, win2=20, win3=5):
    """
    效率切换: 趋势效率高 → 市场方向明确 → 动量有效。
    趋势效率低 → 震荡/方向不明 → 反转有效。
    使用 regime_switch 连续混合，切换信号 = 趋势强度。
    """
    ret = self.close.pct_change(1)
    min_p = max(30, win1 // 2)
    # 动量端: 多时间尺度动量复合
    mom_short = ret.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    mom_long = ret.rolling(win1, min_periods=min_p).mean()
    trend_factor = 0.5 * self.cs_zscore(mom_short) + 0.5 * self.cs_zscore(mom_long)
    # 反转端: 短期反转 + 振幅反转
    reversal_ret = -ret.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    range_pct = (self.high - self.low) / (self.close + 1e-06)
    reversal_range = -range_pct.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    reversal_factor = 0.5 * self.cs_zscore(reversal_ret) + 0.5 * self.cs_zscore(reversal_range)
    # 趋势强度驱动切换
    return self.regime_switch(trend_factor=trend_factor, reversal_factor=reversal_factor, window=win1)


def factor_20260703_fanxinghang_MTR_LiquidityRegime(self, win1=60, win2=20, win3=5):
    """
    流动性自适应: 市场流动性充裕 → 动量因子有效(资金推动趋势)。
    流动性枯竭 → 反转因子有效(流动性冲击后的反弹)。
    流动性度量: Amihud非流动性(市场级) → 高 = 流动性差。
    使用 vol_regime 作为辅助信号: 高波动通常伴随流动性枯竭。
    """
    ret = self.close.pct_change(1)
    min_p = max(30, win1 // 2)
    # 动量端
    mom = ret.rolling(win1, min_periods=min_p).mean() / (ret.rolling(win1, min_periods=min_p).std() + 1e-06)
    # 反转端: 非流动性溢价 → 高非流动性的股票应有反转
    ret_abs = ret.abs()
    dollar_vol = self.close * self.volume
    illiq = ret_abs / (dollar_vol + 1e-06)
    illiq_ma = illiq.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    reversal = self.cs_zscore(illiq_ma)
    # 切换信号: 市场级非流动性 + 波动率
    mkt_illiq = illiq.mean(axis=1).rolling(win1, min_periods=min_p).mean()
    mkt_illiq_pct = mkt_illiq.rank(pct=True).fillna(0.5)
    vol = self.vol_regime(window=win1)
    # 流动性差 + 高波动 → 偏反转; 流动性好 + 低波动 → 偏动量
    switch_signal = 1 - 0.5 * mkt_illiq_pct - 0.5 * vol  # 高值=好环境, 偏动量
    sw = self._broadcast(switch_signal.clip(0, 1))
    raw = sw * self.cs_zscore(mom) + (1 - sw) * reversal
    return self.cs_zscore(raw).ewm(span=win3, min_periods=max(3, win3 // 2)).mean()


# ============================================================
# 因子注册
# ============================================================

FACTOR_LIST = [
    # MR 回归震荡类 (8)
    'factor_20260703_fanxinghang_MR_SkewnessReversal',
    'factor_20260703_fanxinghang_MR_GapReversal',
    'factor_20260703_fanxinghang_MR_OvernightReversal',
    'factor_20260703_fanxinghang_MR_CAPMAlphaReversal',
    'factor_20260703_fanxinghang_MR_CoskewnessReversal',
    'factor_20260703_fanxinghang_MR_VolOfVolReversal',
    'factor_20260703_fanxinghang_MR_BetaReversal',
    'factor_20260703_fanxinghang_MR_DownsideCorrReversal',

    # MT 动量趋势类 (8)
    'factor_20260703_fanxinghang_MT_VolumePriceTrend',
    'factor_20260703_fanxinghang_MT_MomentumAccel',
    'factor_20260703_fanxinghang_MT_FractalEfficiency',
    'factor_20260703_fanxinghang_MT_ResidualMomentum',
    'factor_20260703_fanxinghang_MT_TrendQuality',
    'factor_20260703_fanxinghang_MT_IntradayMom',
    'factor_20260703_fanxinghang_MT_CrossSectionalMom',
    'factor_20260703_fanxinghang_MT_HighLowBreakout',

    # MTR 自动切换类 (4)
    'factor_20260703_fanxinghang_MTR_DispersionAdaptive',
    'factor_20260703_fanxinghang_MTR_SentimentSwitch',
    'factor_20260703_fanxinghang_MTR_EfficiencySwitch',
    'factor_20260703_fanxinghang_MTR_LiquidityRegime',
]
