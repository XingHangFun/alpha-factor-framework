"""
40 Alpha factors — merged from V1 (20260627) + V2 (20260703).
Date: 20260704

因子结构:
  MR  (回归震荡类) x16 — 捕捉均值回复、过度反应后的价格修正
  MT  (动量趋势类) x16 — 捕捉趋势延续、价格/量能的方向性信号
  MTR (自动切换类) x8  — 根据市场状态动态切换动量/反转模式

辅助函数统一从 FunctionPool.py 获取, convert_factors.py 通过
--include-helpers-from FunctionPool.py 在 py→folders 转换时自动引入。
"""

import numpy as np
import pandas as pd


# ============================================================
# MR 类因子 (回归震荡类) — 16 个
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


def factor_20260703_fanxinghang_MR_SkewnessReversal(self, win1=60, win2=5):
    """
    收益偏度反转: 日收益偏度越高 → 彩票型股票 → 散户过度追捧 → 未来收益反转。
    卖空正偏度(赢家), 做多负偏度(被错杀的输家)。偏度比波动率更能刻画散户的"博彩偏好"。
    A股中正偏度与未来收益显著负相关 (Zheng 2020).
    """
    ret = self.close.pct_change(1)
    mu = ret.rolling(win1, min_periods=max(30, win1 // 2)).mean()
    sigma = ret.rolling(win1, min_periods=max(30, win1 // 2)).std()
    skew = ((ret - mu) ** 3).rolling(win1, min_periods=max(30, win1 // 2)).mean() / (sigma ** 3 + 1e-06)
    raw = -skew
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260703_fanxinghang_MR_GapReversal(self, win1=5, win2=5):
    """
    跳空缺口反转: 向上跳空缺口 = 隔夜信息冲击导致开盘价远超前收。
    A股散户倾向于开盘追涨 → 日内回落 → 缺口幅度越大反转概率越高。
    度量: (今开 - 昨收) / 昨收, 取负值做反转。
    """
    gap = (self.open - self.close.shift(1)) / (self.close.shift(1) + 1e-06)
    gap_ma = gap.rolling(win1, min_periods=max(3, win1 // 2)).mean()
    raw = -gap_ma
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260703_fanxinghang_MR_OvernightReversal(self, win1=20, win2=5):
    """
    隔夜收益反转: 将日收益拆为隔夜(close→next open)和日内(open→close)。
    隔夜涨幅过大反映散户盘后情绪过热 → 次日回落。隔夜与日内收益负相关
    是A股的稳健异象 (Liu 2021)。信号: 隔夜收益动量取负。
    """
    overnight_ret = self.open / self.close.shift(1) - 1
    intraday_ret = self.close / self.open - 1
    on_mag = overnight_ret.abs().rolling(win1, min_periods=max(10, win1 // 2)).mean()
    id_mag = intraday_ret.abs().rolling(win1, min_periods=max(10, win1 // 2)).mean()
    on_ratio = on_mag / (on_mag + id_mag + 1e-06)
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
    min_p = max(30, win1 // 2)
    cov = ((ret.sub(mkt_ret, axis=0)).mul(mkt_ret.sub(mkt_ret.mean()), axis=0)
           .rolling(win1, min_periods=min_p).mean())
    var_mkt = mkt_ret.rolling(win1, min_periods=min_p).var()
    beta = cov.div(var_mkt + 1e-06, axis=0)
    mu_ret = ret.rolling(win1, min_periods=min_p).mean()
    mu_mkt = mkt_ret.rolling(win1, min_periods=min_p).mean()
    alpha = mu_ret.sub(beta.mul(mu_mkt, axis=0), axis=0)
    raw = -alpha
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
    mu_ret = ret.rolling(win1, min_periods=min_p).mean()
    mu_dn = mkt_dn.rolling(win1, min_periods=min_p).mean()
    numer = ((ret.sub(mu_ret, axis=0)).mul((mkt_dn - mu_dn).abs() * mkt_dn.abs(), axis=0)
             .rolling(win1, min_periods=min_p).mean())
    denom = (ret.rolling(win1, min_periods=min_p).std()
             .mul(mkt_dn.rolling(win1, min_periods=min_p).std() ** 2 + 1e-06, axis=0))
    coskew = numer / (denom + 1e-06)
    raw = coskew
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
    vol_of_vol_ma = vol_of_vol.rolling(win1, min_periods=max(30, win1 // 2)).mean()
    raw = vol_of_vol - vol_of_vol_ma
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
    beta_short = cov.rolling(max(20, win1 // 3), min_periods=10).mean().div(
        var_mkt.rolling(max(20, win1 // 3), min_periods=10).mean() + 1e-06, axis=0)
    beta_chg = beta_short - beta
    raw = -beta_chg
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
    dn_mask = mkt_ret < 0
    up_mask = mkt_ret > 0
    ret_dn = ret[dn_mask].reindex(ret.index)
    mkt_dn = mkt_ret[dn_mask].reindex(mkt_ret.index)
    corr_dn = (ret_dn.sub(ret_dn.rolling(win1, min_periods=min_p).mean(), axis=0)
               .mul(mkt_dn.sub(mkt_dn.rolling(win1, min_periods=min_p).mean()), axis=0)
               .rolling(win1, min_periods=min_p).mean()
               .div(ret_dn.rolling(win1, min_periods=min_p).std()
                    .mul(mkt_dn.rolling(win1, min_periods=min_p).std() + 1e-06, axis=0) + 1e-06))
    ret_up = ret[up_mask].reindex(ret.index)
    mkt_up = mkt_ret[up_mask].reindex(mkt_ret.index)
    corr_up = (ret_up.sub(ret_up.rolling(win1, min_periods=min_p).mean(), axis=0)
               .mul(mkt_up.sub(mkt_up.rolling(win1, min_periods=min_p).mean()), axis=0)
               .rolling(win1, min_periods=min_p).mean()
               .div(ret_up.rolling(win1, min_periods=min_p).std()
                    .mul(mkt_up.rolling(win1, min_periods=min_p).std() + 1e-06, axis=0) + 1e-06))
    corr_asym = (corr_dn.fillna(0) - corr_up.fillna(0)).fillna(0)
    raw = corr_asym
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


# ============================================================
# MT 类因子 (动量趋势类) — 16 个
# ============================================================

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
    vol_ret_confirm = (ret.rolling(win1, min_periods=max(10, win1 // 2)).mean()
                       * vol_chg.rolling(win2, min_periods=max(5, win2 // 2)).mean())
    raw = vol_ret_confirm
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win3)


def factor_20260703_fanxinghang_MT_VolumePriceTrend(self, win1=20, win2=5, win3=5):
    """
    量价趋势共振: 量价配合是A股最朴素的趋势确认逻辑。
    放量上涨 = 资金真金白银推动 → 趋势可靠; 缩量上涨 = 无量空涨 → 不可持续。
    量价相关系数 × 价格方向 → 既涨且量配合的得分高。
    """
    ret = self.close.pct_change(1)
    vol_chg = self.volume.pct_change(1)
    min_p = max(10, win1 // 2)
    mr = ret.rolling(win1, min_periods=min_p).mean()
    mv = vol_chg.rolling(win1, min_periods=min_p).mean()
    cov = ((ret - mr) * (vol_chg - mv)).rolling(win1, min_periods=min_p).mean()
    sr = ret.rolling(win1, min_periods=min_p).std()
    sv = vol_chg.rolling(win1, min_periods=min_p).std()
    corr = cov / (sr * sv + 1e-06)
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
    raw = (short_mom - long_mom) / (long_mom.abs() + 1e-06)
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
    short_cum = (1 + ret).rolling(win1, min_periods=max(10, win1 // 2)).apply(lambda x: x.prod(), raw=True)
    short_path = ret.abs().rolling(win1, min_periods=max(10, win1 // 2)).sum()
    short_eff = (abs(short_cum - 1) / (short_path + 1e-06)).fillna(0)
    long_cum = (1 + ret).rolling(win2, min_periods=max(30, win2 // 2)).apply(lambda x: x.prod(), raw=True)
    long_path = ret.abs().rolling(win2, min_periods=max(30, win2 // 2)).sum()
    long_eff = (abs(long_cum - 1) / (long_path + 1e-06)).fillna(0)
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
    cov_mkt = ((ret.sub(mkt_ret, axis=0)).mul(mkt_ret.sub(mkt_ret.mean()), axis=0)
               .rolling(win1, min_periods=min_p).mean())
    var_mkt = mkt_ret.rolling(win1, min_periods=min_p).var()
    beta = cov_mkt.div(var_mkt + 1e-06, axis=0)
    residual = ret - beta.mul(mkt_ret, axis=0)
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
    up_days = (ret > 0).rolling(win1, min_periods=min_p).mean()
    up_ret = ret.clip(lower=0).rolling(win1, min_periods=min_p).sum() / (
        (ret > 0).rolling(win1, min_periods=min_p).sum() + 1)
    dn_std = ret.clip(upper=0).rolling(win1, min_periods=min_p).std()
    raw = up_days * up_ret / (1 + dn_std)
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win2)


def factor_20260703_fanxinghang_MT_IntradayMom(self, win1=20, win2=5):
    """
    日内动量延续: 日内方向(open→close)延续到下一日。
    强势日内=买方主导全天，收盘在高位 → 次日大概率延续。
    度量: 日内涨幅 × 收盘位置 (close-low)/(high-low) → 收盘在高位的阳线。
    """
    intraday_ret = self.close / self.open - 1
    hl_range = self.high - self.low
    close_pos = (self.close - self.low) / (hl_range + 1e-06)
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
    long_ret = ret.rolling(win1, min_periods=min_p).mean()
    rank_mean = ret.rolling(win2, min_periods=max(10, win2 // 2)).apply(
        lambda s: s.rank(pct=True).mean(), raw=False)
    rank_std = ret.rolling(win2, min_periods=max(10, win2 // 2)).apply(
        lambda s: s.rank(pct=True).std(), raw=False)
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
    range_pos = (self.close - low_N) / (high_N - low_N + 1e-06)
    high_L = self.high.rolling(win2, min_periods=max(30, win2 // 2)).max()
    low_L = self.low.rolling(win2, min_periods=max(30, win2 // 2)).min()
    range_pos_L = (self.close - low_L) / (high_L - low_L + 1e-06)
    raw = range_pos * range_pos_L
    return self.factor_process(raw, outlier='mad', neutralize=True, zscore=True, smooth=win3)


# ============================================================
# MTR 类因子 (自动切换类) — 8 个
# ============================================================

def factor_20260627_fanxinghang_MTR_CrashProtected(self, win1=60, win2=5, panic_confirm=3):
    """带崩溃保护的动量: 恐慌状态离散切大市值防守, 其余时间追动量."""
    ret = self.close.pct_change(1)
    mom = (ret.rolling(win1, min_periods=max(30, win1 // 2)).mean()
           / (ret.rolling(win1, min_periods=max(30, win1 // 2)).std() + 1e-06))
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


def factor_20260703_fanxinghang_MTR_DispersionAdaptive(self, win1=60, win2=20, win3=5):
    """
    离散度自适应: 截面离散度高 → 个股分化大 → Alpha机会多 → 动量选股有效。
    离散度低 → 同涨同跌 → 个股动量无效 → 切换为反转(均值回复)。
    离散度作为 regime_switch 的连续权重信号。
    """
    ret = self.close.pct_change(1)
    min_p = max(30, win1 // 2)
    mom = ret.rolling(win1, min_periods=min_p).mean() / (ret.rolling(win1, min_periods=min_p).std() + 1e-06)
    reversal = -ret.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    dispersion = self.market_dispersion(window=win1)
    sw = dispersion
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
    cum = (1 + ret).rolling(win1, min_periods=min_p).apply(lambda x: x.prod(), raw=True)
    path = ret.abs().rolling(win1, min_periods=min_p).sum()
    eff = (abs(cum - 1) / (path + 1e-06)).fillna(0)
    mom = eff * np.sign(ret.rolling(win1, min_periods=min_p).mean())
    mom = self.cs_zscore(mom)
    cap = np.log(self.market_cap.replace(0, np.nan))
    vol_inv = 1.0 / (ret.rolling(win1, min_periods=min_p).std() + 1e-06)
    defense_raw = self.cs_zscore(cap) * 0.5 + self.cs_zscore(vol_inv) * 0.5
    defense = self.cs_zscore(defense_raw)
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
    mom_short = ret.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    mom_long = ret.rolling(win1, min_periods=min_p).mean()
    trend_factor = 0.5 * self.cs_zscore(mom_short) + 0.5 * self.cs_zscore(mom_long)
    reversal_ret = -ret.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    range_pct = (self.high - self.low) / (self.close + 1e-06)
    reversal_range = -range_pct.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    reversal_factor = 0.5 * self.cs_zscore(reversal_ret) + 0.5 * self.cs_zscore(reversal_range)
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
    mom = ret.rolling(win1, min_periods=min_p).mean() / (ret.rolling(win1, min_periods=min_p).std() + 1e-06)
    ret_abs = ret.abs()
    dollar_vol = self.close * self.volume
    illiq = ret_abs / (dollar_vol + 1e-06)
    illiq_ma = illiq.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    reversal = self.cs_zscore(illiq_ma)
    mkt_illiq = illiq.mean(axis=1).rolling(win1, min_periods=min_p).mean()
    mkt_illiq_pct = mkt_illiq.rank(pct=True).fillna(0.5)
    vol = self.vol_regime(window=win1)
    switch_signal = 1 - 0.5 * mkt_illiq_pct - 0.5 * vol
    sw = self._broadcast(switch_signal.clip(0, 1))
    raw = sw * self.cs_zscore(mom) + (1 - sw) * reversal
    return self.cs_zscore(raw).ewm(span=win3, min_periods=max(3, win3 // 2)).mean()


# ============================================================
# 因子注册
# ============================================================

FACTOR_LIST = [
    # MR 回归震荡类 (16)
    'factor_20260627_fanxinghang_MR_CrashRebound',
    'factor_20260627_fanxinghang_MR_ExtremeRetReversal',
    'factor_20260627_fanxinghang_MR_HighLowReversal',
    'factor_20260627_fanxinghang_MR_IdioVolReversal',
    'factor_20260627_fanxinghang_MR_IlliquidityPremium',
    'factor_20260627_fanxinghang_MR_MaxRetReversal',
    'factor_20260627_fanxinghang_MR_TurnoverShockReversal',
    'factor_20260627_fanxinghang_MR_VolumePriceDivergence',
    'factor_20260703_fanxinghang_MR_SkewnessReversal',
    'factor_20260703_fanxinghang_MR_GapReversal',
    'factor_20260703_fanxinghang_MR_OvernightReversal',
    'factor_20260703_fanxinghang_MR_CAPMAlphaReversal',
    'factor_20260703_fanxinghang_MR_CoskewnessReversal',
    'factor_20260703_fanxinghang_MR_VolOfVolReversal',
    'factor_20260703_fanxinghang_MR_BetaReversal',
    'factor_20260703_fanxinghang_MR_DownsideCorrReversal',

    # MT 动量趋势类 (16)
    'factor_20260627_fanxinghang_MT_52WeekHigh',
    'factor_20260627_fanxinghang_MT_EarningsMom',
    'factor_20260627_fanxinghang_MT_IndRelStrength',
    'factor_20260627_fanxinghang_MT_InfoDiscreteness',
    'factor_20260627_fanxinghang_MT_LowVolUptrend',
    'factor_20260627_fanxinghang_MT_RelStrength',
    'factor_20260627_fanxinghang_MT_RiskAdjustedMom',
    'factor_20260627_fanxinghang_MT_VolumeConfirm',
    'factor_20260703_fanxinghang_MT_VolumePriceTrend',
    'factor_20260703_fanxinghang_MT_MomentumAccel',
    'factor_20260703_fanxinghang_MT_FractalEfficiency',
    'factor_20260703_fanxinghang_MT_ResidualMomentum',
    'factor_20260703_fanxinghang_MT_TrendQuality',
    'factor_20260703_fanxinghang_MT_IntradayMom',
    'factor_20260703_fanxinghang_MT_CrossSectionalMom',
    'factor_20260703_fanxinghang_MT_HighLowBreakout',

    # MTR 自动切换类 (8)
    'factor_20260627_fanxinghang_MTR_CrashProtected',
    'factor_20260627_fanxinghang_MTR_PanicDefense',
    'factor_20260627_fanxinghang_MTR_RegimeBalanced',
    'factor_20260627_fanxinghang_MTR_VolAdaptive',
    'factor_20260703_fanxinghang_MTR_DispersionAdaptive',
    'factor_20260703_fanxinghang_MTR_SentimentSwitch',
    'factor_20260703_fanxinghang_MTR_EfficiencySwitch',
    'factor_20260703_fanxinghang_MTR_LiquidityRegime',
]
