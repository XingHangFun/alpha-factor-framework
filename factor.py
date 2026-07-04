"""
20 因子集合 — 中证1000 全市场选200, 月度调仓.
按策略类型分组: MR (8个反转), MT (8个趋势动量), MTR (4个自适应混合).
每个因子附经济学直觉, 遵循 rule_v11 命名与参数规范.
"""

import numpy as np
import pandas as pd


# ============================================================================
# MR — 回归震荡类 (Mean Reversion / Reversal)
# ============================================================================

def factor_20260627_fanxinghang_MR_ExtremeRetReversal(self, win1=5, win2=5):
    """极端收益反转: 短期赢家回撤, 输家反弹. A股散户追涨杀跌后均值回复."""
    ret = self.close.pct_change(1)
    short_ret = ret.rolling(win1, min_periods=max(3, win1 // 2)).mean()
    raw = -short_ret
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MR_TurnoverShockReversal(self, win1=60, win2=5, win3=5):
    """换手率冲击反转: 换手率异常飙升 = 注意力驱动交易, 随关注度消退而反转."""
    turnover = self.total_turnover / (self.market_cap + 1e-06)
    ma = turnover.rolling(win1, min_periods=max(30, win1 // 2)).mean()
    std = turnover.rolling(win1, min_periods=max(30, win1 // 2)).std()
    shock = (turnover - ma) / (std + 1e-06)
    raw = -shock.rolling(win2, min_periods=3).mean()
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win3)


def factor_20260627_fanxinghang_MR_IlliquidityPremium(self, win1=20, win2=5):
    """非流动性溢价: Amihud 度量 — 单位成交额引起的价格冲击越大, 流动性补偿越高."""
    ret_abs = self.close.pct_change(1).abs()
    dollar_vol = self.close * self.volume
    illiq = ret_abs / (dollar_vol + 1e-06)
    illiq_ma = illiq.rolling(win1, min_periods=max(10, win1 // 2)).mean()
    raw = illiq_ma
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MR_IdioVolReversal(self, win1=60, win2=5):
    """特质波动率反转: 高特质波动 → 彩票型股票 → 散户偏好 → 未来低收益 (Ang 2006)."""
    ret = self.close.pct_change(1)
    mkt_ret = ret.mean(axis=1)
    tracking_err = (ret.sub(mkt_ret, axis=0)
                    .rolling(win1, min_periods=max(20, win1 // 2)).std())
    raw = -tracking_err
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MR_MaxRetReversal(self, win1=20, win2=5):
    """MAX效应反转: 月内最大日收益越高, 未来收益越低 (Bali 2011 彩票偏好)."""
    ret = self.close.pct_change(1)
    max_ret = ret.rolling(win1, min_periods=max(10, win1 // 2)).max()
    raw = -max_ret
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MR_VolumePriceDivergence(self, win1=20, win2=5):
    """量价背离: 价涨量缩 → 上涨乏力; 价跌量增 → 下跌衰竭. 取背离方向的反转信号."""
    price_chg = self.close.pct_change(win1)
    vol_chg = self.volume.rolling(win1, min_periods=max(10, win1 // 2)).mean().pct_change(win1)
    raw = (self.cs_zscore(vol_chg) - self.cs_zscore(price_chg))
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MR_CrashRebound(self, win1=20, win2=5):
    """暴跌反弹: 窗口内最大回撤越深, 短期反弹动力越强. 恐慌性抛售后的均值回复."""
    peak = self.close.rolling(win1, min_periods=max(10, win1 // 2)).max()
    drawdown = self.close / (peak + 1e-06) - 1
    raw = -drawdown
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MR_HighLowReversal(self, win1=20, win2=5):
    """振幅反转: 近期振幅(高低价差)扩大 → 多空分歧加大 → 极端分歧后趋于收敛."""
    range_pct = (self.high - self.low) / (self.close + 1e-06)
    range_ma = range_pct.rolling(win1, min_periods=max(10, win1 // 2)).mean()
    range_chg = range_pct - range_ma
    raw = -range_chg
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win2)


# ============================================================================
# MT — 动量趋势类 (Momentum / Trend)
# ============================================================================

def factor_20260627_fanxinghang_MT_RiskAdjustedMom(self, win1=60, win2=5):
    """风险调整动量: 收益/波动 → Sharpe-like, 区分"稳步涨"和"高波动赌涨"."""
    ret = self.close.pct_change(1)
    mu = ret.rolling(win1, min_periods=max(30, win1 // 2)).mean()
    sigma = ret.rolling(win1, min_periods=max(30, win1 // 2)).std()
    raw = mu / (sigma + 1e-06)
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MT_52WeekHigh(self, win1=250, win2=5):
    """52周新高: 接近一年高点的股票趋势更强. A股趋势市中突破新高的动量延续."""
    peak = self.close.rolling(win1, min_periods=max(60, win1 // 2)).max()
    raw = self.close / (peak + 1e-06)
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MT_EarningsMom(self, win1=120, win2=20, win3=5):
    """盈利动量代理: 价格加速度 — 近期趋势 vs 远期趋势的差值, 捕捉基本面改善斜率."""
    ret = self.close.pct_change(1)
    short_mom = ret.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    long_mom = ret.rolling(win1, min_periods=max(60, win1 // 2)).mean()
    raw = short_mom - long_mom
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win3)


def factor_20260627_fanxinghang_MT_RelStrength(self, win1=20, win2=5):
    """相对强度: RSI风格 — 上涨日平均涨幅 / (上涨+下跌平均幅度). 捕捉趋势持续性."""
    ret = self.close.pct_change(1)
    up = ret.clip(lower=0).rolling(win1, min_periods=max(10, win1 // 2)).mean()
    dn = ret.clip(upper=0).abs().rolling(win1, min_periods=max(10, win1 // 2)).mean()
    raw = up / (up + dn + 1e-06)
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MT_IndRelStrength(self, win1=20, win2=5):
    """行业相对强度: 个股收益 - 行业均值收益. 捕捉行业内相对领先者."""
    ret = self.close.pct_change(win1)
    ind_mean = ret.mean(axis=1)
    raw = ret.sub(ind_mean, axis=0)
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win2)


def factor_20260627_fanxinghang_MT_VolumeConfirm(self, win1=20, win2=5, win3=5):
    """量能确认趋势: 放量上涨 + 缩量下跌 = 强趋势, 量能与价格方向一致的股票."""
    ret = self.close.pct_change(1)
    vol_chg = self.volume.pct_change(1)
    vol_ret_confirm = ret.rolling(win1, min_periods=max(10, win1 // 2)).mean() * \
                      vol_chg.rolling(win2, min_periods=max(5, win2 // 2)).mean()
    raw = vol_ret_confirm
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win3)


def factor_20260627_fanxinghang_MT_LowVolUptrend(self, win1=60, win2=5):
    """低波动上行: 收益 rank × 逆波动 rank, 高收益且低波动才能得分, 过滤高波动噪声."""
    ret = self.close.pct_change(1)
    mu = ret.rolling(win1, min_periods=max(30, win1 // 2)).mean()
    sigma = ret.rolling(win1, min_periods=max(30, win1 // 2)).std()
    rank_mu = mu.rank(axis=1, pct=True)
    rank_inv_vol = (1.0 / (sigma + 1e-06)).rank(axis=1, pct=True)
    raw = rank_mu * rank_inv_vol
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win2)


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
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win2)


# ============================================================================
# MTR — 动量/反转自适应混合类 (Momentum-Trend-Reversal)
# ============================================================================

def factor_20260627_fanxinghang_MTR_VolAdaptive(self, win1=60, win2=20, win3=5):
    """波动率自适应: 低波动时追趋势, 高波动时做反转. vol_regime 连续权重混合."""
    ret = self.close.pct_change(1)
    mom = ret.rolling(win1, min_periods=max(30, win1 // 2)).mean()
    reversal = -ret.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    sw = self.vol_regime(window=win1)
    sw_df = self._broadcast(sw)
    raw = (1 - sw_df) * self.cs_zscore(mom) + sw_df * self.cs_zscore(reversal)
    return raw.ewm(span=win3, min_periods=max(3, win3 // 2)).mean()


def factor_20260627_fanxinghang_MTR_CrashProtected(self, win1=60, win2=5, panic_confirm=3):
    """带崩溃保护的动量: 恐慌状态离散切大市值防守, 其余时间追动量."""
    ret = self.close.pct_change(1)
    mom = (ret.rolling(win1, min_periods=max(30, win1 // 2)).mean() /
           (ret.rolling(win1, min_periods=max(30, win1 // 2)).std() + 1e-06))
    mom = self.cs_zscore(mom)
    panic = self.panic_state(window=win1)
    enter_defense = panic > 0.7
    return self.discrete_switch(mom,
                                enter_signal=enter_defense,
                                confirm=panic_confirm)


def factor_20260627_fanxinghang_MTR_RegimeBalanced(self, win1=60, win2=20, win3=5):
    """状态平衡混合: 趋势强度高时偏动量, 趋势强度低时偏反转. 连续动态权重."""
    ret = self.close.pct_change(1)
    mom = ret.rolling(win1, min_periods=max(30, win1 // 2)).mean()
    rev = -ret.rolling(win2, min_periods=max(10, win2 // 2)).mean()
    return self.regime_switch(trend_factor=mom, reversal_factor=rev, window=win1)


def factor_20260627_fanxinghang_MTR_PanicDefense(self, win1=5, win2=40, win3=40,
                                                  panic_confirm=3):
    """恐慌防守反转: 参考 CAPDownsideSemi 逻辑, 加 panic_state 离散防守切换."""
    mv = np.log(self.market_cap.replace(0, np.nan))
    mkt_price = (self.close.pct_change(win1) * mv).div(
        mv.sum(axis=1), axis=0).sum(axis=1)
    dn_ret = np.minimum(mkt_price, 0)
    semi = dn_ret.rolling(win2, min_periods=max(5, win2 // 2)).std()
    direction = semi - semi.rolling(win3, min_periods=max(5, win3 // 2)).mean()
    f = -self.market_cap.mul(direction, axis=0)
    f = self.cs_zscore(f)
    panic = self.panic_state(window=win2)
    return self.discrete_switch(f, enter_signal=panic > 0.7, confirm=panic_confirm)


# ============================================================================
# 因子注册
# ============================================================================

FACTOR_LIST = [
    # MR — 回归震荡类 (8)
    'factor_20260627_fanxinghang_MR_ExtremeRetReversal',
    'factor_20260627_fanxinghang_MR_TurnoverShockReversal',
    'factor_20260627_fanxinghang_MR_IlliquidityPremium',
    'factor_20260627_fanxinghang_MR_IdioVolReversal',
    'factor_20260627_fanxinghang_MR_MaxRetReversal',
    'factor_20260627_fanxinghang_MR_VolumePriceDivergence',
    'factor_20260627_fanxinghang_MR_CrashRebound',
    'factor_20260627_fanxinghang_MR_HighLowReversal',
    # MT — 动量趋势类 (8)
    'factor_20260627_fanxinghang_MT_RiskAdjustedMom',
    'factor_20260627_fanxinghang_MT_52WeekHigh',
    'factor_20260627_fanxinghang_MT_EarningsMom',
    'factor_20260627_fanxinghang_MT_RelStrength',
    'factor_20260627_fanxinghang_MT_IndRelStrength',
    'factor_20260627_fanxinghang_MT_VolumeConfirm',
    'factor_20260627_fanxinghang_MT_LowVolUptrend',
    'factor_20260627_fanxinghang_MT_InfoDiscreteness',
    # MTR — 自适应混合类 (4)
    'factor_20260627_fanxinghang_MTR_VolAdaptive',
    'factor_20260627_fanxinghang_MTR_CrashProtected',
    'factor_20260627_fanxinghang_MTR_RegimeBalanced',
    'factor_20260627_fanxinghang_MTR_PanicDefense',
]
