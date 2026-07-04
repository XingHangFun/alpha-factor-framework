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

def panic_state(gen, window: int=60) -> pd.Series:
    """
        恐慌状态 (0~1, 市场级).
        综合下行波动率占比 + 回撤深度 + Chopiness.
        """
    mkt_ret = gen.close.pct_change().mean(axis=1)
    dn = mkt_ret.clip(upper=0)
    dn_vol = dn.rolling(window, min_periods=20).std()
    tv = mkt_ret.rolling(window, min_periods=20).std()
    vol_ratio = (dn_vol / tv.replace(0, np.nan)).fillna(0.5)
    mkt_price = gen.close.mean(axis=1)
    dd = (mkt_price / mkt_price.rolling(window, min_periods=20).max() - 1).clip(upper=0)
    am = mkt_ret.abs().rolling(20, min_periods=10).mean()
    rm = mkt_ret.rolling(20, min_periods=10).mean()
    cp = (am / (rm.abs() + am * 0.01)).clip(1, 20)
    cn = 1 - 1 / cp
    p = 0.35 * vol_ratio.rank(pct=True) + 0.35 * (-dd).rank(pct=True) + 0.3 * cn.rank(pct=True)
    return p.fillna(0.5).clip(0, 1).ewm(span=5, min_periods=3).mean()

def discrete_switch(gen, alpha_factor: pd.DataFrame, defense_factor: pd.DataFrame=None, enter_signal=None, exit_signal=None, confirm: int=1) -> pd.DataFrame:
    """
        离散状态机: enter_signal 持续 confirm 天后切到 defense,
        exit_signal 触发后回到 alpha. 不传 defense 时默认用大市值防守.

        regime_switch 做连续权重混合, 本方法做硬切换 —
        适合尾部危机保护场景, 牺牲连续性换取明确的风控边界.
        """
    if defense_factor is None:
        cap_log = np.log(gen.market_cap.replace(0, np.nan))
        defense_factor = cs_zscore(gen, cap_log)
    if isinstance(enter_signal, pd.Series):
        enter_signal = _broadcast(gen, enter_signal)
    if exit_signal is None:
        exit_signal = ~enter_signal
    elif isinstance(exit_signal, pd.Series):
        exit_signal = _broadcast(gen, exit_signal)
    cols = alpha_factor.columns.intersection(defense_factor.columns).intersection(enter_signal.columns).intersection(exit_signal.columns)
    enter = enter_signal[cols]
    exit_ = exit_signal[cols]
    if confirm > 1:
        enter = enter.rolling(confirm, min_periods=1).sum().ge(confirm)
        exit_ = exit_.rolling(confirm, min_periods=1).sum().ge(confirm)
    state = enter.where(enter, other=(~exit_).where(exit_)).ffill().fillna(False)
    return alpha_factor[cols].where(~state, defense_factor[cols])

def _broadcast(gen, market_series: pd.Series) -> pd.DataFrame:
    """广播市场级Series到个股DataFrame."""
    return pd.DataFrame({c: market_series.values for c in gen.close.columns}, index=market_series.index)

# ---- 因子函数 ----
def compute(gen, params):
    """
    情绪硬切换: 恐慌状态 → 切大市值防守(离散硬切换); 正常状态 → 动量进攻。
    与连续混合不同，恐慌时完全切换到底座资产，不留动量暴露。
    极端尾部保护 > 连续切换的平滑性。
    """
    win1 = int(params.get('win1', 60))
    win2 = int(params.get('win2', 5))
    panic_confirm = int(params.get('panic_confirm', 2))
    ret = gen.close.pct_change(1)
    min_p = max(30, win1 // 2)
    cum = (1 + ret).rolling(win1, min_periods=min_p).apply(lambda x: x.prod(), raw=True)
    path = ret.abs().rolling(win1, min_periods=min_p).sum()
    eff = (abs(cum - 1) / (path + 1e-06)).fillna(0)
    mom = eff * np.sign(ret.rolling(win1, min_periods=min_p).mean())
    mom = cs_zscore(gen, mom)
    cap = np.log(gen.market_cap.replace(0, np.nan))
    vol_inv = 1.0 / (ret.rolling(win1, min_periods=min_p).std() + 1e-06)
    defense_raw = cs_zscore(gen, cap) * 0.5 + cs_zscore(gen, vol_inv) * 0.5
    defense = cs_zscore(gen, defense_raw)
    panic = panic_state(gen, window=win1)
    return discrete_switch(gen, mom, defense_factor=defense, enter_signal=panic > 0.7, confirm=panic_confirm)
