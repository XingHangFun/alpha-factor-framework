# 因子菜谱 Factor Cookbook

配合 rule_v11.txt + FunctionPool.py 使用。本文档提供可复现的因子生成范式，包含经济学直觉、代码范例和已知陷阱。


## 零、最小可运行骨架

```python
import numpy as np
import pandas as pd

def factor_YYYYMMDD_fanxinghang_XX_MyFactor(self, win1=20, win2=5):
    # 1. 构造原始信号 (纯 pandas, 不用任何 self 方法)
    raw = self.close.pct_change(win1)

    # 2. 通过标准流水线清洗
    result = self.factor_process(raw, outlier='mad', neutralize=True,
                                 zscore=True, smooth=win2)

    # 3. 返回干净的因子 (不需要额外 ewm — factor_process 已做)
    return result
```

流水线顺序不可变: NaN清理 → 去极值 → 中性化 → ZScore → 平滑 → Rank → 截断 → 填零。
不需要的步骤通过参数关闭，如 `neutralize=False`。


## 一、范例因子与经济学直觉

### 1.1 MR 类 — 筹码集中度反转 (结构最简单)

**直觉**: A 股筹码分散的股票往往被过度炒作，而筹码集中的股票在散户市场中被低估。
成交量波动率是筹码分散度的代理变量，波动越大 → 散户参与越多 → 未来收益越低。

```python
def factor_20260627_fanxinghang_MR_ChipConcentration(self, win1=60, win2=120, win3=5):
    # 成交量波动率 = 近期量 / 长期均量的变异系数
    vol_ma = self.volume.rolling(win1).mean()
    vol_ratio = self.volume / (vol_ma + 1e-06)

    # 波动率高 → 筹码分散 → 负向信号
    raw = -1.0 * vol_ratio.rolling(win2).std()

    # 写自己的 cs_zscore + ewm 也可以
    alpha = self.cs_zscore(raw)
    return alpha.ewm(span=win3).mean()
```

**参数解释**:
- `win1=60`: 成交量基准窗口 (~1季度)，越长基准越稳定
- `win2=120`: 波动率计算窗口 (~半年)，捕捉中长期筹码变化
- `win3=5`: 平滑窗口，控制换手

**注意**: 这个因子没有用 `factor_process` 也没有 boll 保护，是最简单的单步反转因子。


### 1.2 MR 类 — 市值加权的下行半方差 (带 boll 防守，ARB 锚)

**直觉**: 市值加权的市场下行波动捕捉的是"大资金撤退"的信号。个股对市场下行半方差的暴露越大，
说明越容易在系统性下跌中被抛售，长期应有折价补偿（反转逻辑）。boll 在小盘流动性危机时切大市值防守。

```python
def factor_20260627_fanxinghang_MR_CAPDownsideSemi(self, win1=5, win2=40, win3=40):
    mv = np.log(self.market_cap.replace(0, np.nan))

    # 市值加权的市场收益
    market_price = (self.close.pct_change(win1) * mv).div(
        mv.sum(axis=1), axis=0).sum(axis=1)

    # 仅保留下行部分
    downside_ret = np.minimum(market_price, 0)

    # 下行波动率的变化方向
    market_semi = downside_ret.rolling(win2, min_periods=max(5, win2 // 2)).std()
    direction = market_semi - market_semi.rolling(win3, min_periods=max(5, win3 // 2)).mean()

    # 用市值加权方向信号
    f = -self.market_cap.mul(direction, axis=0)

    # boll: 小盘走弱时切大市值防守
    return self.boll(c=f)
```

**为什么 CAPDownsideSemi 全时段 Calmar 最高 (1.80)**:
- 下行半方差是二阶矩，比一阶矩（均值反转）更稳定
- 市值加权避免了小盘噪声主导信号
- boll 的防守机制削掉了极端回撤

**参数解释**:
- `win1=5`: 短期收益窗口，捕捉近期市场方向
- `win2=40`: 半方差滚动窗口 (~2个月)
- `win3=40`: 半方差变化的基准窗口，`direction > 0` 意味风险在上升


### 1.3 MT 类 — 路径效率趋势 (纯动量，无 boll)

**直觉**: 价格位移 / 路径长度 = 趋势效率。高效率 = 价格朝一个方向走、波动小 = 强趋势。
低效率 = 价格来回波动、原地踏步 = 震荡。A 股在趋势市中动量效应显著。

```python
def factor_20260627_fanxinghang_MT_TrendPathEfficiency(self, win1=250):
    diff = self.close.diff().abs()
    path_length = diff.rolling(win1).sum()                       # 总路程
    displacement = (self.close - self.close.shift(win1)).abs()   # 净位移
    alpha = displacement / (path_length + 1e-08)                  # 效率 = 位移 / 路程
    return alpha.ewm(span=5).mean()
```

**为什么有效**: 趋势效率 > 简单动量收益，因为它区分了"稳步上涨"和"剧烈震荡后勉强收涨"。
后者动量不可持续，效率因子自动过滤。

**参数解释**:
- `win1=250`: 年度窗口 (~1年)，捕捉中长期趋势质量


### 1.4 MTR 类 — 动量 × 稳定性混合 (白盒 MTR)

**直觉**: 纯动量在震荡市反复止损，纯反转在趋势市跑输。MTR 同时持有两者：
动量强度 × 换手稳定性的交叉项 = 在趋势中跟涨且不会频繁换仓的股票。

```python
def factor_20260627_fanxinghang_MTR_HybridDefenseOffense(self, win1=60, win2=120):
    ret = self.close.pct_change(1)

    # 动量信号: 收益 / 波动 → Sharpe-like，避免高波动动量陷阱
    mom = (ret.rolling(win1, min_periods=30).mean() /
           (ret.rolling(win1, min_periods=30).std() + 1e-06))
    rank_mom = mom.rank(axis=1, pct=True)

    # 稳定性信号: 换手率的变异系数取负 → 换手越稳定越好
    turnover = self.total_turnover / (self.market_cap + 1e-06)
    cv_turn = (turnover.rolling(win2, min_periods=60).std() /
               (turnover.rolling(win2, min_periods=60).mean() + 1e-06))
    rank_stability = (-cv_turn).rank(axis=1, pct=True)

    # 交叉项: 动量 × 稳定 → 既涨又稳的股票
    f = rank_mom * rank_stability
    return f.ewm(halflife=5, axis=0).mean()
```

**为什么算 MTR**: 动量部分在趋势市贡献 Alpha，稳定性过滤在震荡市排除假突破。
两者交叉天然实现"趋势强时偏进攻，震荡时偏防守"。

**参数解释**:
- `win1=60`: 动量计算窗口
- `win2=120`: 换手稳定性窗口 (~半年)，需足够长才能识别真稳定


## 二、已知陷阱 (Pitfalls)

### 2.1 cumsum 与 I(1) 非平稳

**陷阱**: 对差分序列做 `cumsum()` 产生单位根过程——序列的起点选择会永久改变后续所有值，
回测结果对起始日期高度敏感。

**检测**: 如果改变回测起始日期 ±5 天导致 Calmar 波动 >30%，因子很可能有 I(1) 问题。

**修复**: 用 `rolling(N).sum()` 替代 `cumsum()`：
```python
# 错误
rs = diff.cumsum()

# 正确
rs = diff.rolling(N, min_periods=max(5, N // 2)).sum()
```

### 2.2 极端参数让逻辑退化

**陷阱**: 极端参数可能让一个看起来复杂的模型退化成简单模型。

**例子**: boll 中 `SHORT=1.1` — EMA span=1.1 接近无平滑，MACD 退化成 `diff_sum.diff()`。
6 参数模型实际在做 2 参数的事情，但参数面上多了 4 个噪声维度。

**修复**: 参数默认值应落在合理区间内，搜索范围由 `convert_factors` 从默认值自动推导。
如果某参数推到极端时因子性能突变，应检查是否发生了逻辑退化。

### 2.3 零方差除零

**陷阱**: `x.std() + 1e-08` 在零方差时产生超大量级而非 NaN，可能掩盖无信号的事实。

**修复**: FunctionPool 的 `cs_zscore` 已将零方差替换为 NaN (`std.replace(0, np.nan)`)，
使用 `self.cs_zscore()` 而非手写版本即可规避。

### 2.4 因子相似度与冗余

**陷阱**: 两个因子相似度 >0.7 时，线性组合不会带来增量 Alpha，只增加换手成本。

**判断**: 看 backtest_result.csv 的"最大相似度"列。
如果两个候选因子的互相相似度 >0.7，只保留全时段 Calmar 更高的那个。

### 2.5 MAD vs Winsorize 选择

| 场景 | 用 |
|------|----|
| 因子值分布有肥尾 (多数 Alpha 因子) | MAD (默认) |
| 因子值接近正态分布 | Winsorize |
| 因子是 rank 输出，无极端值 | 关闭 (outlier=None) |

### 2.6 平滑与换手权衡

`factor_process` 中 `smooth=5` 是 EWM span。span 越大换手越低但信号延迟越大。
月度调仓场景下 span=5 是合理默认值。如果换手率 >50%，增大到 10-15。

### 2.7 防守切换的灵敏度

boll 的离散切换用 `discrete_switch(confirm=N)`。confirm=1 时每根日线噪音都可能触发切换，
confirm=5 时延迟太大，A 股小盘急跌 3 天已经跌完。建议搜索范围 1-5。


## 三、FunctionPool 关键 API 速查

| 你要做什么 | 调用 |
|-----------|------|
| 清洗因子 | `self.factor_process(raw, outlier='mad', neutralize=True)` |
| 截面标准化 | `self.cs_zscore(df, clip=3)` |
| 截面排名 | `self.cs_rank(df)` 或 `self.cs_rank_uniform(df)` |
| 时序标准化 | `self.ts_zscore(df, window=60)` |
| 市值中性化 | `self.cs_neutralize(y, np.log(self.market_cap))` |
| 时序去趋势 | `self.ts_regression_resid(y, x, window=60)` |
| 波动率状态 | `self.vol_regime(window=60)` |
| 恐慌状态 | `self.panic_state(window=60)` |
| 趋势强度 | `self.trend_strength(window=60)` |
| MTR 连续混合 | `self.regime_switch(trend_factor, reversal_factor)` |
| 离散风控切换 | `self.discrete_switch(alpha, defense, enter_signal, confirm=3)` |
| 多因子等权复合 | `self.composite_equal([f1, f2, f3])` |
| 多因子 IC 加权 | `self.composite_ic_weighted([f1, f2], [ic1, ic2])` |
