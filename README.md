# Alpha Factor Research Framework

中证 1000 指数增强 — 双轨攻防量化因子体系

## 概述

本项目是一套面向 **中证 1000 全市场选股** 的 Alpha 因子研究框架，以"双轨攻防"为核心哲学：

- **ARB (All-Weather Robust Base)**：全时段稳健底座轨，追求跨牛熊的长期稳定性
- **RDR (Recent Dynamic Rotation)**：近期爆发轮换轨，捕捉当前市场周期的主线 Alpha

回测设定：中证 1000 全市场选 200 股，月度调仓。

## 因子体系

共计 **40+ Alpha 因子**，按策略属性分为三类：

| 类别 | 含义 | 数量 | 代表因子 |
|------|------|------|---------|
| **MR** (Mean Reversion) | 回归震荡类 — 捕捉过度反应后的价格修正 | 16 | IdioVolReversal, IlliquidityPremium, CAPMAlphaReversal, OvernightReversal |
| **MT** (Momentum Trend) | 动量趋势类 — 捕捉价格/量能的方向性信号 | 16 | TrendPathEfficiency, ResidualMomentum, FractalEfficiency, VolumePriceTrend |
| **MTR** (Momentum-Trend-Reversal) | 自适应切换类 — 根据市场状态动态切换 | 8 | PanicDefense, VolAdaptive, CrashProtected, LiquidityRegime |

## 工程组件

### FunctionPool — 因子工具库

`FunctionPool.py` 提供 27 个基类方法，Mixin 设计：

- **截面变换**：`cs_zscore`, `cs_rank`, `cs_mad_clip`, `cs_winsorize`, `cs_neutralize`
- **时序变换**：`ts_zscore`, `ts_regression_resid`
- **因子处理流水线**：`factor_process()` — NaN 清理 → 去极值 → 中性化 → ZScore → 平滑 → Rank
- **市场状态检测**：`vol_regime`, `panic_state`, `trend_strength`, `market_dispersion`
- **Regime 切换**：`regime_switch` (MTR 连续混合), `discrete_switch` (离散风控)
- **因子复合**：`composite_equal`, `composite_ic_weighted`

### convert_factors — 格式转换工具

`convert_factors.py` 实现回测框架（py 文件）与组合框架（factor 文件夹 + params.csv）的双向无损转换，保证回测与生产环境一致性。

### 参数搜索

从因子函数签名自动推导搜索范围，支持步长搜索与平原测试，评估因子在不同窗口期下的稳定性。

## 项目结构

```
.
├── FunctionPool.py          # 因子工具库 (27个基类方法)
├── factors.py               # V1 因子 (20个, 2026.06.27)
├── factors_v2.py            # V2 因子 (20个, 2026.07.03)
├── convert_factors.py       # py ↔ factor 双向转换脚本
├── rule_v11.txt             # 因子体系白皮书 V11.1
├── factor_cookbook.md       # 因子菜谱 (范例+经济学直觉+已知陷阱)
├── factor/                  # V1 因子文件夹 (组合框架格式)
├── factor_v2/               # V2 因子文件夹 (组合框架格式)
└── factor_roundtrip.py      # 往返一致性校验
```

## 免责声明

本项目仅用于展示量化因子研究方法论与工程能力，不构成任何投资建议。
