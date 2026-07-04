# Alpha Factor Research Framework

中证 1000 指数增强 — 双轨攻防量化因子体系

## 概述

本项目是一套面向 **中证 1000 全市场选股** 的 Alpha 因子研究框架，以"双轨攻防"为核心哲学：

- **ARB (All-Weather Robust Base)**：全时段稳健底座轨，追求跨牛熊的长期稳定性
- **RDR (Recent Dynamic Rotation)**：近期爆发轮换轨，捕捉当前市场周期的主线 Alpha

回测设定：中证 1000 全市场选 200 股，月度调仓。

## 因子体系

共计 **40 Alpha 因子**，按策略属性分为三类：

| 类别 | 含义 | 数量 | 代表因子 |
|------|------|------|---------|
| **MR** (Mean Reversion) | 回归震荡类 — 捕捉过度反应后的价格修正 | 16 | IdioVolReversal, IlliquidityPremium, CAPMAlphaReversal, OvernightReversal, SkewnessReversal |
| **MT** (Momentum Trend) | 动量趋势类 — 捕捉价格/量能的方向性信号 | 16 | ResidualMomentum, FractalEfficiency, VolumePriceTrend, IntradayMom, TrendQuality |
| **MTR** (Momentum-Trend-Reversal) | 自适应切换类 — 根据市场状态动态切换 | 8 | PanicDefense, VolAdaptive, CrashProtected, LiquidityRegime, DispersionAdaptive |

## 工程组件

### FunctionPool — 因子工具库

`FunctionPool.py` 提供截面/时序变换、因子处理流水线、Regime 检测与切换、因子复合（等权 / IC加权 / 贝叶斯岭回归）。

### backtest — 回测框架

`backtest/` 子包提供完整的回测基础设施：
- `BacktestEnv` — 数据容器（pickle 加载 + FunctionPool 方法）
- `FactorBacktestEngine` — 回测循环（调仓、选股、交易成本、绩效）
- `metrics` — Sharpe, Calmar, Sortino, 最大回撤, IC, 换手率等
- `ParamStabilityTester` — 参数敏感度分析与稳定性评估

### convert_factors — 格式转换工具

`convert_factors.py` 实现 py ↔ factor 文件夹双向无损转换。

## 项目结构

```
.
├── __init__.py              # 包入口
├── FunctionPool.py          # 因子工具库
├── factors.py               # 40 个 Alpha 因子
├── convert_factors.py       # py ↔ factor 双向转换
├── README.md
├── docs/
│   ├── rule_v11.txt         # 因子体系白皮书
│   └── factor_cookbook.md   # 因子菜谱
├── backtest/                # 回测框架
│   ├── __init__.py
│   ├── env.py               # BacktestEnv — 数据容器
│   ├── engine.py            # FactorBacktestEngine — 回测引擎
│   ├── metrics.py           # 绩效指标
│   └── stability.py         # 参数稳定性测试
├── tests/
│   └── test_integration.py  # 集成测试
└── factor/                  # 40 个因子文件夹
```

## 免责声明

本项目仅用于展示量化因子研究方法论与工程能力，不构成任何投资建议。
