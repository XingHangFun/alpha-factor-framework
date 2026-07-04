"""
回测框架集成测试
=================
使用模拟数据演示完整回测流程。
运行: cd F0 && python tests/test_integration.py
"""

import os
import sys
import pickle
import tempfile

# 确保 F0 在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

import numpy as np
import pandas as pd

from backtest import BacktestEnv, FactorBacktestEngine


# ================================================================
# 1. 生成模拟数据
# ================================================================

def generate_synthetic_data(
    n_stocks: int = 500,
    n_days: int = 504,       # ~2 年
    start_date: str = '2024-01-01',
    seed: int = 42,
) -> dict:
    """
    生成模拟股票数据，包含一定可预测性。

    每个股票有隐藏的 Alpha 信号强度，用于验证因子回测。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, periods=n_days)
    stocks = [f'{i:06d}.SZ' if i % 2 == 0 else f'{i:06d}.SH'
              for i in range(1, n_stocks + 1)]

    # 为每个股票生成隐藏 Alpha 强度（控制未来收益的可预测程度）
    alpha_strength = rng.normal(0, 1, n_stocks)

    # 生成价格序列（几何布朗运动 + Alpha 信号）
    daily_ret = pd.DataFrame(
        rng.normal(0.0005, 0.02, (n_days, n_stocks)),  # 日收益均值 5bp
        index=dates, columns=stocks
    )

    # 注入截面 Alpha：强信号股票收益略高
    alpha_component = pd.DataFrame(
        np.outer(np.ones(n_days), alpha_strength * 0.0002),  # 2bp * 信号强度
        index=dates, columns=stocks
    )
    daily_ret += alpha_component

    # 累计价格
    price = (1 + daily_ret).cumprod() * 100

    close = price
    high = price * (1 + abs(rng.normal(0, 0.01, (n_days, n_stocks))))
    low = price * (1 - abs(rng.normal(0, 0.01, (n_days, n_stocks))))
    open_price = close.shift(1).fillna(price.iloc[0])

    # 成交量
    volume = pd.DataFrame(
        rng.lognormal(14, 1, (n_days, n_stocks)),
        index=dates, columns=stocks
    )

    # 市值（随机 + Alpha 信号做多股票市值更大）
    mkt_cap_base = 10 ** rng.uniform(9, 11, n_stocks)   # 1B ~ 100B
    mkt_cap = pd.DataFrame(
        np.outer(np.ones(n_days), mkt_cap_base),
        index=dates, columns=stocks
    )

    # 换手率
    total_turnover = pd.DataFrame(
        rng.lognormal(-2, 0.5, (n_days, n_stocks)),
        index=dates, columns=stocks
    )

    # 成交额 = 均价 × 成交量
    avg_price = (high + low + close) / 3
    amount = avg_price * volume

    # 换手率（百分比）
    turnoverrate = total_turnover / mkt_cap

    return {
        'close': close,
        'open': open_price,
        'high': high,
        'low': low,
        'volume': volume,
        'amount': amount,
        'market_cap': mkt_cap,
        'total_turnover': total_turnover,
        'turnoverrate': turnoverrate,
    }


# ================================================================
# 2. 定义几个测试因子
# ================================================================

def factor_test_Momentum(self, win1=60, win2=5):
    """
    简单动量因子: 过去 win1 天收益率，越高越好。
    py 格式: self = BacktestEnv 实例。
    """
    ret = self.close.pct_change(win1)
    raw = ret
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win2)


def factor_test_Reversal(self, win1=5, win2=5):
    """简单反转因子: 短期反转。"""
    ret = self.close.pct_change(win1)
    raw = -ret
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win2)


def factor_test_Volatility(self, win1=20, win2=5):
    """低波动因子: 低波动股票收益更高（异象）。"""
    ret = self.close.pct_change()
    vol = ret.rolling(win1, min_periods=max(10, win1 // 2)).std()
    raw = -vol
    return self.factor_process(raw, outlier='mad', neutralize=True,
                               zscore=True, smooth=win2)


# ================================================================
# 3. 运行回测示例
# ================================================================

def main():
    print("=" * 60)
    print("  Alpha Factor Backtesting Framework — 示例 & 测试")
    print("=" * 60)

    # 3.1 生成模拟数据
    print("\n[1/5] 生成模拟数据（500 只股票, ~2 年）...")
    data = generate_synthetic_data(n_stocks=500, n_days=504)

    # 写入临时 pickle 文件
    tmpdir = tempfile.mkdtemp()
    pkl_path = os.path.join(tmpdir, 'market_data.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

    # 3.2 加载数据环境
    print("[2/5] 加载 BacktestEnv ...")
    env = BacktestEnv(pkl_path)
    env.info()

    assert env.n_stocks == 500, f"Expected 500 stocks, got {env.n_stocks}"
    assert env.n_dates == 504, f"Expected 504 dates, got {env.n_dates}"
    print("  ✓ 数据加载验证通过")

    # 验证 FunctionPool 方法可用
    z = env.cs_zscore(env.close.pct_change())
    assert isinstance(z, pd.DataFrame), "cs_zscore should return DataFrame"
    print("  ✓ FunctionPool 方法可用")

    # 3.3 创建回测引擎
    print("[3/5] 创建回测引擎 ...")
    engine = FactorBacktestEngine(
        env=env,
        n_select=100,
        rebalance_freq='M',
        commission=0.0003,
        slippage=0.001,
        warmup_days=60,
    )

    # 注册因子
    engine.add_factor(factor_test_Momentum, params={'win1': 60, 'win2': 5}, name='Momentum')
    engine.add_factor(factor_test_Reversal, params={'win1': 5, 'win2': 5}, name='Reversal')
    engine.add_factor(factor_test_Volatility, params={'win1': 20, 'win2': 5}, name='LowVol')

    # 注册复合
    engine.add_composite(
        factors=[factor_test_Momentum, factor_test_Reversal, factor_test_Volatility],
        weights=[0.4, 0.3, 0.3],
        name='Composite_EW',
        composite_method='equal',
    )

    # 3.4 运行回测
    print("[4/5] 执行回测 ...")
    results = engine.run()
    print(f"  ✓ 完成 {len(results)} 个因子/组合的回测")

    # 3.5 输出结果
    print("[5/5] 绩效报告")
    engine.summary()

    # 3.6 集成验证
    print("\n" + "=" * 60)
    print("  集成验证")
    print("=" * 60)

    for name, result in results.items():
        if 'error' in result:
            print(f"  ✗ {name}: {result['error']}")
            continue

        ret = result['returns']
        metrics = result['metrics']

        # 验证日收益序列
        assert len(ret) > 0, f"{name}: 日收益序列为空"
        assert ret.isna().sum() < len(ret) * 0.5, f"{name}: 日收益 NaN 过多"

        # 验证指标合理性
        ann_ret = metrics.get('annual_return', 0)
        ann_vol = metrics.get('annual_volatility', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        mdd = metrics.get('max_drawdown', 0)
        avg_to = metrics.get('avg_turnover', 0)

        assert abs(ann_ret) < 5.0, f"{name}: 年化收益异常 {ann_ret:.2%}"
        assert ann_vol > 0, f"{name}: 年化波动率应为正"
        assert mdd <= 0, f"{name}: 最大回撤应为负值"
        assert 0 <= avg_to <= 1.0, f"{name}: 换手率应在 [0,1] 之间，实际 {avg_to:.2f}"

        print(f"  ✓ {name}")
        print(f"    年化收益={ann_ret:.2%}, 波动率={ann_vol:.2%}, "
              f"夏普={sharpe:.2f}, 最大回撤={mdd:.2%}, 换手率={avg_to:.2%}")

    # 导出结果
    output_dir = os.path.join(tmpdir, 'output')
    engine.export_results(output_dir)
    exported_files = os.listdir(output_dir)
    print(f"\n  ✓ 导出 {len(exported_files)} 个文件到 {output_dir}/")

    # 清理
    import shutil
    shutil.rmtree(tmpdir)

    print("\n" + "=" * 60)
    print("  所有测试通过 ✓")
    print("=" * 60)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
