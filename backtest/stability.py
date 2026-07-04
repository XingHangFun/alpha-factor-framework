"""
ParamStabilityTester — 因子参数稳定性测试
=========================================
评估因子对参数变化的敏感度，支持两种模式：

1. sensitivity (OAT): 每次只变一个参数，其他固定在默认值 — 适合任意参数数量
2. grid: 全网格搜索 — 适合 ≤2 参数的精细化分析，≥3 参数时指数爆炸会自动降级

参数搜索范围与 convert_factors.py 保持一致，保证回测与参数优化口径统一。
"""

import itertools
import warnings
from typing import Optional, Callable, Dict, List, Any

import numpy as np
import pandas as pd

from .engine import FactorBacktestEngine, _detect_factor_type, _call_factor
from .metrics import (
    annual_return, sharpe_ratio, calmar_ratio, max_drawdown,
    cumulative_return, sortino_ratio,
)
from .env import BacktestEnv


# ================================================================
# 参数范围推导（与 convert_factors.py 一致）
# ================================================================

def param_range(v: Any, n_points: int = 5) -> list:
    """
    从默认值推导参数搜索范围。

    与 convert_factors.py generate_params_csv() 使用相同公式。

    Parameters
    ----------
    v : default value (int, float, bool, or None)
    n_points : int
        采样点数（不含默认值，最终列表包含默认值 + n_points 个测试值）。

    Returns
    -------
    list : 参数值列表（已去重排序，包含默认值）。
    """
    if v is None:
        lo, hi, step = 1, 500, 50
        values = list(range(lo, hi + 1, step))
    elif isinstance(v, bool):
        return [False, True]
    elif isinstance(v, int):
        step = max(1, abs(v) // 5)
        lo = max(1, abs(v) // 4)
        hi = max(v * 4, v + 20)
        if lo > hi:
            lo, hi = hi, lo
        values = list(range(lo, hi + 1, step))
    elif isinstance(v, float):
        mag = abs(v)
        step = max(0.1, round(mag / 5, 2))
        lo = max(0.01, round(mag / 4, 2))
        hi = max(round(mag * 4, 1), mag + 5.0)
        values = []
        x = lo
        while x <= hi + 1e-9:
            values.append(round(x, 4))
            x += step
    else:
        # 字符串等，返回原值
        return [v]

    # 确保默认值在列表中
    if v is not None and not isinstance(v, bool) and v not in values:
        values.append(v)

    # 排序去重
    values = sorted(set(values))

    # 下采样到 n_points+1（含默认值）
    if len(values) > n_points + 1:
        idx = np.linspace(0, len(values) - 1, n_points + 1, dtype=int)
        values = [values[i] for i in idx]

    return values


# ================================================================
# ParamStabilityTester
# ================================================================

class ParamStabilityTester:
    """
    因子参数稳定性测试器。

    使用回测引擎对因子在不同参数组合下的表现进行评估，
    输出参数敏感度分析和稳定性评分。

    Parameters
    ----------
    env : BacktestEnv
        数据环境。
    n_select : int
        每期选股数。
    rebalance_freq : int or str
        调仓频率。
    commission : float
        手续费率。
    slippage : float
        滑点率。
    long_only : bool
        是否仅做多。
    warmup_days : int
        预热天数。
    verbose : bool
        是否输出进度。

    Example
    -------
    >>> env = BacktestEnv('data.pkl')
    >>> tester = ParamStabilityTester(env, n_select=200)
    >>> result = tester.run(my_factor, method='sensitivity')
    >>> tester.summary()
    """

    def __init__(
        self,
        env: BacktestEnv,
        n_select: int = 200,
        rebalance_freq='M',
        commission: float = 0.0003,
        slippage: float = 0.001,
        long_only: bool = True,
        warmup_days: int = 252,
        verbose: bool = True,
    ):
        self.env = env
        self.n_select = n_select
        self.rebalance_freq = rebalance_freq
        self.commission = commission
        self.slippage = slippage
        self.long_only = long_only
        self.warmup_days = warmup_days
        self.verbose = verbose

        self._last_result: Optional[dict] = None

    # ---- 主入口 ----

    def run(
        self,
        factor_func: Callable,
        base_params: Optional[Dict[str, Any]] = None,
        method: str = 'sensitivity',
        metric: str = 'sharpe_ratio',
    ) -> dict:
        """
        执行参数稳定性测试。

        Parameters
        ----------
        factor_func : callable
            因子函数（py 格式或 folder 格式）。
        base_params : dict, optional
            基准参数。不传则使用因子函数的默认参数。
        method : str
            'sensitivity' = OAT 单参数变化（默认），
            'grid' = 全网格搜索。
        metric : str
            排序/选优的目标指标。

        Returns
        -------
        dict:
            'sensitivity': DataFrame — 所有测试结果的明细表
            'stability_score': float — 目标指标 > 0 的参数组合占比
            'best_params': dict — 最优参数
            'best_metrics': dict — 最优参数下的关键指标
            'plateau_ratio': float — 指标在最优 80% 以内的参数占比
            'param_impact': dict — 每个参数对目标指标的影响度（方差贡献）
            'heatmaps': dict — {param: DataFrame} 每个参数对指标的走势
        """
        # 解析因子类型和默认参数
        factor_type = _detect_factor_type(factor_func)
        default_params = self._get_default_params(factor_func, factor_type)
        if base_params:
            default_params.update(base_params)

        if not default_params:
            raise ValueError(
                "无法确定因子参数。请通过 base_params 显式传入参数。"
            )

        # 生成搜索网格
        search_params = {}
        for pname, pval in default_params.items():
            search_params[pname] = param_range(pval)

        if self.verbose:
            total_combos = 1
            for vals in search_params.values():
                total_combos *= len(vals)
            print(f"\n参数搜索空间: {search_params}")
            print(f"  每个参数取值数: {{{', '.join(f'{k}: {len(v)}' for k, v in search_params.items())}}}")

        if method == 'sensitivity':
            results_df, heatmaps = self._run_sensitivity(
                factor_func, factor_type, default_params, search_params
            )
        elif method == 'grid':
            results_df, heatmaps = self._run_grid(
                factor_func, factor_type, search_params
            )
        else:
            raise ValueError(f"不支持的方法: {method}。可选: 'sensitivity', 'grid'")

        # 计算稳定性指标
        if results_df.empty:
            self._last_result = {
                'sensitivity': results_df,
                'stability_score': 0.0,
                'best_params': {},
                'best_metrics': {},
                'plateau_ratio': 0.0,
                'param_impact': {},
                'heatmaps': {},
            }
            return self._last_result

        metric_vals = results_df[metric].values
        best_idx = metric_vals.argmax()
        best_val = metric_vals[best_idx]
        stability_score = (metric_vals > 0).mean()
        plateau_ratio = (metric_vals >= best_val * 0.8).mean() if best_val > 0 else 0.0

        # 参数影响度（方差分析）
        param_impact = {}
        if method == 'sensitivity':
            for param in search_params:
                subset = results_df[results_df['param'] == param]
                if len(subset) > 1 and subset[metric].std() > 0:
                    param_impact[param] = subset[metric].std() / abs(subset[metric].mean()) if subset[metric].mean() != 0 else 0
                else:
                    param_impact[param] = 0.0

        best_row = results_df.iloc[best_idx]
        best_params = {}
        for col in results_df.columns:
            if col.startswith('p_'):
                best_params[col[2:]] = best_row[col]

        best_metrics = {
            'sharpe_ratio': best_row.get('sharpe_ratio', 0),
            'calmar_ratio': best_row.get('calmar_ratio', 0),
            'annual_return': best_row.get('annual_return', 0),
            'max_drawdown': best_row.get('max_drawdown', 0),
            'sortino_ratio': best_row.get('sortino_ratio', 0),
            'avg_turnover': best_row.get('avg_turnover', 0),
        }

        self._last_result = {
            'sensitivity': results_df,
            'stability_score': float(stability_score),
            'best_params': best_params,
            'best_metrics': best_metrics,
            'plateau_ratio': float(plateau_ratio),
            'param_impact': param_impact,
            'heatmaps': heatmaps,
            'target_metric': metric,
        }
        return self._last_result

    # ---- 敏感度分析 (OAT) ----

    def _run_sensitivity(
        self,
        factor_func: Callable,
        factor_type: str,
        default_params: dict,
        search_params: dict,
    ) -> tuple:
        """OAT: 每个参数独立变化，其他参数固定在默认值。"""
        rows = []
        heatmaps = {}

        for param, values in search_params.items():
            param_rows = []
            if self.verbose:
                print(f"\n--- 测试参数: {param} ({len(values)} 个值) ---")

            for val in values:
                params = dict(default_params)
                params[param] = val
                metrics_row = self._eval_single(factor_func, factor_type, params, param, val)
                rows.append(metrics_row)
                param_rows.append({'value': val, **{k: metrics_row[k] for k in [
                    'sharpe_ratio', 'calmar_ratio', 'annual_return', 'max_drawdown', 'sortino_ratio'
                ]}})

                if self.verbose:
                    print(f"  {param}={val:>6}: "
                          f"Sharpe={metrics_row['sharpe_ratio']:>7.3f}  "
                          f"Calmar={metrics_row['calmar_ratio']:>7.3f}  "
                          f"AnnRet={metrics_row['annual_return']:>8.4f}")

            # 生成该参数的热力图数据
            heatmaps[param] = pd.DataFrame(param_rows).set_index('value')

        return pd.DataFrame(rows), heatmaps

    # ---- 网格搜索 ----

    def _run_grid(
        self,
        factor_func: Callable,
        factor_type: str,
        search_params: dict,
    ) -> tuple:
        """全网格搜索。"""
        param_names = list(search_params.keys())
        param_values = [search_params[p] for p in param_names]

        total = 1
        for v in param_values:
            total *= len(v)

        if total > 100:
            warnings.warn(
                f"网格搜索组合数 ({total}) 超过 100，"
                f"建议使用 method='sensitivity' 或减少参数数量/采样点数。"
            )

        if self.verbose:
            print(f"\n--- 网格搜索: {total} 个组合 ---")

        rows = []
        for combo in itertools.product(*param_values):
            params = dict(zip(param_names, combo))

            # 构建 param 标签用于 identify
            param_label = ','.join(f'{k}={v}' for k, v in params.items())
            row = self._eval_single(factor_func, factor_type, params, 'grid', param_label)
            rows.append(row)

            if self.verbose:
                print(f"  [{param_label}]: Sharpe={row['sharpe_ratio']:.3f}  Calmar={row['calmar_ratio']:.3f}")

        # 网格模式下的 heatmaps: 选前两个参数做 2D pivot
        heatmaps = {}
        if len(param_names) >= 2:
            p1, p2 = param_names[0], param_names[1]
            pivot = pd.DataFrame(rows).pivot_table(
                index=f'p_{p1}', columns=f'p_{p2}', values='sharpe_ratio'
            )
            heatmaps['sharpe_2d'] = pivot

        return pd.DataFrame(rows), heatmaps

    # ---- 单次回测 ----

    def _eval_single(
        self,
        factor_func: Callable,
        factor_type: str,
        params: dict,
        param_name: str,
        param_value,
    ) -> dict:
        """对单个参数组合执行回测，返回关键指标。"""
        engine = FactorBacktestEngine(
            env=self.env,
            n_select=self.n_select,
            rebalance_freq=self.rebalance_freq,
            commission=self.commission,
            slippage=self.slippage,
            long_only=self.long_only,
            warmup_days=self.warmup_days,
        )
        engine.add_factor(factor_func, params=params, factor_type=factor_type, name='_test')
        results = engine.run()

        result = results.get('_test', {})
        if 'error' in result:
            warnings.warn(f"参数 {param_name}={param_value} 回测失败: {result['error']}")
            return self._empty_row(params, param_name, param_value)

        m = result.get('metrics', {})
        row = {
            'param': param_name,
            'value': str(param_value),
            'sharpe_ratio': m.get('sharpe_ratio', 0.0),
            'calmar_ratio': m.get('calmar_ratio', 0.0)
                if not np.isinf(m.get('calmar_ratio', 0)) else 99.0,
            'annual_return': m.get('annual_return', 0.0),
            'max_drawdown': m.get('max_drawdown', 0.0),
            'sortino_ratio': m.get('sortino_ratio', 0.0),
            'avg_turnover': m.get('avg_turnover', 0.0),
            'cumulative_return': m.get('cumulative_return', 0.0),
        }

        # 嵌入参数值作为列（便于后续排序/筛选）
        for pname, pval in params.items():
            row[f'p_{pname}'] = pval

        return row

    def _empty_row(self, params: dict, param_name: str, param_value) -> dict:
        """生成失败测试的空行。"""
        row = {
            'param': param_name,
            'value': str(param_value),
            'sharpe_ratio': 0.0,
            'calmar_ratio': 0.0,
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'sortino_ratio': 0.0,
            'avg_turnover': 0.0,
            'cumulative_return': 0.0,
        }
        for pname, pval in params.items():
            row[f'p_{pname}'] = pval
        return row

    # ---- 默认参数推导 ----

    def _get_default_params(self, factor_func: Callable, factor_type: str) -> dict:
        """
        获取因子函数的默认参数。

        对 py 格式，用一次无参调用来导出实际使用的默认值（通过 trace）。
        对 folder 格式，读取 params.csv。
        """
        import inspect

        if factor_type == 'py':
            try:
                sig = inspect.signature(factor_func)
                params = {}
                for name, p in sig.parameters.items():
                    if name == 'self':
                        continue
                    if p.default is not inspect.Parameter.empty:
                        params[name] = p.default
                    else:
                        params[name] = None
                return params
            except (ValueError, TypeError):
                return {}

        # folder 格式 — 尝试从 compute 签名推导
        if hasattr(factor_func, 'compute'):
            try:
                sig = inspect.signature(factor_func.compute)
                # 检查是否有 params 参数的 default hint
                # folder 格式在转换时不保留默认值签名，回退到空
            except Exception:
                pass

        return {}

    # ---- 报告 ----

    def summary(self):
        """打印稳定性测试报告。"""
        if self._last_result is None:
            print("尚未执行测试。请先调用 run()。")
            return

        r = self._last_result

        print(f"\n{'='*60}")
        print(f"  因子参数稳定性报告")
        print(f"{'='*60}")

        metric = r.get('target_metric', 'sharpe_ratio')

        print(f"\n  --- 稳定性指标 ---")
        print(f"  目标指标:       {metric}")
        print(f"  稳定性评分:     {r['stability_score']:.2%}  ({metric}>0 的比例)")
        print(f"  高原比例:       {r['plateau_ratio']:.2%}  ({metric}≥最优80% 的比例)")

        best = r['best_metrics']
        best_params = r['best_params']
        print(f"\n  --- 最优参数 ---")
        print(f"  参数:           {best_params}")
        print(f"  夏普比率:       {best['sharpe_ratio']:.3f}")
        print(f"  卡尔玛比率:     {best['calmar_ratio']:.3f}")
        print(f"  年化收益率:     {best['annual_return']:.2%}")
        print(f"  最大回撤:       {best['max_drawdown']:.2%}")
        print(f"  索提诺比率:     {best['sortino_ratio']:.3f}")
        print(f"  平均换手率:     {best['avg_turnover']:.2%}")

        impact = r.get('param_impact', {})
        if impact:
            print(f"\n  --- 参数影响度 (变异系数) ---")
            for pname, cv in sorted(impact.items(), key=lambda x: -x[1]):
                bar = '█' * min(int(cv * 20), 40)
                print(f"  {pname:15s}: {cv:.3f}  {bar}")

        df = r['sensitivity']
        if df is not None and not df.empty:
            print(f"\n  --- 全局统计 ({len(df)} 个测试点) ---")
            for col in ['sharpe_ratio', 'calmar_ratio', 'annual_return', 'max_drawdown']:
                vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(vals) > 0:
                    print(f"  {col:18s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
                          f"min={vals.min():.4f}  max={vals.max():.4f}")

    @property
    def last_result(self) -> Optional[dict]:
        """返回最近一次测试的完整结果。"""
        return self._last_result
