"""
FactorBacktestEngine — 核心回测循环
====================================
负责调仓日历、因子计算、选股、组合收益、交易成本、绩效汇总。
"""

import inspect
import warnings
from typing import Optional, Callable, Dict, List, Union, Any

import numpy as np
import pandas as pd

from .env import BacktestEnv
from .metrics import full_report


# ================================================================
# 因子格式检测与适配
# ================================================================

def _detect_factor_type(factor_func: Callable) -> str:
    """
    检测因子调用格式。

    返回
    ----
    'py' : def factor_xxx(self, win1=..., win2=...) -> pd.DataFrame
    'folder' : def compute(gen, params) -> pd.DataFrame
    """
    if hasattr(factor_func, 'compute') and callable(getattr(factor_func, 'compute')):
        return 'folder'
    # 检查第一个参数名
    try:
        sig = inspect.signature(factor_func)
        params = list(sig.parameters.keys())
        if params and params[0] == 'self':
            return 'py'
        if params and params[0] == 'gen':
            return 'folder'
    except (ValueError, TypeError):
        pass
    # 默认按 py 格式处理
    return 'py'


def _call_factor(factor_func: Callable, env: BacktestEnv,
                 factor_type: str, params: Dict[str, Any]) -> pd.DataFrame:
    """
    统一调用因子，返回 DataFrame (index=date, columns=stock)。

    参数
    ----
    factor_func : callable
        因子函数或模块。
    env : BacktestEnv
        数据环境（继承 FunctionPool）。
    factor_type : str
        'py' 或 'folder'。
    params : dict
        参数字典。

    返回
    ----
    pd.DataFrame : 因子值，index=date, columns=stock_code。
    """
    if factor_type == 'folder':
        # 文件夹格式: compute(gen, params)
        return factor_func.compute(env, params or {})

    # py 格式: factor_func(self, win1=X, win2=Y, ...)
    if params:
        return factor_func(env, **params)
    else:
        return factor_func(env)


# ================================================================
# FactorBacktestEngine
# ================================================================

class FactorBacktestEngine:
    """
    Alpha 因子回测引擎。

    参数
    ----
    env : BacktestEnv
        数据环境，已加载行情数据。
    n_select : int
        每期选股数量（默认 200）。
    rebalance_freq : int or str
        调仓间隔。int = 每 N 个交易日；'M' = 每月最后一个交易日。
    commission : float
        单边手续费率（默认 0.0003 = 万三）。
    slippage : float
        单边滑点（默认 0.001 = 千一）。
    long_only : bool
        是否仅做多（默认 True）。False 时做多前 N + 做空后 N。
    warmup_days : int
        预热天数，跳过前 N 个交易日不做调仓（给因子滚动窗口收敛时间）。

    使用示例
    --------
    >>> env = BacktestEnv('./data/')
    >>> engine = FactorBacktestEngine(env, n_select=200, rebalance_freq='M')
    >>> engine.add_factor(my_factor, params={'win1': 60, 'win2': 5}, name='MyFactor')
    >>> engine.run()
    >>> engine.summary()
    """

    def __init__(
        self,
        env: BacktestEnv,
        n_select: int = 200,
        rebalance_freq: Union[int, str] = 'M',
        commission: float = 0.0003,
        slippage: float = 0.001,
        long_only: bool = True,
        warmup_days: int = 252,
    ):
        self.env = env
        self.n_select = n_select
        self.rebalance_freq = rebalance_freq
        self.commission = commission
        self.slippage = slippage
        self.long_only = long_only
        self.warmup_days = warmup_days

        # 因子注册表
        self._factors: List[Dict[str, Any]] = []

        # 结果存储
        self._results: Dict[str, Any] = {}
        self._ran = False

    # ---- 因子注册 ----

    def add_factor(
        self,
        factor_func: Callable,
        params: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        factor_type: Optional[str] = None,
    ):
        """
        注册单个因子。

        参数
        ----
        factor_func : callable
            因子函数（py 格式）或模块（folder 格式）。
        params : dict, optional
            因子参数。不传则使用默认值。
        name : str, optional
            显示名称（不传则自动从函数名推导）。
        factor_type : str, optional
            'py' / 'folder' / None（自动检测）。
        """
        if factor_type is None:
            factor_type = _detect_factor_type(factor_func)

        if name is None:
            if factor_type == 'folder':
                name = getattr(factor_func, '__name__', 'Unknown')
            else:
                name = getattr(factor_func, '__name__', 'Unknown')
            # 去掉 factor_YYYYMMDD_fanxinghang_ 前缀，保留策略分类_逻辑名
            if name.startswith('factor_'):
                parts = name.split('_')
                if len(parts) >= 4:
                    name = f"{parts[3]}_{'_'.join(parts[4:])}"

        self._factors.append({
            'func': factor_func,
            'type': factor_type,
            'params': params or {},
            'name': name,
        })

    def add_composite(
        self,
        factors: List[Callable],
        weights: Optional[List[float]] = None,
        params_list: Optional[List[Dict]] = None,
        name: str = 'Composite',
        composite_method: str = 'equal',
    ):
        """
        注册多因子复合。

        参数
        ----
        factors : list of callable
            因子函数列表。
        weights : list of float, optional
            因子权重（不传则等权）。ic_weighted/bayesian 模式下自动推导。
        params_list : list of dict, optional
            每个因子的参数（与 factors 一一对应）。
        name : str
            组合名称。
        composite_method : str
            'equal' = 等权复合；
            'ic_weighted' = IC 绝对值加权；
            'bayesian' = 贝叶斯岭回归（自动收缩噪声因子权重）。
        """
        if params_list is None:
            params_list = [{} for _ in factors]
        if weights is None:
            weights = [1.0 / len(factors)] * len(factors)

        resolved = []
        for f, p in zip(factors, params_list):
            ftype = _detect_factor_type(f)
            fname = getattr(f, '__name__', 'Unknown')
            resolved.append({'func': f, 'type': ftype, 'params': p, 'name': fname})

        self._factors.append({
            'composite': True,
            'factors': resolved,
            'weights': weights,
            'name': name,
            'composite_method': composite_method,
        })

    # ---- 回测执行 ----

    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行回测。

        返回
        ----
        dict : 每个因子/组合的回测结果。
            key = factor name
            value = {'returns': Series, 'positions': dict, 'turnover': float,
                     'metrics': dict, 'name': str}
        """
        if not self._factors:
            raise ValueError("未注册任何因子。请先调用 add_factor() 或 add_composite()。")

        # 确定日期范围
        all_dates = self.env.dates
        if start_date:
            all_dates = all_dates[all_dates >= pd.Timestamp(start_date)]
        if end_date:
            all_dates = all_dates[all_dates <= pd.Timestamp(end_date)]

        if len(all_dates) < self.warmup_days + 2:
            raise ValueError(
                f"有效日期不足: {len(all_dates)} 天，"
                f"需要至少 {self.warmup_days + 2} 天（warmup + 2个调仓日）。"
            )

        # 生成调仓日列表
        rebalance_dates = self._get_rebalance_dates(all_dates)

        # 对每个因子执行回测
        results = {}
        for factor_spec in self._factors:
            if factor_spec.get('composite'):
                result = self._run_composite(factor_spec, all_dates, rebalance_dates)
            else:
                result = self._run_single(factor_spec, all_dates, rebalance_dates)
            results[factor_spec['name']] = result

        self._results = results
        self._ran = True
        return results

    def _run_single(
        self, factor_spec: Dict, all_dates: pd.DatetimeIndex,
        rebalance_dates: pd.DatetimeIndex
    ) -> Dict[str, Any]:
        """
        执行单因子回测。

        核心流程:
        1. 计算全时段因子值
        2. 每个调仓日: 选前 N 只股票 → 持有到下个调仓日
        3. 计算组合日收益（等权）
        4. 扣除交易成本
        5. 汇总绩效指标
        """
        name = factor_spec['name']

        # 1. 计算因子值（一次调用，全时段）
        try:
            factor_values = _call_factor(
                factor_spec['func'], self.env,
                factor_spec['type'], factor_spec['params']
            )
        except Exception as e:
            warnings.warn(f"因子 {name} 计算失败: {e}。跳过。")
            return {'name': name, 'error': str(e), 'returns': pd.Series(dtype=float)}

        # 确保因子值覆盖回测日期
        factor_values = factor_values.reindex(index=all_dates)

        # 2. 逐期选股 + 持仓跟踪
        positions = {}        # {date: [stock_codes]}
        daily_portfolio_ret = pd.Series(0.0, index=all_dates)
        turnover_list = []

        prev_holdings = set()

        for i, rebal_date in enumerate(rebalance_dates):
            # 定位该调仓日在 all_dates 中的位置
            date_pos = all_dates.get_loc(rebal_date)

            # 获取当日因子值
            fv_today = factor_values.loc[rebal_date].dropna()

            if len(fv_today) < self.n_select:
                # 可选股票不足，全部持有
                selected = list(fv_today.index)
            else:
                if self.long_only:
                    # 做多：选因子值最大的 N 只
                    selected = list(fv_today.nlargest(self.n_select).index)
                else:
                    # 多空：做多前 N + 做空后 N
                    long_stocks = list(fv_today.nlargest(self.n_select).index)
                    short_stocks = list(fv_today.nsmallest(self.n_select).index)
                    selected = long_stocks + short_stocks

            stock_set = set(selected)
            positions[rebal_date] = selected

            # 计算换手率
            if prev_holdings:
                sold = len(prev_holdings - stock_set)
                turnover = sold / len(prev_holdings) if prev_holdings else 0.0
            else:
                turnover = 1.0  # 首次建仓视为 100% 换手
            turnover_list.append(turnover)

            # 确定持有区间: [rebal_date, next_rebal_date)
            if i + 1 < len(rebalance_dates):
                end_pos = all_dates.get_loc(rebalance_dates[i + 1])
            else:
                end_pos = len(all_dates)

            # 计算持有期每日收益（等权）
            ret_slice = self.env.close.pct_change().reindex(index=all_dates)
            for pos in range(date_pos, min(end_pos, len(all_dates))):
                dt = all_dates[pos]
                available = [s for s in selected if s in ret_slice.columns]
                if available:
                    daily_portfolio_ret.loc[dt] = ret_slice.loc[dt, available].mean()

            # 扣除调仓成本
            roundtrip_cost = (self.commission + self.slippage) * 2
            daily_portfolio_ret.loc[rebal_date] -= turnover * roundtrip_cost

            prev_holdings = stock_set

        # 填充未覆盖的日期（warmup 期）
        daily_portfolio_ret = daily_portfolio_ret.fillna(0)

        # 3. 计算绩效指标
        benchmark = self.env.benchmark_returns.reindex(daily_portfolio_ret.index).fillna(0)
        metrics = full_report(
            daily_returns=daily_portfolio_ret,
            benchmark_returns=benchmark,
            holdings_history={str(k.date()): set(v) for k, v in positions.items()},
        )

        # 额外计算平均换手率
        metrics['avg_turnover'] = np.mean(turnover_list) if turnover_list else 0.0

        return {
            'name': name,
            'returns': daily_portfolio_ret,
            'positions': positions,
            'turnover_list': turnover_list,
            'metrics': metrics,
            'factor_values': factor_values,
        }

    def _run_composite(
        self, factor_spec: Dict, all_dates: pd.DatetimeIndex,
        rebalance_dates: pd.DatetimeIndex
    ) -> Dict[str, Any]:
        """
        执行多因子复合回测。

        使用 FunctionPool 的 composite_equal 或 composite_ic_weighted 方法。
        """
        name = factor_spec['name']
        sub_factors = factor_spec['factors']
        weights = factor_spec['weights']
        method = factor_spec.get('composite_method', 'equal')

        # 计算每个子因子的因子值
        factor_dfs = []
        for sub in sub_factors:
            try:
                fv = _call_factor(sub['func'], self.env, sub['type'], sub['params'])
                factor_dfs.append(fv.reindex(index=all_dates))
            except Exception as e:
                warnings.warn(f"子因子 {sub['name']} 计算失败: {e}。跳过。")

        if not factor_dfs:
            return {'name': name, 'error': '所有子因子计算失败', 'returns': pd.Series(dtype=float)}

        # 对齐列（股票交集）
        common_cols = factor_dfs[0].columns
        for fv in factor_dfs[1:]:
            common_cols = common_cols.intersection(fv.columns)
        factor_dfs = [fv[common_cols] for fv in factor_dfs]

        # 复合
        if method == 'ic_weighted':
            # 用全时段 IC 均值作为权重
            fwd_ret = self.env.close.pct_change().shift(-1)
            ic_weights = []
            for fv in factor_dfs:
                ic_vals = []
                for dt in all_dates:
                    fv_dt = fv.loc[dt].dropna()
                    fr_dt = fwd_ret.loc[dt].dropna() if dt in fwd_ret.index else pd.Series()
                    common = fv_dt.index.intersection(fr_dt.index)
                    if len(common) >= 10:
                        ic_vals.append(fv_dt[common].corr(fr_dt[common]))
                ic_weights.append(abs(np.mean(ic_vals)) if ic_vals else 0.0)
            if sum(ic_weights) > 0:
                weights = [w / sum(ic_weights) for w in ic_weights]
            else:
                weights = [1.0 / len(factor_dfs)] * len(factor_dfs)

        elif method == 'bayesian':
            # 贝叶斯线性组合 — 使用 FunctionPool.composite_bayesian
            fwd_ret = self.env.close.pct_change().shift(-1)
            bayes_result = self.env.composite_bayesian(
                factor_dfs, fwd_ret, smooth=5, fit_window=None
            )
            bayes_composite = bayes_result['composite'].reindex(index=all_dates)

            # 包装为兼容格式
            def _composite_factor(env, **kwargs):
                return bayes_composite

            composite_spec = {
                'func': _composite_factor,
                'type': 'py',
                'params': {},
                'name': name,
            }

            result = self._run_single(composite_spec, all_dates, rebalance_dates)
            result['bayesian_diagnostics'] = {
                'weights': bayes_result['weights'],
                'intercept': bayes_result['intercept'],
                'alpha': bayes_result['alpha'],
                'scores': bayes_result['scores'],
            }
            return result

        # 构建临时因子：等权/加权复合
        def _composite_factor(env, **kwargs):
            normalized = [env.cs_zscore(fv) for fv in factor_dfs]
            raw = sum(w * n for w, n in zip(weights, normalized))
            return env.cs_zscore(raw)

        composite_spec = {
            'func': _composite_factor,
            'type': 'py',
            'params': {},
            'name': name,
        }

        return self._run_single(composite_spec, all_dates, rebalance_dates)

    # ---- 调仓日历 ----

    def _get_rebalance_dates(self, all_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        生成调仓日列表。

        参数
        ----
        all_dates : pd.DatetimeIndex
            所有交易日。

        返回
        ----
        pd.DatetimeIndex : 调仓日。
        """
        # 跳过 warmup
        dates = all_dates[self.warmup_days:]

        if self.rebalance_freq == 'M':
            # 每月最后一个交易日
            month_ends = dates.to_series().groupby(
                [dates.year, dates.month]
            ).apply(lambda x: x.index[-1])
            return pd.DatetimeIndex(month_ends.values)

        elif isinstance(self.rebalance_freq, int):
            # 每 N 个交易日
            return dates[::self.rebalance_freq]

        else:
            raise ValueError(f"不支持的调仓频率: {self.rebalance_freq}")

    # ---- 结果输出 ----

    def summary(self, factor_name: Optional[str] = None):
        """
        打印回测结果摘要。

        参数
        ----
        factor_name : str, optional
            指定因子名称。不传则打印所有因子。
        """
        if not self._ran:
            print("尚未执行回测。请先调用 run()。")
            return

        names = [factor_name] if factor_name else list(self._results.keys())

        for name in names:
            if name not in self._results:
                print(f"因子 '{name}' 不存在。")
                continue

            result = self._results[name]

            print(f"\n{'='*60}")
            print(f"  因子: {name}")
            print(f"{'='*60}")

            if 'error' in result:
                print(f"  ERROR: {result['error']}")
                continue

            metrics = result.get('metrics', {})

            labels = [
                ('累计收益率', 'cumulative_return', '{:.2%}'),
                ('年化收益率', 'annual_return', '{:.2%}'),
                ('年化波动率', 'annual_volatility', '{:.2%}'),
                ('夏普比率', 'sharpe_ratio', '{:.2f}'),
                ('卡尔玛比率', 'calmar_ratio', '{:.2f}'),
                ('索提诺比率', 'sortino_ratio', '{:.2f}'),
                ('最大回撤', 'max_drawdown', '{:.2%}'),
                ('日胜率', 'win_rate', '{:.2%}'),
                ('平均换手率', 'avg_turnover', '{:.2%}'),
            ]

            for label, key, fmt in labels:
                if key in metrics:
                    val = metrics[key]
                    # 处理 inf
                    if isinstance(val, float) and np.isinf(val):
                        val_str = '∞' if val > 0 else '-∞'
                    else:
                        val_str = fmt.format(val)
                    print(f"  {label:12s}: {val_str}")

            # 超额指标
            if 'excess_return' in metrics:
                print(f"  {'超额收益':12s}: {metrics['excess_return']:.2%}")
            if 'information_ratio' in metrics:
                print(f"  {'信息比率':12s}: {metrics['information_ratio']:.2f}")

            # 最大回撤详情
            mdd_detail = metrics.get('mdd_detail', {})
            if mdd_detail:
                print(f"  --- 最大回撤详情 ---")
                print(f"  峰顶日期: {mdd_detail.get('peak_date', 'N/A')}")
                print(f"  谷底日期: {mdd_detail.get('trough_date', 'N/A')}")
                print(f"  恢复日期: {mdd_detail.get('recovery_date', 'N/A')}")
                print(f"  持续天数: {mdd_detail.get('duration_days', 0)}")

            # IC 指标
            ic = metrics.get('ic', {})
            if ic:
                print(f"  --- IC 指标 ---")
                print(f"  IC 均值:   {ic.get('ic_mean', 0):.4f}")
                print(f"  IC 标准差: {ic.get('ic_std', 0):.4f}")
                print(f"  ICIR:      {ic.get('icir', 0):.4f}")
                print(f"  Rank IC:   {ic.get('rank_ic_mean', 0):.4f}")
                print(f"  IC 胜率:   {ic.get('ic_win_rate', 0):.2%}")

    def export_results(self, output_dir: str = './backtest_output'):
        """
        导出回测结果到 CSV 文件。

        参数
        ----
        output_dir : str
            输出目录。
        """
        import os

        if not self._ran:
            print("尚未执行回测。请先调用 run()。")
            return

        os.makedirs(output_dir, exist_ok=True)

        for name, result in self._results.items():
            if 'error' in result:
                continue

            safe_name = name.replace('/', '_').replace('\\', '_')

            # 日收益序列
            returns = result.get('returns')
            if returns is not None and len(returns) > 0:
                returns.to_csv(os.path.join(output_dir, f'{safe_name}_daily_returns.csv'),
                               header=['daily_return'])

            # 累计净值
            if returns is not None and len(returns) > 0:
                nav = (1 + returns).cumprod()
                nav.to_csv(os.path.join(output_dir, f'{safe_name}_nav.csv'),
                           header=['nav'])

            # 绩效指标
            metrics = result.get('metrics', {})
            if metrics:
                # 展平指标
                flat = {}
                for k, v in metrics.items():
                    if isinstance(v, dict):
                        for sk, sv in v.items():
                            if not isinstance(sv, (pd.Series, pd.DataFrame)):
                                flat[f'{k}.{sk}'] = sv
                    elif not isinstance(v, (pd.Series, pd.DataFrame)):
                        flat[k] = v
                pd.Series(flat).to_csv(
                    os.path.join(output_dir, f'{safe_name}_metrics.csv'),
                    header=['value']
                )

            # 持仓
            positions = result.get('positions')
            if positions:
                pos_rows = []
                for dt, stocks in positions.items():
                    pos_rows.append({
                        'date': dt,
                        'n_stocks': len(stocks),
                        'stocks': '|'.join(stocks),
                    })
                pd.DataFrame(pos_rows).to_csv(
                    os.path.join(output_dir, f'{safe_name}_positions.csv'),
                    index=False
                )

        print(f"结果已导出到 {output_dir}/")

    @property
    def results(self) -> Dict[str, Any]:
        """返回所有回测结果。"""
        if not self._ran:
            raise RuntimeError("尚未执行回测。请先调用 run()。")
        return self._results
