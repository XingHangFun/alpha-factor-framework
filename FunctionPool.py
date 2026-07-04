"""
量化因子函数库 FunctionPool
============================
作为回测框架 BaseFactorGenerator 的 mixin, 提供因子预处理、截面/时序变换、Regime检测等通用方法。

使用方式: class MyFactorGen(FunctionPool, SomeBase): ...
框架保证 self 上已注入: close, open, high, low, volume, amount, market_cap, total_turnover, turnoverrate
"""

import numpy as np
import pandas as pd


class FunctionPool:
    """
    因子工具集 Mixin.
    不依赖具体基类, 通过 self 访问行情/基本面 DataFrame (index=date, columns=stock).
    """

    # ================================================================
    # 一、截面变换
    # ================================================================

    def cs_zscore(self, x: pd.DataFrame, clip: float = None) -> pd.DataFrame:
        """截面 z-score 标准化."""
        mu = x.mean(axis=1)
        sigma = x.std(axis=1).replace(0, np.nan)
        result = x.sub(mu, axis=0).div(sigma, axis=0)
        if clip is not None:
            result = result.clip(-clip, clip)
        return result

    # 别名 — 兼容回测框架 DataContainer 的 _cs_zscore / _cs_rank 命名
    _cs_zscore = cs_zscore

    def cs_rank(self, x: pd.DataFrame) -> pd.DataFrame:
        """截面 rank 百分位 (0~1)."""
        return x.rank(axis=1, pct=True)

    _cs_rank = cs_rank

    def cs_rank_uniform(self, x: pd.DataFrame) -> pd.DataFrame:
        """截面 rank 映射到 [-1, 1]."""
        r = x.rank(axis=1, pct=True)
        return (r - 0.5) * 2.0

    def cs_winsorize(self, x: pd.DataFrame, sigma: float = 3.0) -> pd.DataFrame:
        """截面 winsorize — 用 mean±sigma*std 截断, 比MAD简洁."""
        mu = x.mean(axis=1)
        std = x.std(axis=1).replace(0, np.nan)
        upper = mu + sigma * std
        lower = mu - sigma * std
        return x.clip(lower=lower, upper=upper, axis=0)

    def cs_mad_clip(self, x: pd.DataFrame, n_sigma: float = 3.0) -> pd.DataFrame:
        """截面 MAD 去极值. 常数1.4826将MAD转为σ等价值."""
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

    # ================================================================
    # 二、时序变换
    # ================================================================

    def ts_zscore(self, x: pd.DataFrame, window: int, clip: float = None) -> pd.DataFrame:
        """时序滚动 z-score."""
        mu = x.rolling(window, min_periods=max(5, window // 2)).mean()
        sigma = x.rolling(window, min_periods=max(5, window // 2)).std().replace(0, np.nan)
        result = x.sub(mu).div(sigma)
        if clip is not None:
            result = result.clip(-clip, clip)
        return result

    def ts_rank(self, x: pd.DataFrame, window: int) -> pd.DataFrame:
        """时序滚动 rank 百分位."""
        return x.rolling(window, min_periods=max(5, window // 2)).apply(
            lambda s: s.rank(pct=True).iloc[-1], raw=False
        )

    def ts_corr(self, x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
        """时序滚动相关系数 corr(x,y), 矢量化实现."""
        min_p = max(10, window // 2)
        mx = x.rolling(window, min_periods=min_p).mean()
        my = y.rolling(window, min_periods=min_p).mean()
        cov = ((x - mx) * (y - my)).rolling(window, min_periods=min_p).mean()
        sx = x.rolling(window, min_periods=min_p).std()
        sy = y.rolling(window, min_periods=min_p).std()
        return cov / (sx * sy).replace(0, np.nan)

    def ts_regression_resid(self, y: pd.DataFrame, x: pd.DataFrame, window: int) -> pd.DataFrame:
        """时序滚动OLS残差, 矢量化实现. resid = y - (alpha + beta * x)."""
        min_p = max(10, window // 2)
        mx = x.rolling(window, min_periods=min_p).mean()
        my = y.rolling(window, min_periods=min_p).mean()
        cov = ((x - mx) * (y - my)).rolling(window, min_periods=min_p).mean()
        var_x = x.rolling(window, min_periods=min_p).var()
        beta = cov / var_x.replace(0, np.nan)
        alpha = my - beta * mx
        return y - (alpha + beta * x)

    def ts_delay(self, x: pd.DataFrame, periods: int) -> pd.DataFrame:
        """滞后."""
        return x.shift(periods)

    def ts_delta(self, x: pd.DataFrame, periods: int) -> pd.DataFrame:
        """N期差分."""
        return x - x.shift(periods)

    def ts_roc(self, x: pd.DataFrame, periods: int) -> pd.DataFrame:
        """变化率."""
        return x / x.shift(periods).replace(0, np.nan) - 1

    # ================================================================
    # 三、因子预处理流水线 (统一入口, 对应 rule_v11 第二章三步清洗)
    # ================================================================

    def factor_process(
        self,
        factor: pd.DataFrame,
        # --- 去极值 ---
        outlier: str = 'mad',              # 'mad' | 'winsorize' | None
        outlier_sigma: float = 3.0,        # 去极值倍数 (MAD用1.4826缩放, winsorize直接用σ)
        # --- 中性化 ---
        neutralize: bool = True,           # 是否做中性化
        neutralize_by: str = 'market_cap', # 'market_cap' | 'industry' | None
        industry: pd.DataFrame = None,     # 行业标签, neutralize_by='industry' 时需要
        # --- 标准化 ---
        zscore: bool = True,               # 截面 Z-Score 标准化
        # --- 平滑 ---
        smooth: int = 5,                   # EWM span, 0=不平滑
        # --- 输出 ---
        rank_output: bool = False,         # 输出映射到 [-1,1] 的截面 rank
        clip: float = 3.0,                # 最终截断区间 [-clip, clip]
        fillna: bool = True,              # 缺失值填零
    ) -> pd.DataFrame:
        """
        因子标准化流水线 — 串联 rule_v11 第二章的三步清洗.

        流水线顺序:
          NaN/Inf 清理 → 去极值(outlier) → 中性化(neutralize) → ZScore → 平滑 → Rank → 截断 → 填零

        每一步都可以通过参数关闭, 默认开启全部标准清洗.
        """
        # 0. NaN / Inf 清理
        factor = factor.replace([np.inf, -np.inf], np.nan)

        # 1. 去极值
        if outlier == 'mad':
            factor = self.cs_mad_clip(factor, n_sigma=outlier_sigma)
        elif outlier == 'winsorize':
            factor = self.cs_winsorize(factor, sigma=outlier_sigma)

        # 2. 中性化
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

        # 3. ZScore 截面标准化
        if zscore:
            factor = self.cs_zscore(factor)

        # 4. EWM 平滑 (降换手)
        if smooth > 0:
            factor = factor.ewm(span=smooth, min_periods=max(3, smooth // 2), adjust=False).mean()

        # 5. Rank 映射 (可选)
        if rank_output:
            factor = self.cs_rank_uniform(factor)

        # 6. 最终截断
        if clip is not None and clip > 0:
            factor = factor.clip(-clip, clip)

        # 7. 缺失值填零
        if fillna:
            factor = factor.fillna(0)
        return factor

    # ================================================================
    # 四、Regime 检测 (市场状态)
    # ================================================================

    def vol_regime(self, window: int = 60) -> pd.Series:
        """
        波动率区间 (0~1, 市场级).
        高值 = 当前处于历史高波动区间.
        """
        mkt_ret = self.close.pct_change().mean(axis=1)  # 等权市场收益
        vol = mkt_ret.rolling(window, min_periods=20).std() * np.sqrt(252)
        return vol.rank(pct=True).fillna(0.5)

    def panic_state(self, window: int = 60) -> pd.Series:
        """
        恐慌状态 (0~1, 市场级).
        综合下行波动率占比 + 回撤深度 + Chopiness.
        """
        mkt_ret = self.close.pct_change().mean(axis=1)
        # 下行波动率占比
        dn = mkt_ret.clip(upper=0)
        dn_vol = dn.rolling(window, min_periods=20).std()
        tv = mkt_ret.rolling(window, min_periods=20).std()
        vol_ratio = (dn_vol / tv.replace(0, np.nan)).fillna(0.5)
        # 回撤深度
        mkt_price = self.close.mean(axis=1)
        dd = (mkt_price / mkt_price.rolling(window, min_periods=20).max() - 1).clip(upper=0)
        # Chopiness
        am = mkt_ret.abs().rolling(20, min_periods=10).mean()
        rm = mkt_ret.rolling(20, min_periods=10).mean()
        cp = (am / (rm.abs() + am * 0.01)).clip(1, 20)
        cn = 1 - 1 / cp
        p = 0.35 * vol_ratio.rank(pct=True) + 0.35 * (-dd).rank(pct=True) + 0.30 * cn.rank(pct=True)
        return p.fillna(0.5).clip(0, 1).ewm(span=5, min_periods=3).mean()

    def trend_strength(self, window: int = 60) -> pd.Series:
        """
        趋势强度 (0~1, 市场级).
        高值 = 市场处于强趋势状态 (无论方向), 适合动量因子.
        低值 = 震荡市, 适合反转因子.
        """
        mkt_ret = self.close.pct_change().mean(axis=1)
        # 路径效率: |累积收益| / 路径长度
        cum = (1 + mkt_ret).rolling(window, min_periods=20).apply(
            lambda x: x.prod(), raw=True
        )
        path = mkt_ret.abs().rolling(window, min_periods=20).sum()
        efficiency = (abs(cum - 1) / path.replace(0, np.nan)).fillna(0)
        return efficiency.rank(pct=True).fillna(0.5)

    def market_dispersion(self, window: int = 20) -> pd.Series:
        """
        截面离散度 (市场级).
        高值 = 个股分化大, Alpha机会多; 低值 = 同涨同跌.
        """
        ret = self.close.pct_change()
        dispersion = ret.std(axis=1).rolling(window, min_periods=10).mean()
        return dispersion.rank(pct=True).fillna(0.5)

    # ================================================================
    # 五、复合与信号构造
    # ================================================================

    def composite_equal(self, factors: list[pd.DataFrame], smooth: int = 5) -> pd.DataFrame:
        """多因子等权复合 — 按列名对齐, 不做位置假设."""
        normalized = [self.cs_zscore(f) for f in factors]
        raw = sum(normalized) / len(normalized)
        return self.cs_zscore(raw).ewm(span=smooth, min_periods=3).mean()

    def composite_ic_weighted(self, factors: list[pd.DataFrame],
                               ic_series: list[pd.Series], smooth: int = 5) -> pd.DataFrame:
        """IC加权复合 — ic_series的绝对值越大权重越高."""
        w = np.array([abs(ic.mean()) for ic in ic_series])
        w = w / w.sum()
        raw = sum(w[i] * self.cs_zscore(f) for i, f in enumerate(factors))
        return self.cs_zscore(raw).ewm(span=smooth, min_periods=3).mean()

    def composite_bayesian(
        self,
        factors: list[pd.DataFrame],
        forward_returns: pd.DataFrame,
        smooth: int = 5,
        fit_window: int = None,
    ) -> dict:
        """
        贝叶斯线性组合 — Bayesian Ridge Regression.

        使用 scikit-learn BayesianRidge，自动通过 evidence approximation
        选择正则化强度。先验: w ~ N(0, λ⁻¹I)，λ ~ Gamma。

        相比等权和 IC 加权，贝叶斯组合的优势:
        - 自动收缩: 噪声因子的权重自动趋近于零
        - 不确定性量化: 每个权重有后验均值和标准差
        - 无超参调优: λ 由 evidence maximization 自动选择

        Parameters
        ----------
        factors : list of pd.DataFrame
            因子值 (index=date, columns=stock)，每个因子应为已标准化的截面值。
        forward_returns : pd.DataFrame
            未来收益 (index=date, columns=stock)，回归目标。
        smooth : int
            输出因子 EWM 平滑 span。
        fit_window : int, optional
            滚动拟合窗口长度。None = 全时段一次拟合。
            设置后每个窗口独立拟合，权重随时间变化。

        Returns
        -------
        dict:
            'composite': pd.DataFrame — 复合因子值 (index=date, columns=stock)
            'weights': list of dict — [{mean, std, index}, ...] 每个因子的后验权重
            'intercept': float — 截距项
            'alpha': float — 权重先验精度 λ
            'lambda': float — 噪声精度 β (=1/σ²)
            'scores': dict — {'r2': float, 'mse': float}
        """
        try:
            from sklearn.linear_model import BayesianRidge
        except ImportError:
            raise ImportError(
                "composite_bayesian 需要 scikit-learn。"
                "请安装: pip install scikit-learn"
            )

        # 1. 对齐数据: 取因子和收益的共同日期/股票
        common_cols = factors[0].columns
        for fv in factors[1:]:
            common_cols = common_cols.intersection(fv.columns)
        common_cols = common_cols.intersection(forward_returns.columns)
        if len(common_cols) < 20:
            raise ValueError(f"共同股票数量不足: {len(common_cols)}，需要至少 20 只。")

        common_dates = factors[0].index.intersection(forward_returns.index)
        for fv in factors[1:]:
            common_dates = common_dates.intersection(fv.index)
        if len(common_dates) < 50:
            raise ValueError(f"共同日期不足: {len(common_dates)}，需要至少 50 天。")

        # 对齐所有数据
        aligned_factors = [fv.reindex(index=common_dates, columns=common_cols)
                          for fv in factors]
        fwd_ret = forward_returns.reindex(index=common_dates, columns=common_cols)

        n_factors = len(aligned_factors)
        n_dates = len(common_dates)
        n_stocks = len(common_cols)

        if fit_window is None:
            fit_window = n_dates  # 全时段一次拟合

        # 2. 构建回归数据: y = forward_ret, X = [f1, f2, ..., fk] (堆叠所有日期×股票)
        #    每个因子先做截面 zscore
        normalized = [self.cs_zscore(fv).fillna(0) for fv in aligned_factors]

        # 3. 拟合贝叶斯回归
        if fit_window >= n_dates:
            # 全时段一次拟合
            X, y = self._stack_for_regression(normalized, fwd_ret)
            model, scores = self._fit_bayesian_ridge(X, y)
            weights = [
                {
                    'index': i,
                    'mean': float(model.coef_[i]),
                    'std': float(np.sqrt(1.0 / model.lambda_))  # approximate
                    if hasattr(model, 'lambda_') else None,
                }
                for i in range(n_factors)
            ]
            intercept = float(model.intercept_)
            alpha = float(model.alpha_) if hasattr(model, 'alpha_') else None
            lambda_ = float(model.lambda_) if hasattr(model, 'lambda_') else None

            # 生成复合因子
            raw_composite = sum(w['mean'] * normalized[i] for i, w in enumerate(weights))
            composite = self.cs_zscore(raw_composite).ewm(
                span=smooth, min_periods=max(3, smooth // 2), adjust=False
            ).mean()

        else:
            # 滚动窗口拟合
            weights = []
            composite = pd.DataFrame(np.nan, index=common_dates, columns=common_cols)
            scores_list = []
            intercept_list = []
            alpha_list = []
            lambda_list = []

            for t in range(fit_window, n_dates):
                train_slice = slice(t - fit_window, t)
                X, y = self._stack_for_regression(
                    [fv.iloc[train_slice] for fv in normalized],
                    fwd_ret.iloc[train_slice]
                )
                model, sc = self._fit_bayesian_ridge(X, y)
                scores_list.append(sc)
                intercept_list.append(float(model.intercept_))
                if hasattr(model, 'alpha_'):
                    alpha_list.append(float(model.alpha_))
                if hasattr(model, 'lambda_'):
                    lambda_list.append(float(model.lambda_))

                w = model.coef_
                raw = sum(float(w[i]) * normalized[i].iloc[t] for i in range(n_factors))
                composite.iloc[t] = raw

            # 聚合结果
            composite = self.cs_zscore(composite).fillna(0).ewm(
                span=smooth, min_periods=max(3, smooth // 2), adjust=False
            ).mean()

            avg_w = np.mean([list(scores_list[0].get('coef_', []))], axis=0) if scores_list else np.zeros(n_factors)
            weights = [
                {'index': i, 'mean': float(avg_w[i]) if i < len(avg_w) else 0.0, 'std': None}
                for i in range(n_factors)
            ]
            intercept = float(np.mean(intercept_list)) if intercept_list else 0.0
            alpha = float(np.mean(alpha_list)) if alpha_list else None
            lambda_ = float(np.mean(lambda_list)) if lambda_list else None
            scores = {
                'r2': float(np.mean([s.get('r2', 0) for s in scores_list])),
                'mse': float(np.mean([s.get('mse', 0) for s in scores_list])),
            }

        # 4. 构建权重诊断
        for w in weights:
            w['abs_weight'] = abs(w['mean'])
        total_abs = sum(w['abs_weight'] for w in weights)
        for w in weights:
            w['pct_contribution'] = w['abs_weight'] / total_abs if total_abs > 0 else 0.0

        return {
            'composite': composite,
            'weights': weights,
            'intercept': intercept,
            'alpha': alpha,
            'lambda': lambda_,
            'scores': scores,
        }

    def _stack_for_regression(
        self,
        normalized_factors: list[pd.DataFrame],
        forward_returns: pd.DataFrame,
    ) -> tuple:
        """
        将因子和收益堆叠为 (N, K) 的回归矩阵。

        每个 (date, stock) 对是一个样本。
        y = forward_return
        X = [f1_value, f2_value, ..., fk_value]
        """
        k = len(normalized_factors)
        # 收集所有 (date, stock) 对
        y_vals = []
        X_vals = []

        flat_fwd = forward_returns.values
        flat_factors = [fv.values for fv in normalized_factors]

        mask = ~np.isnan(flat_fwd)
        for i in range(k):
            mask &= ~np.isnan(flat_factors[i])

        y_vals = flat_fwd[mask]
        X_vals = np.column_stack([ff[mask] for ff in flat_factors])

        return X_vals, y_vals

    def _fit_bayesian_ridge(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """拟合 BayesianRidge 并返回模型和诊断。"""
        from sklearn.linear_model import BayesianRidge
        from sklearn.metrics import r2_score, mean_squared_error

        if len(y) < 10:
            # 样本太少，返回零权重
            class _Dummy:
                coef_ = np.zeros(X.shape[1])
                intercept_ = 0.0
                alpha_ = 1.0
                lambda_ = 1.0
            return _Dummy(), {'r2': 0.0, 'mse': 0.0}

        model = BayesianRidge(
            max_iter=300,
            tol=1e-3,
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6,
            fit_intercept=True,
        )
        model.fit(X, y)

        y_pred = model.predict(X)
        scores = {
            'r2': float(r2_score(y, y_pred)),
            'mse': float(mean_squared_error(y, y_pred)),
            'coef_': model.coef_.tolist(),
        }

        return model, scores

    def regime_switch(
        self,
        trend_factor: pd.DataFrame,   # MT: 动量趋势类
        reversal_factor: pd.DataFrame, # MR: 反转震荡类
        switch_signal: pd.Series = None,
        window: int = 60,
    ) -> pd.DataFrame:
        """
        MTR (连续版): 自动切换动量/反转模式.
        switch_signal: 趋势强度 (0~1), 连续权重混合.
        默认使用 trend_strength 作为切换信号.
        """
        if switch_signal is None:
            switch_signal = self.trend_strength(window)
        sw = self._broadcast(switch_signal)
        raw = sw * self.cs_zscore(trend_factor) + (1 - sw) * self.cs_zscore(reversal_factor)
        return self.cs_zscore(raw)

    def discrete_switch(
        self,
        alpha_factor: pd.DataFrame,
        defense_factor: pd.DataFrame = None,
        enter_signal=None,                  # bool (DataFrame or market-level Series)
        exit_signal=None,                   # bool, 默认 = ~enter_signal
        confirm: int = 1,                   # 连续确认天数, 防单日噪声
    ) -> pd.DataFrame:
        """
        离散状态机: enter_signal 持续 confirm 天后切到 defense,
        exit_signal 触发后回到 alpha. 不传 defense 时默认用大市值防守.

        regime_switch 做连续权重混合, 本方法做硬切换 —
        适合尾部危机保护场景, 牺牲连续性换取明确的风控边界.
        """
        # 默认防守端: 大市值
        if defense_factor is None:
            cap_log = np.log(self.market_cap.replace(0, np.nan))
            defense_factor = self.cs_zscore(cap_log)

        # broadcast market-level Series → DataFrame
        if isinstance(enter_signal, pd.Series):
            enter_signal = self._broadcast(enter_signal)
        if exit_signal is None:
            exit_signal = ~enter_signal
        elif isinstance(exit_signal, pd.Series):
            exit_signal = self._broadcast(exit_signal)

        cols = (alpha_factor.columns
                .intersection(defense_factor.columns)
                .intersection(enter_signal.columns)
                .intersection(exit_signal.columns))

        enter = enter_signal[cols]
        exit_ = exit_signal[cols]

        if confirm > 1:
            enter = enter.rolling(confirm, min_periods=1).sum().ge(confirm)
            exit_ = exit_.rolling(confirm, min_periods=1).sum().ge(confirm)

        state = (enter.where(enter, other=(~exit_).where(exit_))
                     .ffill().fillna(False))

        return alpha_factor[cols].where(~state, defense_factor[cols])

    def _broadcast(self, market_series: pd.Series) -> pd.DataFrame:
        """广播市场级Series到个股DataFrame."""
        return pd.DataFrame(
            {c: market_series.values for c in self.close.columns},
            index=market_series.index
        )

    # ================================================================
    # 六、辅助
    # ================================================================

    def filter_suspend(self, x: pd.DataFrame, threshold: float = 50) -> pd.DataFrame:
        """
        标记停牌/流动性差的股票 (用连续零成交量天数).
        返回 True=正常, False=停牌.
        """
        zero_count = (self.volume == 0).rolling(threshold, min_periods=10).sum()
        return zero_count < threshold * 0.5
