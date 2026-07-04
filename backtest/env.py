"""
BacktestEnv — 回测数据容器
==========================
继承 FunctionPool，从 pickle 文件加载行情数据，注入为 self 属性供因子函数使用。
"""

import os
import pickle
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd

from FunctionPool import FunctionPool


# 因子函数需要的数据属性列表
REQUIRED_ATTRS = [
    'close', 'open', 'high', 'low',
    'volume', 'amount',
    'market_cap', 'total_turnover', 'turnoverrate',
]


class BacktestEnv(FunctionPool):
    """
    回测数据环境，继承 FunctionPool 的所有因子工具方法。

    数据加载支持两种格式:

    1. 单个 pickle 文件，内容为 dict:
       {
           'close':          pd.DataFrame (index=date, columns=stock_code),
           'open':           pd.DataFrame,
           ...,
           'market_cap':     pd.DataFrame,
       }

    2. 目录路径，目录下每个 .pkl 文件名 = 属性名:
       data/
       ├── close.pkl
       ├── open.pkl
       ├── ...
       └── market_cap.pkl

    参数
    ----
    data_path : str
        pickle 文件路径或数据目录路径。
    benchmark_col : str, optional
        用作基准的列名（如 '000852.SH' 中证1000指数）。
        如果为 None 且数据中存在单列索引，将使用等权平均作为基准。
    """

    def __init__(self, data_path: str, benchmark_col: Optional[str] = None):
        # 先加载数据（不调用 super().__init__，FunctionPool 是纯 Mixin）
        self._data_path = data_path
        self._benchmark_col = benchmark_col
        self._raw_data: Dict[str, pd.DataFrame] = {}
        self._load_data(data_path)

        # 校验必需属性
        self._validate_attrs()

        # 注入为 self 属性
        self._inject_attrs()

        # 统一日期和股票代码交集
        self._align_index()

        # 计算基准收益序列
        self._benchmark_returns: Optional[pd.Series] = None
        self._compute_benchmark()

    # ---- 数据加载 ----

    def _load_data(self, data_path: str):
        """从 pickle 文件加载所有数据。"""
        path = Path(data_path)

        if path.is_file():
            # 格式 1: 单个文件包含 dict
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if not isinstance(data, dict):
                raise TypeError(
                    f"pickle 文件内容应为 dict，实际为 {type(data).__name__}。"
                    f"支持的格式: {{'close': DataFrame, 'open': DataFrame, ...}}"
                )
            self._raw_data = {k: v for k, v in data.items() if self._is_valid_df(v)}

        elif path.is_dir():
            # 格式 2: 目录下多个文件
            for f in sorted(path.glob('*.pkl')):
                attr_name = f.stem
                with open(f, 'rb') as fh:
                    data = pickle.load(fh)
                if self._is_valid_df(data):
                    self._raw_data[attr_name] = data

        elif not path.exists():
            raise FileNotFoundError(f"数据路径不存在: {data_path}")
        else:
            raise ValueError(f"不支持的数据路径类型: {data_path}")

    @staticmethod
    def _is_valid_df(obj) -> bool:
        """检查是否为有效的股票数据 DataFrame。"""
        return isinstance(obj, pd.DataFrame) and not obj.empty

    # ---- 属性校验 ----

    def _validate_attrs(self):
        """检查必需属性是否都已加载。"""
        missing = set(REQUIRED_ATTRS) - set(self._raw_data.keys())
        if missing:
            raise KeyError(
                f"缺少必需的数据字段: {missing}。"
                f"已加载: {list(self._raw_data.keys())}。"
                f"需要: {REQUIRED_ATTRS}"
            )

    def _inject_attrs(self):
        """将原始数据注入为 self 属性。"""
        for attr in REQUIRED_ATTRS:
            df = self._raw_data[attr].copy()
            # 确保 index 为 DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            # 确保列名为字符串
            df.columns = df.columns.astype(str)
            setattr(self, attr, df.sort_index())

    # ---- 数据对齐 ----

    def _align_index(self):
        """
        对齐所有 DataFrame 的日期和股票代码。
        只保留所有价格数据共同覆盖的日期和股票。
        """
        # 日期对齐（以 close 为基准，因为核心定价数据）
        base_dates = self.close.index

        # 股票对齐：取所有价格字段的列交集
        price_cols = set(self.close.columns)
        for attr in ['open', 'high', 'low']:
            price_cols &= set(getattr(self, attr).columns)
        stock_cols = sorted(price_cols)

        if len(stock_cols) == 0:
            raise ValueError("所有价格字段的股票代码交集为空，请检查数据。")

        # 截取所有属性到对齐的 index/columns
        for attr in REQUIRED_ATTRS:
            df = getattr(self, attr)
            aligned = df.reindex(index=base_dates, columns=stock_cols)
            setattr(self, attr, aligned)

        self._stocks = stock_cols
        self._dates = base_dates

    # ---- 基准收益 ----

    def _compute_benchmark(self):
        """计算基准收益序列（中证1000等权或指定列）。"""
        if self._benchmark_col and self._benchmark_col in self.close.columns:
            bench_close = self.close[self._benchmark_col]
            self._benchmark_returns = bench_close.pct_change().fillna(0)
        else:
            # 默认：全市场等权收益
            self._benchmark_returns = self.close.pct_change().mean(axis=1).fillna(0)

    @property
    def benchmark_returns(self) -> pd.Series:
        """基准日收益序列。"""
        return self._benchmark_returns

    # ---- 属性查询 ----

    @property
    def stocks(self) -> list:
        """股票代码列表。"""
        return self._stocks

    @property
    def dates(self) -> pd.DatetimeIndex:
        """所有交易日。"""
        return self._dates

    @property
    def n_stocks(self) -> int:
        """股票数量。"""
        return len(self._stocks)

    @property
    def n_dates(self) -> int:
        """交易日数量。"""
        return len(self._dates)

    # ---- 切片 ----

    def slice(self, start_date=None, end_date=None):
        """
        返回日期范围内的数据切片（原地修改 — 仅供回测引擎 walk-forward 使用）。

        参数
        ----
        start_date : str or datetime, optional
            起始日期（包含）。
        end_date : str or datetime, optional
            结束日期（包含）。

        返回
        ----
        self，已切片到指定日期范围。
        """
        date_mask = pd.Series(True, index=self._dates)
        if start_date is not None:
            date_mask &= self._dates >= pd.Timestamp(start_date)
        if end_date is not None:
            date_mask &= self._dates <= pd.Timestamp(end_date)

        idx = self._dates[date_mask]
        for attr in REQUIRED_ATTRS:
            df = getattr(self, attr)
            setattr(self, attr, df.loc[idx])

        self._dates = idx
        return self

    # ---- 调试信息 ----

    def info(self):
        """打印数据摘要。"""
        print(f"BacktestEnv @ {self._data_path}")
        print(f"  日期: {self._dates[0].date()} ~ {self._dates[-1].date()} ({self.n_dates} 天)")
        print(f"  股票: {self.n_stocks} 只")
        print(f"  基准: {'指定列 ' + self._benchmark_col if self._benchmark_col else '等权平均'}")
        for attr in REQUIRED_ATTRS:
            df = getattr(self, attr)
            nan_pct = df.isna().mean().mean() * 100
            print(f"  {attr:20s} — shape={str(df.shape):12s} NaN={nan_pct:.1f}%")

    def __repr__(self):
        return f"BacktestEnv(stocks={self.n_stocks}, dates={self.n_dates})"
