#!/usr/bin/env python3
"""
A 股行情数据获取脚本
=====================
从 AKShare 拉取中证 1000 成分股日线数据，导出为 BacktestEnv 可用的 pickle 文件。

用法:
    # 安装依赖
    pip install akshare pyarrow

    # 下载中证 1000 全部成分股（默认从 2020-01-01 至今）
    python scripts/fetch_data.py

    # 指定指数和日期范围
    python scripts/fetch_data.py --index csi500 --start 2022-01-01 --end 2025-12-31

    # 指定输出路径
    python scripts/fetch_data.py -o data/csi1000.pkl

支持指数:
    csi1000  — 中证 1000 (默认)
    csi500   — 中证 500
    csi300   — 沪深 300
    zz500    — 中证 500 (备选)
    hs300    — 沪深 300 (备选)

输出格式:
    单个 .pkl 文件，内含 dict，BacktestEnv 可直接加载。
"""

import os
import sys
import argparse
import pickle
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

# AKShare 的 sleep 警告比较吵
warnings.filterwarnings('ignore')


# ================================================================
# 指数成分股获取
# ================================================================

INDEX_MAP = {
    'csi1000': '000852',   # 中证 1000
    'csi500':  '000905',   # 中证 500
    'csi300':  '000300',   # 沪深 300
    'zz500':   '000905',
    'hs300':   '000300',
}


def get_index_constituents(index_code: str) -> list[str]:
    """获取指数最新成分股列表。"""
    import akshare as ak

    try:
        # 中证指数官网
        if index_code in ('000852', '000905', '000300'):
            df = ak.index_stock_cons_csindex(symbol=index_code)
            if '成分券代码' in df.columns:
                codes = df['成分券代码'].dropna().astype(str).tolist()
            elif 'stock_code' in df.columns:
                codes = df['stock_code'].dropna().astype(str).tolist()
            else:
                codes = df.iloc[:, 0].dropna().astype(str).tolist()
            return codes
    except Exception as e:
        print(f"  [WARN] 中证官网获取失败 ({e})，尝试东方财富备用接口...")

    # 备用：东方财富指数成分股
    try:
        df = ak.index_stock_cons(symbol=index_code)
        if '品种代码' in df.columns:
            codes = df['品种代码'].dropna().astype(str).tolist()
        elif 'stock_code' in df.columns:
            codes = df['stock_code'].dropna().astype(str).tolist()
        else:
            codes = df.iloc[:, 0].dropna().astype(str).tolist()
        return codes
    except Exception as e:
        raise RuntimeError(
            f"无法获取指数 {index_code} 的成分股。"
            f"请检查 akshare 版本或网络。\n"
            f"  错误: {e}"
        )


# ================================================================
# 个股日线数据获取
# ================================================================

def fetch_stock_daily(
    stock_code: str,
    start_date: str,
    end_date: str,
    adjust: str = 'qfq',   # qfq=前复权, hfq=后复权, ''=不复权
) -> Optional[pd.DataFrame]:
    """
    拉取单只股票的日线行情。

    Returns
    -------
    DataFrame with columns:
        date, open, high, low, close, volume, amount, turnoverrate
    失败返回 None。
    """
    import akshare as ak

    # AKShare 股票代码格式处理
    code_raw = stock_code
    if '.' not in code_raw:
        # 判断交易所
        if code_raw.startswith(('6', '9')):
            symbol = f'{code_raw}.SH'
        else:
            symbol = f'{code_raw}.SZ'
    else:
        symbol = code_raw

    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period='daily',
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        if df is None or df.empty:
            return None

        # 标准化列名
        col_map = {
            '日期': 'date',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            '成交额': 'amount',
            '换手率': 'turnoverrate',
        }
        df = df.rename(columns=col_map)

        # 确保有 date 列
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

        # 保留需要的列
        keep = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnoverrate']
        df = df[[c for c in keep if c in df.columns]]

        return df

    except Exception as e:
        print(f"  [SKIP] {symbol}: {e}")
        return None


# ================================================================
# 市值数据获取（盘后静态数据）
# ================================================================

def fetch_market_cap(stock_codes: list[str], date: str = None) -> pd.Series:
    """
    获取股票总市值。

    通过 AKShare 的 stock_zh_a_spot 获取最新总市值，
    然后用后复权价格推算历史市值（简化处理）。

    如果只需要回测因子（不需要真实基本面），可以用随机市值替代。
    """
    import akshare as ak

    try:
        df = ak.stock_zh_a_spot()
        cap_map = {}
        for _, row in df.iterrows():
            code = str(row.get('代码', ''))
            mcap = row.get('总市值', np.nan)
            if code and not pd.isna(mcap):
                cap_map[code] = float(mcap)
        return cap_map
    except Exception as e:
        print(f"  [WARN] 市值获取失败: {e}")
        return {}


# ================================================================
# 数据聚合：个股 → 面板 DataFrame
# ================================================================

def aggregate_to_panel(
    stock_codes: list[str],
    start_date: str,
    end_date: str,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    将多只股票的日线数据聚合为面板 (date × stock) 的 DataFrame dict。

    Returns
    -------
    dict with keys: close, open, high, low, volume, amount,
                    market_cap, total_turnover, turnoverrate
    """
    fields = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnoverrate']
    panel = {k: {} for k in fields}

    n_total = len(stock_codes)
    n_success = 0

    for i, code in enumerate(stock_codes):
        if verbose and (i % 50 == 0 or i == n_total - 1):
            print(f"  [{i+1}/{n_total}] 获取中... ({n_success} 成功)", end='\r')

        df = fetch_stock_daily(code, start_date, end_date)
        if df is None or df.empty:
            continue

        n_success += 1
        for field in fields:
            if field in df.columns:
                panel[field][code] = df[field]

    print(f"\n  完成: {n_success}/{n_total} 只股票成功获取")

    # 构建面板 DataFrame
    result = {}
    for field in fields:
        if panel[field]:
            result[field] = pd.DataFrame(panel[field]).sort_index()
        else:
            result[field] = pd.DataFrame()

    # 补充缺失字段
    # total_turnover = volume * close（近似成交股数×价格）
    if 'total_turnover' not in result or result['total_turnover'].empty:
        if 'close' in result and 'volume' in result:
            result['total_turnover'] = (
                result['close'] * result['volume']
            ).reindex_like(result['close'])

    # market_cap — 如果有的话从 akshare 获取，否则用近似值
    if 'market_cap' not in result or result['market_cap'].empty:
        cap_info = fetch_market_cap(stock_codes)
        if cap_info:
            # 构建市值 DataFrame
            common_codes = [c for c in result['close'].columns if c in cap_info]
            cap_df = pd.DataFrame(
                {c: cap_info[c] for c in common_codes},
                index=result['close'].index
            ).ffill()
            result['market_cap'] = cap_df
        else:
            # 回退：用成交额 / 换手率 粗略估算
            if 'amount' in result and 'turnoverrate' in result:
                result['market_cap'] = (
                    result['amount'] / (result['turnoverrate'] / 100 + 1e-6)
                )
            else:
                print("  [WARN] 无法获取市值数据，因子中性化会受影响。")

    return result


# ================================================================
# 主入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='A 股行情数据获取 — 导出 BacktestEnv pickle'
    )
    parser.add_argument('--index', default='csi1000',
                       choices=list(INDEX_MAP.keys()),
                       help='目标指数 (默认 csi1000)')
    parser.add_argument('--start', default='2020-01-01',
                       help='起始日期 YYYY-MM-DD (默认 2020-01-01)')
    parser.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'),
                       help='结束日期 (默认今天)')
    parser.add_argument('-o', '--output', default='data/csi1000.pkl',
                       help='输出 pickle 路径 (默认 data/csi1000.pkl)')
    parser.add_argument('--no-pickle', action='store_true',
                       help='不保存 pickle，仅打印数据摘要')
    args = parser.parse_args()

    print("=" * 60)
    print(f"  A 股数据获取 — {args.index}")
    print("=" * 60)

    # 1. 获取成分股
    print(f"\n[1/3] 获取 {args.index} 成分股...")
    index_code = INDEX_MAP[args.index]
    stocks = get_index_constituents(index_code)
    print(f"  获取到 {len(stocks)} 只成分股")

    if len(stocks) < 10:
        print("  [ERROR] 成分股不足，退出。")
        sys.exit(1)

    # 2. 拉取日线
    print(f"\n[2/3] 拉取 {args.start} ~ {args.end} 日线数据...")
    panel = aggregate_to_panel(stocks, args.start, args.end)
    close = panel.get('close', pd.DataFrame())

    if close.empty:
        print("  [ERROR] 未获取到任何数据。请检查网络或 akshare 版本。")
        sys.exit(1)

    print(f"\n  数据摘要:")
    print(f"    日期: {close.index[0].date()} ~ {close.index[-1].date()} "
          f"({len(close)} 天)")
    print(f"    股票: {len(close.columns)} 只")
    for name, df in panel.items():
        if not df.empty:
            nan_pct = df.isna().mean().mean() * 100
            print(f"    {name:20s} — shape={str(df.shape):12s} NaN={nan_pct:.1f}%")

    # 3. 保存
    if not args.no_pickle:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'wb') as f:
            pickle.dump(panel, f)
        size_mb = os.path.getsize(args.output) / 1024 / 1024
        print(f"\n[3/3] 已保存 → {args.output} ({size_mb:.1f} MB)")
        print(f"\n使用方式:")
        print(f"  from backtest import BacktestEnv")
        print(f"  env = BacktestEnv('{args.output}')")
        print(f"  env.info()")


if __name__ == '__main__':
    main()
