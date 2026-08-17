# -*- coding: utf-8 -*-
"""技术指标计算模块：移动平均线 MA、MACD、RSI（Wilder 平滑）。"""

from __future__ import annotations

import pandas as pd


def add_ma(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    """简单移动平均线：MA_N = close 的 N 日均值。"""
    for w in windows:
        df[f"MA{w}"] = df["close"].rolling(w, min_periods=w).mean()
    return df


def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """MACD 指标：
    DIF = EMA(fast) - EMA(slow)；
    DEA = EMA(signal, DIF)；
    柱 = 2 × (DIF - DEA)（与国内行情软件取值习惯一致）。
    """
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["DIF"] = ema_fast - ema_slow
    df["DEA"] = df["DIF"].ewm(span=signal, adjust=False).mean()
    df["MACD"] = (df["DIF"] - df["DEA"]) * 2
    return df


def add_rsi(df: pd.DataFrame, period=14) -> pd.DataFrame:
    """RSI（Wilder 平滑）：基于平均涨幅与平均跌幅的相对强弱指标，取值 0~100。"""
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder 平滑等价于 alpha=1/period 的指数加权
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, 1e-12)
    df[f"RSI{period}"] = 100 - 100 / (1 + rs)
    return df


def add_all(df: pd.DataFrame, ma_windows=(5, 10, 20),
            macd_params=(12, 26, 9), rsi_period=14) -> pd.DataFrame:
    """一次性计算全部技术指标（在副本上操作，不污染原始数据）。"""
    out = df.copy()
    add_ma(out, ma_windows)
    add_macd(out, *macd_params)
    add_rsi(out, rsi_period)
    return out
