# -*- coding: utf-8 -*-
"""量化统计模块：区间收益、极值、最大回撤、年化波动率、成交量与 RSI 状态。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_stats(df: pd.DataFrame, rsi_period=14, trading_days=252) -> dict:
    """基于分析窗口数据计算统计指标，返回 dict（含可供解读的文字结论）。"""
    close = df["close"]
    last_close = float(close.iloc[-1])

    # ---- RSI 超买超卖判断 ----
    rsi_col = f"RSI{rsi_period}"
    end_rsi = float(df[rsi_col].dropna().iloc[-1])
    if end_rsi > 70:
        rsi_signal = "> 70，处于超买区间，短期回调风险较高"
    elif end_rsi < 30:
        rsi_signal = "< 30，处于超卖区间，存在超跌反弹机会"
    else:
        rsi_signal = "位于 30~70 之间，无明显超买超卖"

    # ---- 风险指标：最大回撤 / 年化波动率 ----
    daily_ret = close.pct_change().dropna()
    drawdown = close / close.cummax() - 1
    max_dd_idx = drawdown.idxmin()
    annual_vol = float(daily_ret.std(ddof=1) * np.sqrt(trading_days) * 100)

    # ---- MACD / 均线状态文字解读 ----
    dif, dea, macd = float(df["DIF"].iloc[-1]), float(df["DEA"].iloc[-1]), float(df["MACD"].iloc[-1])
    if dif > dea and macd > 0:
        macd_status = f"DIF({dif:.3f}) > DEA({dea:.3f})，MACD 红柱，多头格局"
    elif dif < dea and macd < 0:
        macd_status = f"DIF({dif:.3f}) < DEA({dea:.3f})，MACD 绿柱，空头格局"
    else:
        macd_status = f"DIF({dif:.3f}) / DEA({dea:.3f})，动能转换中"

    mas = [float(df[f"MA{w}"].iloc[-1]) for w in (5, 10, 20)]
    if last_close > mas[0] > mas[1] > mas[2]:
        ma_status = "收盘价 > MA5 > MA10 > MA20，均线多头排列"
    elif last_close < mas[0] < mas[1] < mas[2]:
        ma_status = "收盘价 < MA5 < MA10 < MA20，均线空头排列"
    else:
        ma_status = "均线相互缠绕，处于震荡格局"

    return {
        "start": df.index[0].date(),
        "end": df.index[-1].date(),
        "trading_days": len(df),
        "period_return_pct": (last_close / float(close.iloc[0]) - 1) * 100,
        "high": float(close.max()),
        "high_date": close.idxmax().date(),
        "low": float(close.min()),
        "low_date": close.idxmin().date(),
        "max_drawdown_pct": float(drawdown.min()) * 100,
        "max_dd_date": max_dd_idx.date(),
        "annual_vol_pct": annual_vol,
        "avg_volume": float(df["volume"].mean()),
        "last_close": last_close,
        "end_rsi": end_rsi,
        "rsi_signal": rsi_signal,
        "macd_status": macd_status,
        "ma_status": ma_status,
    }


def stats_table(stats: dict) -> pd.DataFrame:
    """把统计结果转成结构化表格（控制台打印与报告共用）。"""
    s = stats
    rows = [
        ("分析区间", f"{s['start']} ~ {s['end']}", f"共 {s['trading_days']} 个交易日"),
        ("区间涨跌幅", f"{s['period_return_pct']:+.2f}%", "期末收盘 / 期初收盘 - 1（前复权）"),
        ("区间最高价", f"{s['high']:.2f} 元", f"出现于 {s['high_date']}"),
        ("区间最低价", f"{s['low']:.2f} 元", f"出现于 {s['low_date']}"),
        ("最大回撤", f"{s['max_drawdown_pct']:.2f}%", f"自区间最高点的最大回撤（{s['max_dd_date']}）"),
        ("年化波动率", f"{s['annual_vol_pct']:.2f}%", "日收益率标准差 × √252"),
        ("平均日成交量", f"{s['avg_volume'] / 1e4:.2f} 万手", "区间内日成交量均值"),
        ("期末 RSI(14)", f"{s['end_rsi']:.1f}", s["rsi_signal"]),
        ("MACD 状态", s["macd_status"], "DIF 与 DEA 相对位置"),
        ("均线状态", s["ma_status"], "收盘价与 MA5/MA10/MA20 排列"),
    ]
    return pd.DataFrame(rows, columns=["指标", "数值", "说明"])
