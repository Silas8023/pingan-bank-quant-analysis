# -*- coding: utf-8 -*-
"""图表模块：K 线技术分析四联图与双均线回测对比图。

设计要点：
- 300 DPI、16 × 10 英寸专业商务画布，浅灰网格、无多余边框；
- 主图 K 线（A 股红涨绿跌）+ MA5/MA10/MA20；
- 子图从上到下依次为成交量、MACD、RSI(14)，成交量颜色跟随 K 线；
- 主图标注半年最低点、阶段高点与斐波那契支撑 / 压力位。
"""

from __future__ import annotations

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import config as cfg


def setup_chinese_font():
    """注册系统中文字体并设为默认，返回字体名；找不到中文字体时返回 None。"""
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",    # 微软雅黑
        r"C:\Windows\Fonts\simhei.ttf",  # 黑体
        r"C:\Windows\Fonts\simsun.ttc",  # 宋体
    ]
    for path in candidates:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = family
            plt.rcParams["font.size"] = 12
            plt.rcParams["axes.unicode_minus"] = False
            return family
    plt.rcParams["axes.unicode_minus"] = False
    return None


def _x_value(ax, df: pd.DataFrame, date) -> float:
    """兼容 mplfinance 日期数轴 / 整数数轴两种横轴的坐标换算。"""
    if ax.get_xlim()[1] > 1000:      # 日期数轴（约 46000）
        return mdates.date2num(date)
    return df.index.get_loc(date)    # 整数数轴（0..N-1）


def plot_technical_chart(df: pd.DataFrame, title: str, out_path, font_family=None,
                         rsi_period=None):
    """绘制 K 线技术分析四联图：K线+均线 / 成交量 / MACD / RSI。"""
    rsi_period = rsi_period or cfg.RSI_PERIOD
    chart = df.copy()

    # A 股配色：红涨绿跌，成交量继承 K 线颜色
    mc = mpf.make_marketcolors(
        up=cfg.UP_COLOR, down=cfg.DOWN_COLOR, edge="inherit", wick="inherit", volume="inherit"
    )
    style_kwargs = dict(
        base_mpf_style="yahoo",
        marketcolors=mc,
        gridstyle="--",
        gridcolor=cfg.GRID_COLOR,
        facecolor="white",
        figcolor="white",
    )
    if font_family:
        style_kwargs["rc"] = {"font.family": font_family}
    style = mpf.make_mpf_style(**style_kwargs)

    # 各子图叠加对象
    hist_colors = [cfg.UP_COLOR if v >= 0 else cfg.DOWN_COLOR for v in chart["MACD"]]
    apds = [
        mpf.make_addplot(chart[f"MA{w}"], color=cfg.MA_COLORS[w], width=1.4)
        for w in cfg.MA_WINDOWS
    ]
    apds += [
        mpf.make_addplot(chart["MACD"], type="bar", panel=2, color=hist_colors,
                         width=0.8, ylabel="MACD"),
        mpf.make_addplot(chart["DIF"], panel=2, color=cfg.MACD_COLORS["dif"], width=1.0),
        mpf.make_addplot(chart["DEA"], panel=2, color=cfg.MACD_COLORS["dea"], width=1.0),
        mpf.make_addplot(chart[f"RSI{rsi_period}"], panel=3, color=cfg.RSI_COLOR,
                         width=1.2, ylabel=f"RSI({rsi_period})"),
        mpf.make_addplot(pd.Series(cfg.RSI_OVERBOUGHT, index=chart.index), panel=3,
                         color="#bbbbbb", linestyle="--", width=0.8),
        mpf.make_addplot(pd.Series(cfg.RSI_OVERSOLD, index=chart.index), panel=3,
                         color="#bbbbbb", linestyle="--", width=0.8),
    ]

    fig, axes = mpf.plot(
        chart[["open", "high", "low", "close", "volume"]],
        type="candle",
        volume=True,
        addplot=apds,
        style=style,
        title="",
        ylabel="价格（元）",
        ylabel_lower="成交量（手）",
        figsize=cfg.FIG_SIZE,
        panel_ratios=(4, 1, 1.4, 1),
        xrotation=30,
        returnfig=True,
        tight_layout=False,
        datetime_format="%Y-%m-%d",
    )

    main, vol_ax, macd_ax, rsi_ax = axes[0], axes[1], axes[2], axes[3]

    # 统一样式：去掉上/右边框，浅灰虚线网格
    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.grid(True, color=cfg.GRID_COLOR, linestyle="--", linewidth=0.7, alpha=0.8)

    # X 轴月度刻度（仅日期数轴生效，整数数轴场景跳过）
    if axes[0].get_xlim()[1] > 1000:
        for ax in axes:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # ---- 关键点位标注：阶段高点 / 半年最低点 ----
    low_pos, high_pos = chart["low"].idxmin(), chart["high"].idxmax()
    low_val, high_val = float(chart.loc[low_pos, "low"]), float(chart.loc[high_pos, "high"])
    pad = (high_val - low_val) * 0.12
    main.set_ylim(low_val - pad, high_val + pad)
    x_end = main.get_xlim()[1]
    low_x, high_x = _x_value(main, chart, low_pos), _x_value(main, chart, high_pos)

    main.annotate(
        f"阶段高点 {high_val:.2f}", xy=(high_x, high_val),
        xytext=(high_x, high_val + pad * 0.5),
        arrowprops=dict(arrowstyle="->", color=cfg.UP_COLOR, lw=1.2),
        color="#c0392b", fontsize=11, fontweight="bold", ha="center",
    )
    main.annotate(
        f"半年最低点 {low_val:.2f}", xy=(low_x, low_val),
        xytext=(low_x, low_val - pad * 0.5),
        arrowprops=dict(arrowstyle="->", color=cfg.DOWN_COLOR, lw=1.2),
        color="#1e8449", fontsize=11, fontweight="bold", ha="center",
    )

    # ---- 斐波那契支撑 / 压力位虚线 ----
    span = high_val - low_val
    levels = [
        ("阶段高点（压力位）", high_val, cfg.UP_COLOR),
        ("0.618 关键位", high_val - 0.618 * span, "#8e44ad"),
        ("0.5 中轴", high_val - 0.5 * span, "#f39c12"),
        ("0.382 关键位", high_val - 0.382 * span, "#16a085"),
        ("半年最低点（支撑位）", low_val, cfg.DOWN_COLOR),
    ]
    for label, price, color in levels:
        main.axhline(price, color=color, linestyle="--", linewidth=1.0, alpha=0.75)
        main.text(x_end, price, f"{label} {price:.2f}", ha="right", va="bottom",
                  fontsize=9.5, color=color)

    # ---- 图例与坐标轴标签 ----
    main_handles = [
        Line2D([0], [0], color=cfg.MA_COLORS[w], lw=1.4, label=f"MA{w}")
        for w in cfg.MA_WINDOWS
    ]
    main_handles.append(Line2D([0], [0], color="#888888", lw=1.0, ls="--", label="支撑/压力位"))
    main.legend(handles=main_handles, loc="upper left", frameon=True, framealpha=0.92,
                edgecolor="#cccccc", fontsize=10)
    main.set_ylabel("价格（元）", fontsize=12)

    macd_ax.legend(
        handles=[
            Line2D([0], [0], color=cfg.MACD_COLORS["dif"], lw=1.0, label="DIF(12,26)"),
            Line2D([0], [0], color=cfg.MACD_COLORS["dea"], lw=1.0, label="DEA(9)"),
            Patch(facecolor=cfg.UP_COLOR, label="MACD 柱（红多绿空）"),
        ],
        loc="upper left", fontsize=9, frameon=True, framealpha=0.92, edgecolor="#cccccc",
    )
    macd_ax.set_ylabel("MACD", fontsize=11)

    rsi_ax.set_ylim(0, 100)
    rsi_ax.text(x_end, cfg.RSI_OVERBOUGHT + 3, f"超买 {cfg.RSI_OVERBOUGHT}",
                ha="right", fontsize=9, color="#c0392b")
    rsi_ax.text(x_end, cfg.RSI_OVERSOLD - 7, f"超卖 {cfg.RSI_OVERSOLD}",
                ha="right", fontsize=9, color="#1e8449")
    rsi_ax.set_ylabel(f"RSI({rsi_period})", fontsize=11)
    rsi_ax.set_xlabel("日期", fontsize=12)

    fig.subplots_adjust(hspace=0.38, top=0.93, bottom=0.07, left=0.065, right=0.945)
    fig.suptitle(title, fontsize=19, fontweight="bold", color="#222222")
    if out_path:
        fig.savefig(out_path, dpi=cfg.DPI, facecolor="white")
        plt.close(fig)
        return out_path
    return fig


def plot_backtest_chart(bt: dict, title: str, out_path):
    """绘制双均线策略 vs 基准（买入持有）净值对比图，附回撤子图。"""
    eq = bt["equity"]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )

    ax1.plot(eq.index, eq["strategy_equity"], color="#d64541", lw=1.8,
             label=f"双均线策略（MA{cfg.BT_FAST}/MA{cfg.BT_SLOW}）")
    ax1.plot(eq.index, eq["benchmark_equity"], color="#3498db", lw=1.8,
             label="买入持有基准")
    ax1.axhline(1.0, color="#999999", lw=0.8, linestyle="--")

    # 标记金叉买入 / 死叉卖出时点
    pos_diff = eq["position"].diff().fillna(eq["position"])
    buys = eq[(pos_diff != 0) & (eq["position"] == 1)]
    sells = eq[(pos_diff != 0) & (eq["position"] == 0)]
    ax1.scatter(buys.index, buys["strategy_equity"], marker="^", s=70,
                color=cfg.UP_COLOR, zorder=5, label="金叉买入")
    ax1.scatter(sells.index, sells["strategy_equity"], marker="v", s=70,
                color=cfg.DOWN_COLOR, zorder=5, label="死叉卖出")

    # 期末净值标注
    s_end = float(eq["strategy_equity"].iloc[-1])
    b_end = float(eq["benchmark_equity"].iloc[-1])
    ax1.text(eq.index[-1], s_end, f" 策略 {(s_end - 1) * 100:+.2f}%",
             va="center", color="#d64541", fontweight="bold", fontsize=11)
    ax1.text(eq.index[-1], b_end, f" 基准 {(b_end - 1) * 100:+.2f}%",
             va="center", color="#3498db", fontweight="bold", fontsize=11)

    # 绩效摘要框
    m = bt["metrics"]
    info = (f"策略总收益 {m['策略总收益%']:+.2f}%　基准 {m['基准总收益%']:+.2f}%　"
            f"年化 {m['策略年化收益%']:+.2f}%　最大回撤 {m['策略最大回撤%']:.2f}%　"
            f"夏普 {m['夏普比率']:.2f}　胜率 {m['胜率%']:.1f}%")
    ax1.text(0.99, 0.98, info, transform=ax1.transAxes, ha="right", va="top", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.88,
                       edgecolor="#cccccc"))

    ax2.fill_between(eq.index, eq["strategy_dd"] * 100, 0, color="#95a5a6", alpha=0.55,
                     label="策略回撤")
    ax2.set_ylabel("回撤（%）", fontsize=11)
    ax2.legend(loc="lower left", fontsize=9, frameon=True, edgecolor="#cccccc")

    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(True, color=cfg.GRID_COLOR, linestyle="--", linewidth=0.7, alpha=0.8)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(labelsize=10)
    ax1.set_ylabel("净值（期初 = 1）", fontsize=11)
    ax1.legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.92,
               edgecolor="#cccccc")

    fig.suptitle(title, fontsize=17, fontweight="bold", color="#222222")
    fig.subplots_adjust(top=0.93)
    fig.savefig(out_path, dpi=cfg.DPI, facecolor="white")
    plt.close(fig)
    return out_path
