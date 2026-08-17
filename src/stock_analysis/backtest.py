# -*- coding: utf-8 -*-
"""双均线（金叉 / 死叉）策略回测模块。

策略规则（业务参数由人工设定，代码负责忠实实现）：
- 快线上穿慢线 → T 日收盘产生买入信号，T+1 日持有；
- 快线下穿慢线 → T 日收盘产生卖出信号，T+1 日空仓；
- 信号次日生效，避免“未来函数”；
- 每笔成交按单边成本费率扣费，贴近真实交易。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_dual_ma_backtest(df: pd.DataFrame, fast=5, slow=20, cost_rate=0.0005,
                         trading_days=252) -> dict:
    """运行双均线回测，返回净值序列、交易明细与绩效指标。"""
    work = df.copy()
    fast_col, slow_col = f"MA{fast}", f"MA{slow}"
    if fast_col not in work.columns or slow_col not in work.columns:
        raise ValueError(f"缺少均线列 {fast_col}/{slow_col}，请先调用 add_all 计算指标")

    # 信号：快线 > 慢线 持有，否则空仓；shift(1) 使仓位次日生效
    raw_signal = (work[fast_col] > work[slow_col]).astype(int)
    work["position"] = raw_signal.shift(1).fillna(0).astype(int)

    work["benchmark_ret"] = work["close"].pct_change().fillna(0.0)
    work["strat_ret"] = work["position"] * work["benchmark_ret"]
    # 仓位变化当天扣单边交易成本
    turnover = work["position"].diff().abs().fillna(work["position"])
    work["strat_ret"] -= turnover * cost_rate

    work["benchmark_equity"] = (1 + work["benchmark_ret"]).cumprod()
    work["strategy_equity"] = (1 + work["strat_ret"]).cumprod()
    work["strategy_dd"] = work["strategy_equity"] / work["strategy_equity"].cummax() - 1

    trades = _extract_trades(work)
    metrics = _compute_metrics(work, trades, trading_days)
    return {"equity": work, "trades": trades, "metrics": metrics}


def _extract_trades(work: pd.DataFrame) -> list:
    """从仓位序列还原买卖交易记录（一次完整买入 → 卖出为一笔）。"""
    trades, entry = [], None
    for date, row in work.iterrows():
        if int(row["position"]) == 1 and entry is None:
            entry = (date, float(row["close"]))
        elif int(row["position"]) == 0 and entry is not None:
            entry_date, entry_price = entry
            trades.append({
                "买入日期": entry_date.date(),
                "买入价": round(entry_price, 2),
                "卖出日期": date.date(),
                "卖出价": round(float(row["close"]), 2),
                "区间收益%": round((float(row["close"]) / entry_price - 1) * 100, 2),
                "备注": "",
            })
            entry = None
    # 期末仍持仓：按最后收盘价结算，便于统计完整
    if entry is not None:
        entry_date, entry_price = entry
        last = work.iloc[-1]
        trades.append({
            "买入日期": entry_date.date(),
            "买入价": round(entry_price, 2),
            "卖出日期": last.name.date(),
            "卖出价": round(float(last["close"]), 2),
            "区间收益%": round((float(last["close"]) / entry_price - 1) * 100, 2),
            "备注": "期末仍持仓，按最后收盘价结算",
        })
    return trades


def _compute_metrics(work: pd.DataFrame, trades: list, trading_days: int) -> dict:
    """计算回测绩效：总收益、年化、最大回撤、夏普、胜率等。"""
    eq, bm = work["strategy_equity"], work["benchmark_equity"]
    n = len(work)
    strat_total = float(eq.iloc[-1] - 1)
    bm_total = float(bm.iloc[-1] - 1)
    annualize = lambda x: (1 + x) ** (trading_days / max(n, 1)) - 1  # noqa: E731

    ret = work["strat_ret"]
    sharpe = float(ret.mean() / ret.std() * np.sqrt(trading_days)) if ret.std() > 0 else 0.0
    wins = [t for t in trades if t["区间收益%"] > 0]
    return {
        "策略总收益%": round(strat_total * 100, 2),
        "基准总收益%": round(bm_total * 100, 2),
        "超额收益%": round((strat_total - bm_total) * 100, 2),
        "策略年化收益%": round(annualize(strat_total) * 100, 2),
        "基准年化收益%": round(annualize(bm_total) * 100, 2),
        "策略最大回撤%": round(float(work["strategy_dd"].min()) * 100, 2),
        "基准最大回撤%": round(float((bm / bm.cummax() - 1).min()) * 100, 2),
        "夏普比率": round(sharpe, 2),
        "交易次数": len(trades),
        "胜率%": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
    }
