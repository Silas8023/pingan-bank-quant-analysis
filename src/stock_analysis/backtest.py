# -*- coding: utf-8 -*-
"""双均线策略回测模块（增强版）。

相对 v1 的升级：
1. 完整绩效指标：年化收益、最大回撤、夏普、索提诺、卡玛、信息比率、波动率；
2. 仓位管理：无管理 / 固定比例 / 凯利公式 / 风险平价（目标波动率）；
3. 真实交易成本：佣金 + 印花税（卖出）+ 滑点；
4. A 股涨跌停限制：涨停无法买入、跌停无法卖出。

策略规则（业务逻辑由人工设定）：
- 快线上穿慢线 → T 日收盘产生信号，T+1 生效；
- 信号次日生效，避免未来函数；仓位与成本模拟贴近真实交易。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
LIMIT_PCT = 0.10  # A 股主板涨跌停幅度


# ---------- 仓位管理 ----------
def _kelly_fraction(work: pd.DataFrame) -> float:
    """根据历史交易盈亏计算凯利仓位（p - q/b），限制在 [0, 1]。"""
    trades = _extract_trades(work)
    if len(trades) < 3:
        return 0.5
    wins = [t["区间收益%"] for t in trades if t["区间收益%"] > 0]
    losses = [t["区间收益%"] for t in trades if t["区间收益%"] <= 0]
    if not wins or not losses:
        return 0.5
    p = len(wins) / len(trades)
    b = abs(np.mean(wins)) / abs(np.mean(losses))
    f = p - (1 - p) / b if b > 0 else 0.0
    return float(np.clip(f, 0.0, 1.0))


def _position_with_sizing(signal: pd.Series, sizing: str, sizing_param,
                          work: pd.DataFrame, target_vol: float) -> pd.Series:
    """把 0/1 信号按仓位管理方法转换为实际仓位。"""
    if sizing == "fixed":
        return signal * float(sizing_param or 0.8)
    if sizing == "kelly":
        temp = work.copy()
        temp["position"] = signal
        f = _kelly_fraction(temp)
        return signal * f
    if sizing == "risk_parity":
        # 目标波动率缩放：仓位 = 目标年化波动 / 滚动 20 日年化波动，截断 [0,1]
        ret = work["close"].pct_change()
        realized_vol = ret.rolling(20).std() * np.sqrt(TRADING_DAYS)
        pos = (target_vol / realized_vol.replace(0, np.nan)).fillna(1.0)
        pos = pos.clip(upper=1.0).fillna(1.0)
        return signal * pos
    return signal.astype(float)  # none


def _limit_days(work: pd.DataFrame) -> tuple:
    """计算每个交易日是否涨停 / 跌停（基于前收盘 ±10%，保留两位小数）。"""
    prev_close = work["close"].shift(1)
    limit_up = (prev_close * (1 + LIMIT_PCT)).round(2)
    limit_down = (prev_close * (1 - LIMIT_PCT)).round(2)
    is_limit_up = (work["close"] >= limit_up - 0.005) & prev_close.notna()
    is_limit_down = (work["close"] <= limit_down + 0.005) & prev_close.notna()
    return is_limit_up, is_limit_down


def run_dual_ma_backtest(df: pd.DataFrame, fast=5, slow=20, cost_rate=0.0005,
                         trading_days=252, sizing="none", sizing_param=None,
                         commission_rate=0.00025, stamp_tax_rate=0.0005,
                         slippage_bps=1.0, limit_enforce=True,
                         target_vol=0.15) -> dict:
    """运行增强版双均线回测，返回净值序列、交易明细与完整绩效指标。

    参数：
    - sizing: none / fixed / kelly / risk_parity；
    - commission_rate: 单边佣金费率（默认 0.025%）；
    - stamp_tax_rate: 卖出印花税率（默认 0.05%）；
    - slippage_bps: 滑点（基点，默认 1bp）；
    - limit_enforce: 是否执行涨跌停限制。
    """
    work = df.copy()
    fast_col, slow_col = f"MA{fast}", f"MA{slow}"
    if fast_col not in work.columns or slow_col not in work.columns:
        raise ValueError(f"缺少均线列 {fast_col}/{slow_col}，请先计算指标")

    raw_signal = (work[fast_col] > work[slow_col]).astype(int)
    raw_signal = raw_signal.shift(1).fillna(0)  # 信号次日生效
    position = _position_with_sizing(raw_signal, sizing, sizing_param, work, target_vol)

    if limit_enforce:
        is_limit_up, is_limit_down = _limit_days(work)
        prev_pos = position.shift(1).fillna(0.0)
        # 涨停日无法买入（0→正），跌停日无法卖出（正→0）
        block_buy = is_limit_up & (prev_pos <= 0) & (position > 0)
        block_sell = is_limit_down & (prev_pos > 0) & (position <= 0)
        position = position.mask(block_buy, prev_pos).mask(block_sell, prev_pos)

    work["position"] = position
    work["benchmark_ret"] = work["close"].pct_change().fillna(0.0)
    work["strat_ret"] = work["position"].shift(1) * work["benchmark_ret"]
    work["strat_ret"] = work["strat_ret"].fillna(0.0)

    # ---- 交易成本：佣金 + 印花税 + 滑点 ----
    turnover = work["position"].diff().abs().fillna(work["position"])
    sell_turnover = (-work["position"].diff()).clip(lower=0).fillna(0.0)
    cost = (
        turnover * commission_rate
        + sell_turnover * stamp_tax_rate
        + turnover * slippage_bps / 10000.0
    )
    work["strat_ret"] -= cost

    work["benchmark_equity"] = (1 + work["benchmark_ret"]).cumprod()
    work["strategy_equity"] = (1 + work["strat_ret"]).cumprod()
    work["strategy_dd"] = work["strategy_equity"] / work["strategy_equity"].cummax() - 1

    trades = _extract_trades(work)
    metrics = compute_performance_metrics(
        work["strat_ret"], work["benchmark_ret"],
        work["strategy_equity"], work["benchmark_equity"],
        trading_days=trading_days, trades=trades,
    )
    return {"equity": work, "trades": trades, "metrics": metrics,
            "sizing": sizing, "cost_params": {
                "commission_rate": commission_rate,
                "stamp_tax_rate": stamp_tax_rate,
                "slippage_bps": slippage_bps,
                "limit_enforce": limit_enforce,
            }}


def compute_performance_metrics(strat_ret, bench_ret, strat_eq, bench_eq,
                                trading_days=252, trades=None) -> dict:
    """完整绩效指标：收益 / 回撤 / 夏普 / 索提诺 / 卡玛 / 信息比率等。"""
    strat_ret = pd.Series(strat_ret).fillna(0.0)
    bench_ret = pd.Series(bench_ret).fillna(0.0)
    n = len(strat_ret)
    strat_total = float(strat_eq.iloc[-1] - 1)
    bench_total = float(bench_eq.iloc[-1] - 1)
    annualize = lambda x: (1 + x) ** (trading_days / max(n, 1)) - 1  # noqa: E731

    dd = strat_eq / strat_eq.cummax() - 1
    max_dd = float(dd.min())
    ann_ret = annualize(strat_total)

    vol = float(strat_ret.std(ddof=1) * np.sqrt(trading_days)) if n > 1 else 0.0
    sharpe = float(strat_ret.mean() / strat_ret.std(ddof=1) * np.sqrt(trading_days)) \
        if strat_ret.std(ddof=1) > 0 else 0.0
    downside = strat_ret[strat_ret < 0]
    downside_dev = float(downside.std(ddof=1) * np.sqrt(trading_days)) \
        if len(downside) > 1 and downside.std(ddof=1) > 0 else 0.0
    sortino = float(ann_ret / downside_dev) if downside_dev > 0 else 0.0
    calmar = float(ann_ret / abs(max_dd)) if max_dd < 0 else 0.0

    excess = strat_ret - bench_ret
    ir = float(excess.mean() / excess.std(ddof=1) * np.sqrt(trading_days)) \
        if len(excess) > 1 and excess.std(ddof=1) > 0 else 0.0
    bench_dd = float((bench_eq / bench_eq.cummax() - 1).min())

    trades = trades or []
    wins = [t for t in trades if t["区间收益%"] > 0]
    return {
        "策略总收益%": round(strat_total * 100, 2),
        "基准总收益%": round(bench_total * 100, 2),
        "超额收益%": round((strat_total - bench_total) * 100, 2),
        "策略年化收益%": round(ann_ret * 100, 2),
        "基准年化收益%": round(annualize(bench_total) * 100, 2),
        "策略最大回撤%": round(max_dd * 100, 2),
        "基准最大回撤%": round(bench_dd * 100, 2),
        "年化波动率%": round(vol * 100, 2),
        "夏普比率": round(sharpe, 2),
        "索提诺比率": round(sortino, 2),
        "卡玛比率": round(calmar, 2),
        "信息比率": round(ir, 2),
        "交易次数": len(trades),
        "胜率%": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
    }


def _extract_trades(work: pd.DataFrame) -> list:
    """从仓位序列还原交易记录（以有效持仓 > 0 为入场）。"""
    trades, entry = [], None
    for date, row in work.iterrows():
        pos = float(row["position"])
        if pos > 0 and entry is None:
            entry = (date, float(row["close"]))
        elif pos <= 0 and entry is not None:
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


def compare_sizing(df: pd.DataFrame, fast=5, slow=20, **kwargs) -> pd.DataFrame:
    """对比不同仓位管理方法的核心绩效，返回汇总表。"""
    rows = []
    for sizing in ("none", "fixed", "kelly", "risk_parity"):
        bt = run_dual_ma_backtest(df, fast=fast, slow=slow,
                                  sizing=sizing, sizing_param=0.8, **kwargs)
        m = bt["metrics"]
        rows.append({
            "仓位管理": {"none": "无管理(全仓)", "fixed": "固定比例80%",
                         "kelly": "凯利公式", "risk_parity": "风险平价(目标波动)"}[sizing],
            "总收益%": m["策略总收益%"],
            "年化收益%": m["策略年化收益%"],
            "最大回撤%": m["策略最大回撤%"],
            "夏普": m["夏普比率"],
            "索提诺": m["索提诺比率"],
            "卡玛": m["卡玛比率"],
            "交易次数": m["交易次数"],
        })
    return pd.DataFrame(rows)
