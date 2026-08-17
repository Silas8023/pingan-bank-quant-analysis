# -*- coding: utf-8 -*-
"""Markdown 分析报告生成：数据说明 + 统计指标 + 技术解读 + 回测 + 图表。"""

from __future__ import annotations

from pathlib import Path

from .stats import stats_table

# 报告内展示的核心回测代码片段（完整实现见 backtest.py）
_CODE_SNIPPET = """```python
# 双均线策略核心逻辑：T 日收盘出信号，T+1 日生效，避免未来函数
raw_signal = (df["MA5"] > df["MA20"]).astype(int)
df["position"] = raw_signal.shift(1).fillna(0).astype(int)
df["strat_ret"] = df["position"] * df["close"].pct_change().fillna(0)
df["strategy_equity"] = (1 + df["strat_ret"]).cumprod()
```"""


def _md_table(columns: list, rows: list) -> str:
    """把二维数据渲染成 Markdown 表格。"""
    lines = ["| " + " | ".join(str(c) for c in columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def write_report(stats: dict, bt: dict, out_path, image_names: list, code: str,
                 name: str, months: int, adjust: str, market: str) -> Path:
    """生成完整分析报告并写盘（UTF-8）。"""
    s = stats
    stat_rows = [(r["指标"], r["数值"], r["说明"]) for _, r in stats_table(stats).iterrows()]
    metric_rows = list(bt["metrics"].items())
    trade_rows = [
        (t["买入日期"], t["买入价"], t["卖出日期"], t["卖出价"], t["区间收益%"], t["备注"])
        for t in bt["trades"]
    ]
    if not trade_rows:
        trade_rows = [("—", "—", "—", "—", "—", "区间内无完整交易")]

    kline_img, bt_img = image_names[0], image_names[1]
    lines = [
        f"# {name}（{code}.{market}）近 {months} 个月技术分析报告",
        "",
        "> **AI 辅助编码声明**：本项目由 AI 辅助完成编码实现；业务逻辑、参数设定与分析结论由本人设计并复核。",
        "",
        "## 1. 项目概览",
        "",
        f"- 数据源：akshare（腾讯 → 东方财富 → 新浪自动回退），前复权日线",
        f"- 分析区间：{s['start']} ~ {s['end']}，共 {s['trading_days']} 个交易日",
        "- 输出产物：技术分析四联图、双均线回测对比图、本报告",
        "",
        "## 2. 量化统计指标",
        "",
        _md_table(["指标", "数值", "说明"], stat_rows),
        "",
        "## 3. 技术面解读",
        "",
        f"- **趋势 / 均线**：{s['ma_status']}；期末收盘价 {s['last_close']:.2f} 元。",
        f"- **动能 / MACD**：{s['macd_status']}。",
        f"- **情绪 / RSI**：期末 RSI(14) = {s['end_rsi']:.1f}，{s['rsi_signal']}。",
        f"- **关键点位**：区间最高 {s['high']:.2f} 元（压力位）、最低 {s['low']:.2f} 元（支撑位），"
        "斐波那契 0.382 / 0.5 / 0.618 关键位详见图中虚线标注。",
        f"- **风险画像**：区间最大回撤 {s['max_drawdown_pct']:.2f}%，年化波动率 {s['annual_vol_pct']:.2f}%。",
        "",
        "## 4. 双均线策略回测",
        "",
        "策略规则（业务逻辑由人工设计）：快线 MA5 上穿慢线 MA20 时次日买入持有，"
        "下穿时次日卖出空仓；信号 T 日收盘产生、T+1 日生效，避免未来函数；"
        "每笔成交按单边 0.05% 成本扣费。基准为区间内买入持有。",
        "",
        _md_table(["指标", "数值"], metric_rows),
        "",
        "### 交易明细",
        "",
        _md_table(["买入日期", "买入价", "卖出日期", "卖出价", "区间收益%", "备注"], trade_rows),
        "",
        "## 5. 关键实现逻辑（代码节选）",
        "",
        _CODE_SNIPPET,
        "",
        "完整代码采用模块化拆分：数据获取 / 指标计算 / 回测 / 统计 / 绘图 / 报告，"
        "均带详细中文注释，便于复现与评审。",
        "",
        "## 6. 图表",
        "",
        f"![技术分析图]({kline_img})",
        "",
        f"![双均线回测对比图]({bt_img})",
        "",
    ]
    out_path = Path(out_path)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
