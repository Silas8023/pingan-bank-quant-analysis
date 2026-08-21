# -*- coding: utf-8 -*-
"""平安银行量化分析 Web 看板（Streamlit）。

运行：
    streamlit run app.py

功能：行情与技术指标、增强回测（绩效/仓位/成本）、ML 预测、宏观相关性、
新闻情绪、AI 自然语言问答（可配置 OPENAI_API_KEY 调用 GPT，否则使用本地规则问答）。
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import streamlit as st

from stock_analysis import config as cfg
from stock_analysis.backtest import compare_sizing, run_dual_ma_backtest
from stock_analysis.charting import plot_technical_chart, setup_chinese_font
from stock_analysis.data_fetcher import fetch_stock_data, resolve_symbol
from stock_analysis.indicators import add_all
from stock_analysis.macro import macro_correlation
from stock_analysis.ml_forecast import predict_direction
from stock_analysis.sentiment import sentiment_series
from stock_analysis.stats import compute_stats, stats_table
from stock_analysis.timeseries import arima_returns, garch_volatility

st.set_page_config(page_title="平安银行量化分析看板", layout="wide")
setup_chinese_font()


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(code: str, months: int):
    df_all = fetch_stock_data(code, months, cfg.ADJUST)
    df_all = add_all(df_all, cfg.MA_WINDOWS,
                     (cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL), cfg.RSI_PERIOD)
    window = pd.Timestamp.today().normalize() - pd.DateOffset(months=months)
    return df_all.loc[df_all.index >= window].copy()


def rule_answer(question: str, df: pd.DataFrame, bt: dict) -> str:
    """本地规则问答：无需 API Key 也能回答常见问题。"""
    m = bt["metrics"]
    stats = compute_stats(df, cfg.RSI_PERIOD, cfg.TRADING_DAYS)
    q = question.lower()
    if any(k in q for k in ("涨跌", "收益", "表现")):
        return (f"近 {len(df)} 个交易日区间涨跌幅 {stats['period_return_pct']:+.2f}%，"
                f"双均线策略收益 {m['策略总收益%']:+.2f}% vs 基准 {m['基准总收益%']:+.2f}%。")
    if any(k in q for k in ("回撤", "风险")):
        return (f"策略最大回撤 {m['策略最大回撤%']:.2f}%，年化波动率 {m['年化波动率%']:.2f}%，"
                f"卡玛比率 {m['卡玛比率']:.2f}。")
    if any(k in q for k in ("rsi", "超买", "超卖")):
        rsi = stats["end_rsi"]
        return f"期末 RSI(14) = {rsi:.1f}，{'超买' if rsi > 70 else '超卖' if rsi < 30 else '中性'}。"
    if any(k in q for k in ("macd", "趋势")):
        return stats["macd_status"]
    if any(k in q for k in ("夏普", "sharpe", "绩效")):
        return (f"夏普 {m['夏普比率']:.2f}、索提诺 {m['索提诺比率']:.2f}、"
                f"信息比率 {m['信息比率']:.2f}、胜率 {m['胜率%']:.1f}%。")
    if any(k in q for k in ("预测", "明天", "未来", "机器学习")):
        return "可在「ML 预测」页签查看模型准确率与 AUC；本地问答不做投资预测。"
    return ("我可以回答：区间涨跌/收益、回撤/风险、RSI 超买超卖、MACD 趋势、"
            "夏普等绩效指标。配置 OPENAI_API_KEY 后可调用 GPT 生成完整分析。")


def chat_section(df: pd.DataFrame, bt: dict) -> None:
    st.subheader("AI 助手（自然语言查询）")
    question = st.text_input("输入问题，例如：近6个月涨跌幅多少？风险指标如何？")
    if question:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if key:
            try:
                import openai
                client = openai.OpenAI(api_key=key)
                resp = client.chat.completions.create(
                    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "system",
                               "content": "你是金融分析助手，用简洁中文回答平安银行相关问题。"},
                              {"role": "user", "content": question}],
                )
                st.write(resp.choices[0].message.content)
                return
            except Exception as exc:  # noqa: BLE001
                st.info(f"GPT 调用失败（{exc}），使用本地规则回答：")
        st.write(rule_answer(question, df, bt))


def main() -> None:
    st.title("平安银行（000001.SZ）量化分析看板")
    st.caption("数据：akshare 多源回退 · 指标：MA/MACD/RSI · 回测：双均线增强版（绩效/仓位/成本/涨跌停）")

    with st.sidebar:
        st.header("参数设置")
        code = st.text_input("股票代码", "000001")
        months = st.slider("分析月数", 3, 24, 6)
        fast = st.slider("快线 MA", 3, 20, 5)
        slow = st.slider("慢线 MA", 10, 60, 20)
        sizing = st.selectbox("仓位管理", ["none", "fixed", "kelly", "risk_parity"],
                              format_func=lambda x: {"none": "无管理(全仓)", "fixed": "固定比例80%",
                                                     "kelly": "凯利公式", "risk_parity": "风险平价(目标波动)"}[x])
        slip = st.slider("滑点(bp)", 0, 20, 1)
        limit = st.checkbox("涨跌停限制", True)

    if fast >= slow:
        st.error("快线周期必须小于慢线周期")
        return

    with st.spinner("加载行情并计算指标…"):
        try:
            df = load_data(code, months)
        except Exception as exc:  # noqa: BLE001
            st.error(f"数据获取失败：{exc}")
            return

    bt = run_dual_ma_backtest(df, fast=fast, slow=slow, sizing=sizing,
                              slippage_bps=float(slip), limit_enforce=limit)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["行情与技术指标", "回测与绩效", "仓位管理对比", "ML 预测", "宏观 / 情绪 / AI"])

    with tab1:
        st.dataframe(stats_table(compute_stats(df, cfg.RSI_PERIOD, cfg.TRADING_DAYS)))
        fig, _ = plot_technical_chart(df, f"{code} 近{months}个月技术分析", None)
        st.pyplot(fig)

    with tab2:
        metrics = bt["metrics"]
        cols = st.columns(4)
        for col, key in zip(cols, ["策略年化收益%", "策略最大回撤%", "夏普比率", "索提诺比率"]):
            col.metric(key, metrics[key])
        cols = st.columns(4)
        for col, key in zip(cols, ["卡玛比率", "信息比率", "年化波动率%", "胜率%"]):
            col.metric(key, metrics[key])
        st.dataframe(pd.DataFrame([{"指标": k, "数值": v} for k, v in metrics.items()]))
        if bt["trades"]:
            st.dataframe(pd.DataFrame(bt["trades"]))

    with tab3:
        st.dataframe(compare_sizing(df, fast=fast, slow=slow,
                                    slippage_bps=float(slip), limit_enforce=limit))
        st.caption("风险平价仓位 = 目标年化波动(15%) / 滚动20日年化波动，截断 [0,1]")

    with tab4:
        if st.button("训练模型（随机森林预测次日涨跌）"):
            with st.spinner("训练中…"):
                try:
                    res = predict_direction(df)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("准确率", f"{res['accuracy']:.1%}")
                    c2.metric("AUC", f"{res['auc']:.3f}")
                    c3.metric("双均线对照命中率", f"{res['ma_benchmark_accuracy']:.1%}")
                    st.write("特征重要性 Top10：")
                    imp = pd.Series(res["feature_importance"]).sort_values(ascending=True)
                    st.bar_chart(imp)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"训练失败：{exc}")

    with tab5:
        if st.button("宏观相关性（CPI/PMI/LPR vs 月收益）"):
            with st.spinner("获取宏观数据…"):
                try:
                    res = macro_correlation(df, code=code)
                    if not res["correlation"].empty:
                        st.dataframe(res["correlation"])
                        st.json(res["ols"])
                    else:
                        st.warning("宏观数据样本不足或获取失败")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"宏观分析失败：{exc}")
        if st.button("新闻情绪分析"):
            with st.spinner("抓取新闻…"):
                try:
                    res = sentiment_series(df, code)
                    if res["ok"]:
                        st.success(res["message"])
                        st.dataframe(res["daily_sentiment"].tail(20))
                    else:
                        st.warning(res["message"])
                except Exception as exc:  # noqa: BLE001
                    st.error(f"情绪分析失败：{exc}")
        chat_section(df, bt)


if __name__ == "__main__":
    main()
