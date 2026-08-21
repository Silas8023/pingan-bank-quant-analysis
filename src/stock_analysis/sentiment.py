# -*- coding: utf-8 -*-
"""新闻情绪分析模块：抓取平安银行新闻 + 金融情感词典打分，与股价对照。"""

from __future__ import annotations

import re
import json
import urllib.request

import akshare as ak
import numpy as np
import pandas as pd

POSITIVE = ["增长", "上涨", "新高", "突破", "盈利", "净利", "营收增长", "利好", "增持",
            "回购", "分红", "中标", "获批", "超预期", "改善", "复苏", "回升", "盈利改善"]
NEGATIVE = ["下滑", "下跌", "亏损", "风险", "利空", "减持", "处罚", "违规", "诉讼",
            "不良", "逾期", "低于预期", "恶化", "承压", "收缩", "爆雷", "违约"]
NEGATION = ["不", "未", "无", "否认", "终止", "取消"]


def score_text(text: str) -> float:
    """基于金融情感词典打分：正词 +1、负词 -1，否定词反转。"""
    if not text:
        return 0.0
    score = 0.0
    for word in POSITIVE:
        if word in text:
            score += text.count(word)
    for word in NEGATIVE:
        if word in text:
            score -= text.count(word)
    for word in NEGATION:
        if word in text:
            score = -score
            break
    return float(score)


def fetch_news(code: str = "000001") -> pd.DataFrame | None:
    """尝试多个新闻源，返回 date/title/content 三列的 DataFrame；全部失败返回 None。"""
    # 1) 东方财富（多数环境可用）
    try:
        raw = ak.stock_news_em(symbol=code)
        if raw is not None and not raw.empty:
            raw = raw.rename(columns={"发布时间": "date", "新闻标题": "title", "新闻内容": "content"})
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
            raw = raw.dropna(subset=["date"]).sort_values("date")
            if "content" not in raw.columns:
                raw["content"] = raw["title"]
            print(f"[情绪] 使用东方财富新闻源，共 {len(raw)} 条")
            return raw[["date", "title", "content"]]
    except Exception as exc:  # noqa: BLE001
        print(f"[情绪] 东方财富源失败（{exc}）")

    # 2) 同花顺个股新闻页（内嵌 JSON，含标题与时间戳）
    try:
        raw = _ths_news(code)
        if raw is not None and not raw.empty:
            print(f"[情绪] 使用同花顺新闻源，共 {len(raw)} 条")
            return raw
    except Exception as exc:  # noqa: BLE001
        print(f"[情绪] 同花顺源失败（{exc}）")

    # 3) 财新（仅全市场快讯，需过滤关键词）
    try:
        raw = ak.stock_news_main_cx()
        if raw is not None and not raw.empty:
            text = raw.apply(lambda r: str(r.values), axis=1)
            hit = raw[text.str.contains("平安银行", na=False)].copy()
            if not hit.empty:
                hit["date"] = pd.Timestamp.now().normalize()
                hit["title"] = hit["summary"]
                hit["content"] = hit["summary"]
                print(f"[情绪] 使用财新新闻源，共 {len(hit)} 条")
                return hit[["date", "title", "content"]]
    except Exception as exc:  # noqa: BLE001
        print(f"[情绪] 财新源失败（{exc}）")
    return None


def _ths_news(code: str) -> pd.DataFrame | None:
    """解析同花顺个股新闻页内嵌 JSON。"""
    url = f"http://basic.10jqka.com.cn/{code}/news.html"
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
                      "Referer": "http://basic.10jqka.com.cn/"})
    html = urllib.request.urlopen(req, timeout=25).read().decode("gbk", errors="ignore")
    items = []
    for m in re.finditer(r"\{[^{}]*?\"type\":\"news\"[^{}]*?\}", html):
        try:
            obj = json.loads(m.group(0))
            if "title" in obj and "ctime" in obj:
                items.append({
                    "date": pd.to_datetime(int(obj["ctime"]), unit="s"),
                    "title": obj["title"],
                    "content": obj["title"],
                })
        except Exception:  # noqa: BLE001
            continue
    if not items:
        return None
    return pd.DataFrame(items).sort_values("date").reset_index(drop=True)


def sentiment_series(df: pd.DataFrame, code="000001") -> dict:
    """按日聚合新闻情绪得分，与股价收益率对照（相关系数 + 简单图表数据）。"""
    news = fetch_news(code)
    if news is None or news.empty:
        return {"ok": False, "message": "无可用新闻源，跳过情绪分析"}
    news["score"] = news.apply(lambda r: score_text(f"{r['title']} {r['content']}"), axis=1)
    daily = news.groupby(news["date"].dt.date)["score"].agg(["mean", "count"]).rename(
        columns={"mean": "sentiment", "count": "news_count"})
    stock = df[["close"]].copy()
    stock["ret"] = stock["close"].pct_change()
    merged = stock.join(daily, how="inner")
    corr = float(merged["ret"].corr(merged["sentiment"])) if len(merged) > 2 else float("nan")
    return {
        "ok": True,
        "news_count": int(len(news)),
        "daily_sentiment": daily,
        "corr_with_return": round(corr, 4),
        "n_overlap_days": int(len(merged)),
        "message": f"新闻情绪与当日收益率相关系数 {corr:.3f}（样本 {len(merged)} 天）",
    }
