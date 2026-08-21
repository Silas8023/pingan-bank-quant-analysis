# -*- coding: utf-8 -*-
"""宏观数据模块：CPI / PMI / LPR 与平安银行股价收益的相关性分析。"""

from __future__ import annotations

import akshare as ak
import numpy as np
import pandas as pd
import re


def _to_monthly_series(df: pd.DataFrame, value_cols) -> pd.Series:
    """把 akshare 宏观表转成以月份为索引的 Series（优先同比列）。"""
    for col in df.columns:
        if "日期" in str(col) or "月份" in str(col) or "时间" in str(col) or "date" in str(col).lower():
            date_col = col
            break
    else:
        return None
    ser = df.set_index(date_col)
    # 兼容 "2026年07月份" / "2026-07" / datetime 三种日期格式
    parsed = []
    for v in ser.index:
        s = str(v)
        m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", s)
        if m:
            parsed.append(pd.Timestamp(f"{m.group(1)}-{int(m.group(2)):02d}-01"))
        else:
            parsed.append(pd.to_datetime(v, errors="coerce"))
    ser.index = parsed
    ser = ser[ser.index.notna()]
    for col in value_cols:
        if col in ser.columns:
            out = pd.to_numeric(ser[col], errors="coerce").dropna()
            out.index = pd.to_datetime(out.index).to_period("M").to_timestamp("M")
            return out.groupby(out.index).last()
    return None


def fetch_macro() -> dict:
    """获取 CPI / PMI / LPR 月度序列（尽力而为，失败项返回 None）。"""
    result = {}
    try:
        cpi = ak.macro_china_cpi()
        result["CPI同比%"] = _to_monthly_series(cpi, ["全国-同比增长", "同比", "全国-同比"])
    except Exception as exc:  # noqa: BLE001
        result["CPI同比%"] = None
    try:
        pmi = ak.macro_china_pmi()
        result["PMI"] = _to_monthly_series(pmi, ["制造业-指数", "指数", "制造业PMI"])
    except Exception as exc:  # noqa: BLE001
        result["PMI"] = None
    try:
        lpr = ak.macro_china_lpr()
        result["LPR1Y%"] = _to_monthly_series(lpr, ["LPR1Y", "1Y", "一年期"])
    except Exception as exc:  # noqa: BLE001
        result["LPR1Y%"] = None
    return result


def macro_correlation(df: pd.DataFrame, code=None, lookback_months=36) -> dict:
    """合并月末股价月收益与宏观序列，输出相关系数与简单 OLS 结果。

    若传入 code，将自动拉取更长的历史行情（默认 36 个月）以保证月度样本充足。
    """
    from statsmodels.api import add_constant, OLS

    if code is not None:
        from .data_fetcher import fetch_stock_data
        df = fetch_stock_data(code, lookback_months, "qfq")
    monthly_close = df["close"].resample("ME").last()
    stock_ret = monthly_close.pct_change().dropna() * 100
    table = pd.DataFrame({"平安银行月收益%": stock_ret})

    macro = fetch_macro()
    for name, ser in macro.items():
        if ser is not None:
            s = ser[ser.index <= stock_ret.index.max()]
            table[name] = s.reindex(stock_ret.index).values if len(s) else np.nan
    table = table.dropna(how="any")
    if len(table) < 8:
        return {"correlation": pd.DataFrame(), "ols": {}, "n": len(table)}

    corr = table.corr()
    ols_results = {}
    for col in [c for c in table.columns if c != "平安银行月收益%"]:
        try:
            X = add_constant(table[col])
            fit = OLS(table["平安银行月收益%"], X).fit()
            ols_results[col] = {
                "coef": round(float(fit.params[col]), 4),
                "p_value": round(float(fit.pvalues[col]), 4),
                "r2": round(float(fit.rsquared), 4),
                "n": int(fit.nobs),
            }
        except Exception as exc:  # noqa: BLE001
            ols_results[col] = {"error": str(exc)[:80]}
    return {"correlation": corr, "ols": ols_results, "n": len(table)}
