# -*- coding: utf-8 -*-
"""数据获取与清洗模块。

设计：
1. 通过 akshare 依次尝试腾讯、东方财富、新浪三个免费数据源（自动回退）；
2. 统一列名与类型，剔除缺失值 / 停牌零成交行，按日期去重排序；
3. 返回带“指标预热期”的完整日线数据，分析窗口由调用方按自然月裁剪。
"""

from __future__ import annotations

import datetime as dt
import urllib.request

import akshare as ak
import pandas as pd

# 不同数据源列名到统一列名的映射
_COL_ALIASES = {
    "date": ["date", "日期"],
    "open": ["open", "开盘"],
    "high": ["high", "最高"],
    "low": ["low", "最低"],
    "close": ["close", "收盘"],
    "volume": ["volume", "vol", "成交量"],
    "amount": ["amount", "成交额"],
}


def resolve_symbol(code: str) -> str:
    """把 6 位数字代码规范成带市场前缀形式：000001 -> sz000001、600519 -> sh600519。"""
    c = code.strip().lower()
    if len(c) == 8 and c[:2] in ("sh", "sz", "bj"):
        return c
    if not (len(c) == 6 and c.isdigit()):
        raise ValueError(f"无法识别的股票代码：{code}（请输入 6 位数字或带 sh/sz/bj 前缀）")
    if c.startswith(("60", "68", "90")):
        return "sh" + c
    if c.startswith(("00", "20", "30")):
        return "sz" + c
    if c.startswith(("43", "83", "87", "92")):
        return "bj" + c
    return "sh" + c  # 兜底


def fetch_stock_name(code: str):
    """从腾讯实时行情接口取股票名称；失败返回 None（不阻塞主流程）。"""
    symbol = resolve_symbol(code)
    url = f"https://qt.gtimg.cn/q={symbol}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode("gbk", errors="ignore")
        parts = text.split("~")
        if len(parts) > 2 and parts[1]:
            return parts[1]
    except Exception:  # noqa: BLE001  名称仅用于展示，失败不影响分析
        pass
    return None


def fetch_stock_data(code="000001", months=6, adjust="qfq", warmup_days=45):
    """获取近 N 个月日 K 数据（含指标预热期），返回清洗后的 DataFrame。"""
    symbol = resolve_symbol(code)
    end = dt.date.today()
    # 预热期多取约 45 个自然日，保证 MA20 等指标在分析起点即有有效值
    start = end - dt.timedelta(days=int(months * 30.4) + warmup_days)
    start_str, end_str = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

    sources = (
        ("腾讯", lambda: ak.stock_zh_a_hist_tx(
            symbol=symbol, start_date=start_str, end_date=end_str, adjust=adjust)),
        ("东方财富", lambda: ak.stock_zh_a_hist(
            symbol=code, period="daily", start_date=start_str, end_date=end_str, adjust=adjust)),
        ("新浪", lambda: ak.stock_zh_a_daily(
            symbol=symbol, start_date=start_str, end_date=end_str, adjust=adjust)),
    )
    errors = []
    for label, func in sources:
        try:
            raw = func()
            if raw is None or raw.empty:
                raise ValueError("返回数据为空")
            cleaned = _clean(raw)
            print(f"[数据源] 使用 {label} 行情，共 {len(cleaned)} 行（含预热期）")
            return cleaned
        except Exception as exc:  # noqa: BLE001  逐个数据源尝试，全部失败才报错
            errors.append(f"{label}: {exc}")
            print(f"[数据源] {label} 获取失败（{exc}），尝试备用源…")
    raise RuntimeError("所有行情数据源均失败：" + "；".join(errors))


def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    """统一列名与类型，并清洗：排序、去重、剔除无效价格与零成交行。"""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for target, aliases in _COL_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename[alias] = target
                break
    df = df.rename(columns=rename)

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"数据缺少必要列：{missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # 缺失值处理：日期/价格任一为空即剔除
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    before = len(df)
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]  # 剔除无效价格
    df = df[df["volume"] > 0]                                        # 剔除停牌零成交行
    dropped = before - len(df)
    if dropped:
        print(f"[数据清洗] 剔除 {dropped} 行无效 / 零成交数据")
    if df.empty:
        raise ValueError("清洗后没有有效数据")

    df = df.set_index("date").sort_index()
    keep = [c for c in ("open", "high", "low", "close", "volume", "amount") if c in df.columns]
    return df[keep]
