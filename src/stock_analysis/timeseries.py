# -*- coding: utf-8 -*-
"""时间序列建模模块：ARIMA 收益建模 + GARCH 波动率建模（风险补充）。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def arima_returns(df: pd.DataFrame, order=(1, 0, 1), forecast_days=5) -> dict:
    """对日收益率（%）建立 ARIMA 模型，输出拟合信息与未来 5 日预测。"""
    from statsmodels.tsa.arima.model import ARIMA

    ret = df["close"].pct_change().dropna() * 100
    model = ARIMA(ret, order=order).fit()
    forecast = model.get_forecast(steps=forecast_days)
    mean = np.asarray(forecast.predicted_mean)
    ci = forecast.conf_int(alpha=0.05)
    return {
        "model": model,
        "aic": float(model.aic),
        "bic": float(model.bic),
        "params": {k: round(float(v), 4) for k, v in model.params.items()},
        "forecast_mean_pct": [round(float(x), 3) for x in mean],
        "forecast_ci": [[round(float(x), 3) for x in row] for row in np.asarray(ci)],
    }


def garch_volatility(df: pd.DataFrame, p=1, q=1) -> dict:
    """用 GARCH(1,1) 建模收益率波动率，输出年化条件波动与 5 日预测。"""
    from arch import arch_model

    ret = df["close"].pct_change().dropna() * 100
    model = arch_model(ret, vol="GARCH", p=p, q=q, mean="Constant", dist="normal")
    res = model.fit(disp="off")
    cond_vol = np.sqrt(res.conditional_volatility)  # 日波动率(%)
    last_ann_vol = float(cond_vol.iloc[-1] * np.sqrt(252))
    fc = res.forecast(horizon=5, reindex=False)
    fc_vol = np.sqrt(fc.variance.values[-1]) * np.sqrt(252)
    return {
        "model": res,
        "aic": float(res.aic),
        "bic": float(res.bic),
        "last_annualized_vol_pct": round(last_ann_vol, 2),
        "forecast_annualized_vol_pct": [round(float(x), 2) for x in fc_vol],
        "params": {k: round(float(v), 4) for k, v in res.params.items()},
    }
