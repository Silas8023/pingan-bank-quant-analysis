# -*- coding: utf-8 -*-
"""机器学习预测模块：随机森林 / XGBoost 预测个股次日涨跌。

输出：测试集准确率、AUC、混淆矩阵、特征重要性，并与双均线信号对照。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["ret_1", "ret_2", "ret_3", "ret_5", "vol_chg", "range",
            "ma_gap", "macd_hist", "rsi"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造特征与标签：用当日信息预测次日涨跌。"""
    out = df.copy()
    close = out["close"]
    for lag in (1, 2, 3, 5):
        out[f"ret_{lag}"] = close.pct_change(lag)
    out["vol_chg"] = out["volume"].pct_change()
    out["range"] = (out["high"] - out["low"]) / close
    out["ma_gap"] = close / out["MA20"] - 1
    out["macd_hist"] = out["MACD"]
    out["rsi"] = out["RSI14"]
    out["target"] = (close.shift(-1) > close).astype(int)  # 次日上涨=1
    return out


def _get_model(model_type: str, seed: int):
    if model_type == "xgb":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=200, max_depth=3,
                                 learning_rate=0.05, random_state=seed,
                                 eval_metric="logloss")
        except ImportError:
            print("[ML] xgboost 未安装，回退随机森林")
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=200, max_depth=4,
                                  random_state=seed, class_weight="balanced")


def predict_direction(df: pd.DataFrame, model_type="rf",
                      test_ratio=0.3, seed=42) -> dict:
    """训练模型并评估，返回指标与对照结果。"""
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

    data = build_features(df).dropna()
    if len(data) < 40:
        raise ValueError(f"样本不足（{len(data)}），无法训练")
    X, y = data[FEATURES], data["target"].astype(int)
    split = int(len(data) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = _get_model(model_type, seed)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else float("nan")
    cm = confusion_matrix(y_test, pred)

    # 双均线对照：同窗口内 MA5>MA20 信号对次日方向的命中率
    test_data = data.iloc[split:]
    ma_sig = (test_data["MA5"] > test_data["MA20"]).astype(int)
    ma_acc = float((ma_sig == test_data["target"]).mean())

    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return {
        "model_type": "XGBoost" if model_type == "xgb" and "XGB" in type(model).__name__
        else "RandomForest",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": float(acc),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "ma_benchmark_accuracy": ma_acc,
        "feature_importance": importances.head(10).to_dict(),
        "model": model,
    }
