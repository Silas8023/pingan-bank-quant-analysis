#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A 股个股技术分析主程序（金融科技作品集项目）。

用法示例：
    python main.py                                  # 默认：平安银行 000001，近 6 个月
    python main.py --code 600519                    # 任意 A 股代码（自动识别市场）
    python main.py --code 000001 --name 平安银行 --months 6 --fast 5 --slow 20
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from stock_analysis import config as cfg
from stock_analysis.backtest import compare_sizing, run_dual_ma_backtest
from stock_analysis.charting import plot_backtest_chart, plot_technical_chart, setup_chinese_font
from stock_analysis.data_fetcher import fetch_stock_data, fetch_stock_name, resolve_symbol
from stock_analysis.indicators import add_all
from stock_analysis.macro import macro_correlation
from stock_analysis.ml_forecast import predict_direction
from stock_analysis.report import write_report
from stock_analysis.sentiment import sentiment_series
from stock_analysis.stats import compute_stats, stats_table
from stock_analysis.storage import StockDB
from stock_analysis.timeseries import arima_returns, garch_volatility


def parse_args():
    parser = argparse.ArgumentParser(description="A 股个股技术分析（金融科技作品集项目）")
    parser.add_argument("--code", default=cfg.DEFAULT_CODE, help="6 位 A 股代码，默认 000001")
    parser.add_argument("--name", default=None, help="股票名称（可选，缺省自动获取）")
    parser.add_argument("--months", type=int, default=cfg.DEFAULT_MONTHS, help="分析区间（自然月）")
    parser.add_argument("--fast", type=int, default=cfg.BT_FAST, help="双均线快线周期")
    parser.add_argument("--slow", type=int, default=cfg.BT_SLOW, help="双均线慢线周期")
    parser.add_argument("--out-dir", type=Path, default=cfg.OUTPUT_DIR, help="输出目录")
    parser.add_argument("--ml", action="store_true", help="运行机器学习预测（次日涨跌）")
    parser.add_argument("--ts", action="store_true", help="运行 ARIMA/GARCH 时间序列建模")
    parser.add_argument("--macro", action="store_true", help="宏观数据相关性分析")
    parser.add_argument("--sentiment", action="store_true", help="新闻情绪分析")
    parser.add_argument("--db", action="store_true", help="SQLite 入库与自动更新")
    parser.add_argument("--all", action="store_true", help="运行全部增强模块")
    return parser.parse_args()


def main():
    # 兼容部分 Windows 控制台的 UTF-8 输出
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    args = parse_args()
    symbol = resolve_symbol(args.code)
    market = symbol[:2].upper()
    name = args.name or fetch_stock_name(args.code) or args.code

    print(f"===== {name}（{args.code}.{market}）近 {args.months} 个月技术分析 =====")

    # 1. 获取数据（含指标预热期）
    df_all = fetch_stock_data(args.code, args.months, cfg.ADJUST)

    # 2. 计算技术指标（在预热数据上计算，保证窗口起点指标有效）
    df_all = add_all(df_all, cfg.MA_WINDOWS,
                     (cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL), cfg.RSI_PERIOD)

    # 3. 按自然月裁剪分析窗口
    window_start = pd.Timestamp.today().normalize() - pd.DateOffset(months=args.months)
    df = df_all.loc[df_all.index >= window_start].copy()
    if len(df) < 30:
        raise RuntimeError(f"分析区间内有效交易日不足（{len(df)} 天），请检查代码或网络")

    # 4. 量化统计
    stats = compute_stats(df, cfg.RSI_PERIOD, cfg.TRADING_DAYS)
    print("\n[量化统计指标]")
    print(stats_table(stats).to_string(index=False))

    # 5. 双均线回测
    bt = run_dual_ma_backtest(df, args.fast, args.slow, cfg.BT_COST_RATE, cfg.TRADING_DAYS)
    print("\n[双均线回测指标]")
    print(pd.DataFrame([{"指标": k, "数值": v} for k, v in bt["metrics"].items()])
          .to_string(index=False))
    if bt["trades"]:
        print("\n[交易明细]")
        print(pd.DataFrame(bt["trades"]).to_string(index=False))
    else:
        print("\n[交易明细] 区间内没有完整交易")

    # ---- 增强模块 ----
    if args.all or (args.ml or args.ts or args.macro or args.sentiment or args.db):
        print("\n===== 增强分析模块 =====")

    if args.all:
        print("\n[仓位管理对比]")
        print(compare_sizing(df, args.fast, args.slow).to_string(index=False))

    if args.all or args.ml:
        print("\n[机器学习预测 · 次日涨跌]")
        try:
            ml = predict_direction(df)
            print(f"模型：{ml['model_type']} | 训练 {ml['train_size']} / 测试 {ml['test_size']}")
            print(f"准确率 {ml['accuracy']:.2%} | AUC {ml['auc']:.3f} | 双均线对照命中率 {ml['ma_benchmark_accuracy']:.2%}")
        except Exception as exc:  # noqa: BLE001
            print(f"ML 预测失败：{exc}")

    if args.all or args.ts:
        print("\n[ARIMA 收益建模]")
        try:
            ar = arima_returns(df)
            print(f"AIC {ar['aic']:.1f} | BIC {ar['bic']:.1f} | 未来5日收益均值(%) {ar['forecast_mean_pct']}")
        except Exception as exc:  # noqa: BLE001
            print(f"ARIMA 失败：{exc}")
        print("[GARCH 波动率建模]")
        try:
            gc = garch_volatility(df)
            print(f"当前年化波动 {gc['last_annualized_vol_pct']}% | 未来5日年化波动(%) {gc['forecast_annualized_vol_pct']}")
        except Exception as exc:  # noqa: BLE001
            print(f"GARCH 失败：{exc}")

    if args.all or args.macro:
        print("\n[宏观相关性 · CPI/PMI/LPR vs 平安银行月收益]")
        try:
            mc = macro_correlation(df, code=args.code)
            if not mc["correlation"].empty:
                print(mc["correlation"].to_string())
                for macro_name, ols in mc["ols"].items():
                    print(f"{macro_name}: 系数 {ols.get('coef')} | p值 {ols.get('p_value')} | R² {ols.get('r2')}")
            else:
                print("宏观数据样本不足或获取失败")
        except Exception as exc:  # noqa: BLE001
            print(f"宏观分析失败：{exc}")

    if args.all or args.sentiment:
        print("\n[新闻情绪分析]")
        try:
            se = sentiment_series(df, args.code)
            print(se.get("message", se.get("message", "无数据")))
        except Exception as exc:  # noqa: BLE001
            print(f"情绪分析失败：{exc}")

    if args.all or args.db:
        print("\n[SQLite 存储与自动更新]")
        try:
            db = StockDB(args.out_dir / "pingan.db")
            info = db.auto_update(args.code, args.months, cfg.ADJUST)
            print(f"入库 {info['inserted_or_updated']} 行，库内共 {info['total_rows']} 行 -> {info['db_path']}")
            db.close()
        except Exception as exc:  # noqa: BLE001
            print(f"SQLite 失败：{exc}")

    # 6. 绘图与报告
    font_family = setup_chinese_font()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    title = f"{name}({args.code}.{market})近{args.months}个月技术分析｜金融科技作品集项目"
    kline_path = args.out_dir / f"{name}K线图.png"
    bt_path = args.out_dir / f"{name}双均线回测对比图.png"
    report_path = args.out_dir / f"{name}技术分析报告.md"

    plot_technical_chart(df, title, kline_path, font_family)
    plot_backtest_chart(
        bt,
        f"{name}({args.code}.{market})双均线策略回测｜策略收益 vs 基准收益",
        bt_path,
    )
    write_report(stats, bt, report_path, [kline_path.name, bt_path.name],
                 args.code, name, args.months, cfg.ADJUST, market)

    print(f"\n已生成：\n  {kline_path}\n  {bt_path}\n  {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001  顶层异常兜底，输出友好提示
        print(f"运行失败：{exc}", file=sys.stderr)
        sys.exit(1)
