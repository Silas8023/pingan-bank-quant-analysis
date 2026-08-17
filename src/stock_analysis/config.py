# -*- coding: utf-8 -*-
"""全局配置：股票参数、技术指标参数、回测参数、图表样式与输出路径。

业务逻辑与参数由人工设计，此处集中管理便于复现与调参。
"""

from pathlib import Path

# ---- 股票与时间 ----
DEFAULT_CODE = "000001"        # 平安银行
DEFAULT_NAME = "平安银行"
DEFAULT_MONTHS = 6             # 分析区间（自然月）
ADJUST = "qfq"                 # 前复权，消除除权除息造成的价格跳空

# ---- 技术指标参数（经典默认值） ----
MA_WINDOWS = (5, 10, 20)
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
RSI_PERIOD = 14
RSI_OVERBOUGHT, RSI_OVERSOLD = 70, 30

# ---- 双均线回测参数 ----
BT_FAST, BT_SLOW = 5, 20       # 快线 / 慢线周期
BT_COST_RATE = 0.0005          # 单边交易成本（佣金 + 滑点近似）
TRADING_DAYS = 252             # 年化用交易日数

# ---- 图表样式（专业商务风，A 股红涨绿跌） ----
FIG_SIZE = (16, 10)            # 画布 16 × 10 英寸
DPI = 300                      # 高清输出 300 DPI
UP_COLOR = "#e74c3c"           # 上涨红
DOWN_COLOR = "#27ae60"         # 下跌绿
MA_COLORS = {5: "#f39c12", 10: "#3498db", 20: "#9b59b6"}
MACD_COLORS = {"dif": "#e91e63", "dea": "#00bcd4"}
RSI_COLOR = "#7d3cff"
GRID_COLOR = "#e8e8e8"

# ---- 输出 ----
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
