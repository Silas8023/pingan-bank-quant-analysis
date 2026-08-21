# 平安银行（000001.SZ）技术分析 · 金融科技作品集项目

基于 Python + akshare + mplfinance 的 A 股个股技术分析项目：自动获取近 6 个月日 K 行情，
输出 300 DPI 高清专业图表（K 线 + 成交量 + MACD + RSI 四联图、双均线回测对比图）、
量化统计指标与 Markdown 分析报告，并扩展机器学习预测、波动率建模、宏观相关性、
新闻情绪分析与交互式 Web 看板。

> **AI 辅助编码声明**：本项目由 AI 辅助完成编码实现；业务逻辑、参数设定与分析结论由本人设计并复核。

## 快速开始

```bash
pip install -r requirements.txt
python main.py                                  # 平安银行 000001，近 6 个月
python main.py --all                            # 全流程 + 增强模块（ML/时序/宏观/情绪/SQLite）
python main.py --code 600519                    # 任意 A 股代码
streamlit run app.py                            # 交互看板
```

## 增强功能（v2）

- **完整绩效指标**：年化收益、最大回撤、夏普、索提诺、卡玛、信息比率、年化波动率；
- **仓位管理**：无管理 / 固定比例 / 凯利公式 / 风险平价（目标波动率）四路对比；
- **真实交易成本**：佣金 + 印花税 + 滑点，以及 A 股涨跌停限制（涨停无法买入、跌停无法卖出）；
- **机器学习**：随机森林预测次日涨跌（准确率 / AUC / 特征重要性），与双均线信号对照；
- **时间序列**：ARIMA 收益建模 + GARCH(1,1) 波动率建模；
- **宏观数据**：CPI / PMI / LPR 与月收益的相关性与回归分析；
- **新闻情绪**：个股新闻情感词典打分，与收益率对照；
- **Web 看板**：Streamlit 支持调节均线周期、仓位、成本参数，内置 AI 自然语言问答。

## 输出产物

- `output/<名称>K线图.png`：16×10 英寸 / 300 DPI 技术分析四联图
- `output/<名称>双均线回测对比图.png`：策略收益 vs 基准收益对比图
- `output/<名称>技术分析报告.md`：统计指标 + 技术解读 + 回测明细
- `output/jupyter-notebook/<名称>K线技术分析.ipynb / .html`：可复现研究 Notebook（含全部代码与图表；HTML 版可直接在浏览器查阅）
- `output/<名称>投研报告.md / .html`：基本面投研报告（finance-analyst 工作流产出；HTML 版可直接在浏览器查阅）
- `app.py`：Streamlit 交互看板；`output/pingan.db`：SQLite 行情库（自动更新）

## 项目结构

```text
main.py                  主程序（编排全流程 + 增强模块开关）
app.py                   Streamlit 交互看板
stock_analysis/
  config.py              全局参数（指标周期、画布、配色）
  data_fetcher.py        数据获取与清洗（akshare 多源回退）
  indicators.py          MA / MACD / RSI 指标计算
  backtest.py            增强回测（绩效/仓位/成本/涨跌停）
  ml_forecast.py         机器学习预测（随机森林）
  timeseries.py          ARIMA / GARCH 建模
  macro.py               宏观相关性分析（CPI/PMI/LPR）
  sentiment.py           新闻情绪分析
  storage.py             SQLite 存储与自动更新
  stats.py               量化统计（收益/回撤/波动率/RSI 状态）
  charting.py            技术四联图 + 回测对比图
  report.py              Markdown 分析报告生成
```

## 方法说明

- 数据：akshare 依次尝试腾讯、东方财富、新浪三个免费源，自动回退；前复权日线。
- 指标：MA(5/10/20)、MACD(12,26,9)、RSI(14, Wilder 平滑)。
- 回测：MA5/MA20 金叉买入、死叉卖出，信号次日生效；成本含佣金/印花税/滑点，并模拟涨跌停。
- 统计：区间涨跌幅、最高/最低、最大回撤、年化波动率、平均成交量、期末 RSI 超买超卖判断。

## 配套 Codex Skill

本工作流已封装为 Codex Skill：`stock-analysis`。之后在 Codex 中直接输入股票代码
（如“分析 600519 贵州茅台”）即可自动生成整套分析。

## 作品集新增

- **可复现研究 Notebook**：`output/jupyter-notebook/平安银行K线技术分析.ipynb`，
  由官方 jupyter-notebook 技能模板生成，完整执行验证通过（0 报错、图表内嵌），
  修改 `CODE` 即可分析任意 A 股。
- **基本面投研报告**：`output/平安银行投研报告.md`，按 finance-analyst 工作流
  （结论摘要 / 关键驱动 / 数据校验 / 风险与反方观点 / 下一步尽调）产出，数据取自
  同花顺、新浪、腾讯免费公开接口。
- **增强分析**：机器学习预测、GARCH 波动率、宏观相关性、新闻情绪、SQLite 存储，
  详见 `stock_analysis/` 各模块与 Notebook 第 6~10 节。
