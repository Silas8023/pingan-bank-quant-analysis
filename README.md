# 平安银行个股量化分析系统

**项目简介**：用 Python + AI Agent 做的全流程个股分析

## 技术栈

Python | Pandas | NumPy | SQL | mplfinance | Codex

## 项目流程

1. 数据获取（akshare）
2. 数据清洗（Pandas）
3. 指标计算（MA/MACD/RSI）
4. 可视化（多面板K线图）
5. AI智能分析（Codex finance-analyst）

## 成果展示

![平安银行K线四联图](figures/平安银行K线图.png)

![平安银行双均线回测对比图](figures/平安银行双均线回测对比图.png)

## AI工具分工

- ChatGPT：代码生成与调试
- Codex：投研分析 + Notebook标准化
- 人工：需求设计、业务逻辑、结果校验；把项目文件传上去

## 运行方式

```bash
pip install -r requirements.txt
python src/main.py
```
