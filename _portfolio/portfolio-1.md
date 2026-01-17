---
---
title: "医疗数据分析与建模项目"
collection: portfolio
type: "Data Analysis"
permalink: /portfolio/medical-data-analysis
date: 2026-01-17
excerpt: "本项目对医疗数据进行了全面分析，包括数据清洗、可视化、线性回归分析，并用逻辑回归、随机森林、支持向量机进行预测建模，还开展了聚类分析。"
header:
  teaser: /images/histograms.png
tags:
  - 医疗数据
  - 数据分析
  - 机器学习
  - 预测模型
  - 聚类分析
tech_stack:
  - name: Python
  - name: Scikit-learn
  - name: Statsmodels
---

### 项目背景 (Background)
本项目旨在对医疗数据进行深入分析和建模。通过数据预处理、统计分析、预测模型评估和聚类分析等步骤，以挖掘数据中的潜在信息，为医疗决策提供支持。

### 核心实现 (Implementation)
#### 1. 数据预处理
首先读取数据，使用中位数填充缺失值，并定义常用特征和目标变量。
```python
data_raw = pd.read_excel("data.xlsx")  # 原始数据
data_clean = data_raw.fillna(data_raw.median())  # 清洗后的数据
FEATURES = ['age_month', 'lab_5237_min', 'lab_5227_min', 
            'lab_5225_range', 'lab_5235_max', 'lab_5257_min']
TARGET = 'HOSPITAL_EXPIRE_FLAG'
---


