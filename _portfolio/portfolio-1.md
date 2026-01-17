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
#### 2. 数据可视化
绘制直方图和箱线图来直观展示数据分布。
```python
colname = FEATURES
# 直方图绘制
fig, axs = plt.subplots(int(len(colname)/2), 2, constrained_layout=True, figsize=(8, 6), dpi=150)
for i in range(len(colname)):
    sns.histplot(x=colname[i], data=data_clean, alpha=0.4, kde=True, ax=axs[i//2, i%2])
plt.suptitle("直方图")

# 箱线图绘制
fig, axs = plt.subplots(int(len(colname) / 2), 2, constrained_layout=True, figsize=(8, 6), dpi=150)
for i in range(len(colname)):
    sns.boxplot(data=data_clean, x=TARGET, y=colname[i], ax=axs[i // 2, i % 2])
plt.suptitle("箱线图")
plt.show()
```

#### 3. 线性回归分析
使用线性回归模型分析特征与目标变量之间的关系。
```python
mod = smf.ols(formula='HOSPITAL_EXPIRE_FLAG ~ age_month + lab_5237_min +lab_5227_min + lab_5225_range + lab_5235_max + lab_5257_min', data=data_clean)
res = mod.fit()
print(res.summary())
```

#### 4. 预测模型评估
将数据划分为训练集和测试集，分别使用逻辑回归、随机森林和支持向量机进行建模和评估。

**逻辑回归**
```python
X = data_clean.drop(TARGET, axis=1)
y = data_clean[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_pred_lr)
print(f"逻辑回归 - 准确率: {accuracy_lr:.4f}, 召回率: {recall_lr:.4f}, AUC: {roc_auc_lr:.4f}")
```

**随机森林**
```python
rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)
print(f"随机森林 - 准确率: {accuracy_rf:.4f}, 召回率: {recall_rf:.4f}, AUC: {roc_auc_rf:.4f}")
```

**支持向量机**
```python
svc = SVC(kernel='linear', C=1.0, probability=False, random_state=42)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
accuracy_svc = accuracy_score(y_test, y_pred_svc)
recall_svc = recall_score(y_test, y_pred_svc)
roc_auc_svc = roc_auc_score(y_test, y_pred_svc)
print(f"支持向量机 - 准确率: {accuracy_svc:.4f}, 召回率: {recall_svc:.4f}, AUC: {roc_auc_svc:.4f}")
```

#### 5. 聚类分析
选择部分特征进行聚类分析。
```python
features = ['lab_5237_min', 'lab_5227_min', 'lab_5225_range', 'lab_5235_max', 'lab_5257_min']
data_clustering = data_clean[features]
```

### 分析结果 (Results & Analysis)
![直方图](/images/histograms.png)
直方图展示了各个特征的数据分布情况，通过观察可以了解数据的集中趋势和离散程度。

![箱线图](/images/boxplots.png)
箱线图反映了不同特征在目标变量不同类别下的分布差异，有助于发现数据中的异常值和特征之间的关系。

通过线性回归分析，我们可以得到各个特征对目标变量的影响程度。而在预测模型评估中，逻辑回归、随机森林和支持向量机各自给出了准确率、召回率和AUC等评估指标，这些指标可以帮助我们选择最适合的模型进行预测。聚类分析则可以将数据划分为不同的簇，有助于发现数据的内在结构。
```

