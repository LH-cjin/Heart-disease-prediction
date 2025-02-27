import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Streamlit 标题
st.title("基于机器学习的心脏病预测")

# 从 GitHub 读取 CSV 文件
url = "https://raw.githubusercontent.com/LH-cjin/Heart-disease-prediction/main/heart.csv"  # 替换为实际 GitHub 仓库地址
data = pd.read_csv(url)

# 显示数据的前几行
st.write("数据示例", data.head())

# 数据预处理：去除目标变量并分割数据
predictors = data.drop("target", axis=1)
target = data["target"]

# 分割数据集为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型选择
model_type = st.selectbox('选择一个模型', ['逻辑回归', '朴素贝叶斯', '决策树', '随机森林', 'KNN'])

# 训练并评估选择的模型
if model_type == '逻辑回归':
    model = LogisticRegression(random_state=0)
    model.fit(X_train_scaled, Y_train)
elif model_type == '朴素贝叶斯':
    model = GaussianNB()
    model.fit(X_train_scaled, Y_train)
elif model_type == '决策树':
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train_scaled, Y_train)
elif model_type == '随机森林':
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train_scaled, Y_train)
elif model_type == 'KNN':
    model = KNeighborsClassifier()
    model.fit(X_train_scaled, Y_train)

# 预测结果
y_pred = model.predict(X_test_scaled)

# 评估指标
accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = 2 * (precision * recall) / (precision + recall)

st.write(f"准确率: {accuracy * 100:.2f}%")
st.write(f"精确度: {precision * 100:.2f}%")
st.write(f"召回率: {recall * 100:.2f}%")
st.write(f"F1 分数: {f1 * 100:.2f}%")

# 混淆矩阵
cm = confusion_matrix(Y_test, y_pred)
st.write("混淆矩阵:")
st.write(cm)

# 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
st.pyplot(fig)

# （可选）下载混淆矩阵图
st.download_button(label="下载混淆矩阵", data=fig.savefig("/mnt/data/confusion_matrix.png"), file_name="confusion_matrix.png")
