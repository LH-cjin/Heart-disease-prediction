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
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

# Streamlit 标题
st.title("基于机器学习的心脏病预测")

# 从 GitHub 读取 CSV 文件
url = "https://raw.githubusercontent.com/LH-cjin/Heart-disease-prediction/main/heart.csv"
data = pd.read_csv(url)

# 显示数据的前几行，调整表格样式
st.write("数据示例")

# 使用 st.dataframe 并调整大小来放大表格，适配所有15列，避免横向滚动
st.dataframe(data, width=1500, height=500)  # 更大的宽度以适配15列，确保不滚动

# 数据预处理：去除目标变量并分割数据
predictors = data.drop("target", axis=1)
target = data["target"]

# 分割数据集为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sidebar: 模型选择和超参数设置
st.sidebar.header('模型设置')
model_type = st.sidebar.selectbox('选择一个模型', ['逻辑回归', '朴素贝叶斯', '决策树', '随机森林', 'KNN'])

# 超参数配置（针对部分模型）
max_depth = st.sidebar.slider('决策树最大深度', min_value=1, max_value=20, value=5) if model_type == '决策树' else None
n_estimators = st.sidebar.slider('随机森林树数', min_value=10, max_value=200, value=100) if model_type == '随机森林' else None
n_neighbors = st.sidebar.slider('KNN 邻居数', min_value=1, max_value=20, value=5) if model_type == 'KNN' else None

# 训练并评估选择的模型
if model_type == '逻辑回归':
    model = LogisticRegression(random_state=0)
    model.fit(X_train_scaled, Y_train)
elif model_type == '朴素贝叶斯':
    model = GaussianNB()
    model.fit(X_train_scaled, Y_train)
elif model_type == '决策树':
    model = DecisionTreeClassifier(random_state=0, max_depth=max_depth)
    model.fit(X_train_scaled, Y_train)
elif model_type == '随机森林':
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    model.fit(X_train_scaled, Y_train)
elif model_type == 'KNN':
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, Y_train)

# 预测结果
y_pred = model.predict(X_test_scaled)

# 计算评估指标
accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = 2 * (precision * recall) / (precision + recall)

# ------------------------ 评估指标美化 ------------------------
# 创建指标表格
metrics_data = {
    '指标': ['准确率', '精确度', '召回率', 'F1 分数'],
    '值': [f"{accuracy * 100:.2f}%", f"{precision * 100:.2f}%", f"{recall * 100:.2f}%", f"{f1 * 100:.2f}%"]
}
metrics_df = pd.DataFrame(metrics_data)

# 使用Styler设置精美表格
styled_metrics = metrics_df.style \
    .set_table_styles([{
        'selector': 'th',
        'props': [('background-color', '#40466e'), 
                ('color', 'white'),
                ('font-size', '18px')]
    }, {
        'selector': 'td',
        'props': [('font-size', '16px')]
    }]) \
    .highlight_max(color='#FFD700', axis=0) \
    .set_properties(**{
        'text-align': 'center',
        'width': '200px',
        'background-color': '#F8F9FA'
    })

st.write("### 模型评估指标")
st.table(styled_metrics)

# ------------------------ 混淆矩阵显示优化 ------------------------
cm = confusion_matrix(Y_test, y_pred)

# 通过 st.columns 实现并排显示图表
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    
    # 添加双语标题
    try:
        ax.set_title("混淆矩阵 (Confusion Matrix)", 
                    fontsize=16, 
                    weight='bold',
                    color='black')
    except:
        ax.set_title("Confusion Matrix",
                    fontsize=16,
                    weight='bold',
                    color='black')
    
    st.pyplot(fig)

# ------------------------ ROC 曲线和 AUC ------------------------
fpr, tpr, _ = roc_curve(Y_test, model.predict_proba(X_test_scaled)[:, 1])
roc_auc = auc(fpr, tpr)

with col2:
    fig_roc, ax_roc = plt.subplots(figsize=(5, 3))  # 缩小 ROC 图
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label='ROC 曲线 (AUC = %0.2f)' % roc_auc)
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('假阳性率 (False Positive Rate)')
    ax_roc.set_ylabel('真阳性率 (True Positive Rate)')
    ax_roc.set_title('接收操作特征曲线 (ROC Curve)')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

# 特征重要性展示（仅对于树模型有效）
if model_type in ['决策树', '随机森林']:
    feature_importances = model.feature_importances_
    features = predictors.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    st.write("特征重要性:")
    st.write(importance_df)

    # 绘制特征重要性图
    fig_feat, ax_feat = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax_feat)
    ax_feat.set_title('特征重要性')
    st.pyplot(fig_feat)
