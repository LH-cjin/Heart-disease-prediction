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

# 设置 Streamlit 页面配置
st.set_page_config(page_title="心脏病预测", page_icon="❤️", layout="wide")

# Streamlit 标题
st.title("💖 基于机器学习的心脏病预测")

# 从 GitHub 读取 CSV 数据
url = "https://raw.githubusercontent.com/LH-cjin/Heart-disease-prediction/main/heart.csv"
data = pd.read_csv(url)

# 显示数据的前 5 行
st.subheader("📊 数据示例（前 5 行）")
st.write(data.head())

# 分割数据集
predictors = data.drop("target", axis=1)
target = data["target"]
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🎯 **侧边栏：模型选择和超参数**
st.sidebar.header('🛠 选择模型和超参数')
model_type = st.sidebar.selectbox('📌 选择一个模型', ['逻辑回归', '朴素贝叶斯', '决策树', '随机森林', 'KNN'])

# **超参数设置**
max_depth = st.sidebar.slider('🌳 决策树最大深度', 1, 20, 5) if model_type == '决策树' else None
n_estimators = st.sidebar.slider('🌲 随机森林树数', 10, 200, 100) if model_type == '随机森林' else None
n_neighbors = st.sidebar.slider('👥 KNN 邻居数', 1, 20, 5) if model_type == 'KNN' else None

# **训练模型**
if model_type == '逻辑回归':
    model = LogisticRegression(random_state=0)
elif model_type == '朴素贝叶斯':
    model = GaussianNB()
elif model_type == '决策树':
    model = DecisionTreeClassifier(random_state=0, max_depth=max_depth)
elif model_type == '随机森林':
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
elif model_type == 'KNN':
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

# 训练模型
model.fit(X_train_scaled, Y_train)
y_pred = model.predict(X_test_scaled)

# **计算评估指标**
accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = 2 * (precision * recall) / (precision + recall)

# **显示评估指标**
st.subheader("📈 评估指标")
metrics_data = pd.DataFrame({
    '指标': ['准确率', '精确度', '召回率', 'F1 分数'],
    '值': [f"{accuracy * 100:.2f}%", f"{precision * 100:.2f}%", f"{recall * 100:.2f}%", f"{f1 * 100:.2f}%"]
})
st.table(metrics_data)  # **修正：使用 st.table() 显示表格**

# 📊 **并排显示：混淆矩阵 & ROC 曲线**
col1, col2 = st.columns(2)

# **混淆矩阵**
with col1:
    st.subheader("📌 混淆矩阵")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title("Confusion Matrix", fontsize=14, color='black')
    st.pyplot(fig)

# **ROC 曲线 & AUC**
with col2:
    st.subheader("📈 ROC 曲线 & AUC")
    if hasattr(model, "predict_proba"):  # 确保模型支持 `predict_proba()`
        fpr, tpr, _ = roc_curve(Y_test, model.predict_proba(X_test_scaled)[:, 1])
        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('FPR')
        ax_roc.set_ylabel('TPR')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
    else:
        st.write("❌ 该模型不支持概率预测，无法绘制 ROC 曲线")

# 🌟 **特征重要性（仅适用于树模型）**
if model_type in ['决策树', '随机森林']:
    st.subheader("🌲 特征重要性")
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({'特征': predictors.columns, '重要性': feature_importances})
    importance_df = importance_df.sort_values(by='重要性', ascending=False)

    # **显示特征重要性表格**
    st.write(importance_df)

    # **绘制特征重要性柱状图**
    fig_feat, ax_feat = plt.subplots(figsize=(6, 4))
    sns.barplot(x='重要性', y='特征', data=importance_df, ax=ax_feat)
    ax_feat.set_title('Feature Importance')
    st.pyplot(fig_feat)

# 🎯 **实时预测：用户输入数据**
st.subheader("🔍 进行实时心脏病预测")


# **用户输入特征**
user_input = {}

# 1️⃣ **限定 age 必须为 0-120 的整数**
user_input['age'] = st.number_input("年龄 (age)", min_value=0, max_value=120, value=40, step=1, format="%d")

# 2️⃣ **限定 sex 只能输入 0（女性）或 1（男性）**
user_input['sex'] = st.selectbox("性别 (sex)", options=[0, 1], format_func=lambda x: "👩 女性" if x == 0 else "👨 男性")

# 3️⃣ **限定 cp（胸痛类型），提供描述**
cp_options = {
    0: "典型心绞痛",
    1: "非典型心绞痛",
    2: "非心绞痛",
    3: "无症状"
}
user_input['cp'] = st.selectbox("胸痛类型 (cp)", options=list(cp_options.keys()), format_func=lambda x: f"{x}: {cp_options[x]}")

# 4️⃣ **限定 trestbps（静息血压），范围 94-200 mm Hg**
user_input['trestbps'] = st.number_input("静息血压 (trestbps) (mm Hg)", min_value=94, max_value=200, value=120)

# **继续输入其他变量（示例）**
user_input['chol'] = st.number_input("胆固醇 (chol) (mg/dl)", min_value=126, max_value=564, value=200)
user_input['thalach'] = st.number_input("最大心率 (thalach)", min_value=70, max_value=210, value=150)

# **进行预测**
if st.button("🚀 预测心脏病风险"):
    # 预处理输入数据
    user_input_df = pd.DataFrame([user_input])  # 转换为 DataFrame
    user_input_scaled = scaler.transform(user_input_df)  # 标准化
    prediction = model.predict(user_input_scaled)[0]  # 预测类别
    probability = model.predict_proba(user_input_scaled)[0][1] if hasattr(model, "predict_proba") else None

    # **显示预测结果**
    if prediction == 1:
        st.error(f"⚠️ 该患者可能有心脏病，风险概率: {probability:.2f}" if probability else "⚠️ 该患者可能有心脏病")
    else:
        st.success(f"✅ 该患者心脏健康，风险概率: {probability:.2f}" if probability else "✅ 该患者心脏健康")



