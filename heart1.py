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
import matplotlib.font_manager as fm  # 新增字体管理模块

# ------------------------ 中文显示配置 ------------------------
# 显式指定中文字体路径（根据服务器环境调整路径）
font_path = '/usr/share/fonts/truetype/msttcorefonts/SimHei.ttf'  # Linux常见路径示例
try:
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = font_prop.get_name()
except:
    # 备选方案：使用系统已有中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ------------------------ Streamlit界面 ------------------------
st.title("基于机器学习的心脏病预测")

# 数据加载与展示
url = "https://raw.githubusercontent.com/LH-cjin/Heart-disease-prediction/main/heart.csv"
data = pd.read_csv(url)

st.write("### 数据示例（自适应宽度）")
st.dataframe(
    data,
    use_container_width=True,  # 关键调整：自适应容器宽度
    height=300,  # 显示固定高度
    hide_index=True  # 隐藏索引列
)

# 数据预处理与模型训练（保持原逻辑）
predictors = data.drop("target", axis=1)
target = data["target"]
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 侧边栏模型选择
st.sidebar.header('模型设置')
model_type = st.sidebar.selectbox('选择一个模型', ['逻辑回归', '朴素贝叶斯', '决策树', '随机森林', 'KNN'])

# 模型超参数设置（保持原逻辑）
# ...（原有超参数设置代码保持不变）

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

# 其余部分保持不变...
