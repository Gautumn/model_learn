# https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?select=diabetes.csv
# 根据患者数据来预测患者是否患有糖尿病

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('./diabetes.csv')
print("Initial shape:", df.shape)
print(df.info())
print(df.head())
print(df.describe())


######################################################
######### 二：数据预处理：给模型准备"干净"的数据
######################################################
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols] = df[cols].replace(0, np.nan)
df.fillna(df.median(), inplace=True)
print("\nMissing values after cleaning:\n", df.isnull().sum())
'''
🔍 在做什么？ 我们把生理指标中的0值（比如血糖值为0，这在医学上不可能）替换为"空值"（NaN），再用中位数填充这些空值。
❓ 为什么这样做？
- 想象医生看病时，如果病人的检查报告写着"血糖：0"，医生会认为这是数据错误，而不是真的没血糖
- 中位数比平均数更适合处理这种情况（比如少数肥胖患者会拉高BMI的平均值）
- 如果不处理这些异常值，模型会学到错误的规律（比如"血糖为0的人都健康"）
'''


df['BMI_Age'] = df['BMI'] * df['Age']
df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin']+1)
print("\nNew features:\n", df[['BMI_Age', 'Glucose_Insulin_Ratio']].head())
'''
🔍 在做什么？ 我们创建了两个新指标：

- BMI×年龄（可能反映长期肥胖对身体的影响）
- 血糖/胰岛素比值（反映胰岛素抵抗程度）
❓ 为什么这样做？ 就像医生不仅看单项指标，还会综合分析多项指标的关系。比如同样的BMI值，年轻人和老年人的健康风险不同。这些新特征能帮助模型发现更复杂的规律。
'''

######################################################
######### 三：模型数据处理
######################################################
X = df.drop('Outcome', axis=1)  # 所有特征（症状）
y = df['Outcome']                      # 目标变量（是否患病）
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining size:", X_train.shape)
print("Test size:", X_test.shape)
'''
🔍 在做什么？ 我们把数据分成两部分：
- 训练集（80%数据）：给模型学习用（相当于教材）
- 测试集（20%数据）：检验模型学得怎么样（相当于考试）

❓ 为什么这样做？
- 如果用全部数据训练，模型可能会"死记硬背"答案（就像学生背题通过考试，但没真正学会）
- stratify=y 确保训练集和测试集中患病比例相同（避免模型只学了大部分健康人数据）
- random_state=42 保证每次运行结果相同（方便调试）
'''

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
'''
🔍 在做什么？ 把所有特征缩放到同一尺度（比如年龄范围是21-81，而BMI是18-67）。
❓ 为什么这样做？ 想象称体重时，一个用公斤，一个用斤，数值大小会误导模型。标准化后，所有特征对模型的影响更加公平。
'''

######################################################
######### 四：选择算法：决策树如何“思考”
######################################################
model = DecisionTreeClassifier(
    max_depth=4,                # 树的最大深度（避免想太多）
    min_samples_split=20,       # 至少20个样本才分裂（避免钻牛角尖）
    class_weight='balanced',    # 平衡不同类别的重要性
    random_state=42
)
model.fit(X_train_scaled, y_train)  # 训练模型
'''
在做什么？ 我们用决策树算法训练模型，就像教计算机画一棵"判断树"：
如果血糖 > 120 → 检查BMI
  如果BMI > 30 → 很可能患病
  否则 → 检查年龄
...

❓ 为什么这样做？
- 决策树最像人类的判断过程，容易理解
- max_depth=4 防止模型过度复杂（就像医生不会问你100个问题才诊断）
- class_weight='balanced' 解决数据不平衡问题（比如90%人健康，模型不能简单全预测为健康）
'''


######################################################
######### 五：模型评估：如何判断模型"看病准不准"
######################################################
from sklearn.metrics import (accuracy_score, classification_report,
                           confusion_matrix, roc_auc_score)

y_pred = model.predict(X_test_scaled)               # 让模型预测测试集
print("Accuracy:", accuracy_score(y_test, y_pred))  # 准确率
print("ROC AUC:", roc_auc_score(y_test, y_pred))    # ROC曲线下面积
print("\nClassification Report:\n", classification_report(y_test, y_pred))
'''
🔍 在做什么？ 我们用4个指标给模型"打分"：
- 准确率（Accuracy） ：模型预测正确的比例（类似考试得分率）
- ROC AUC ：衡量模型区分患者和健康人的能力（值越接近1越好）
- 分类报告 ：详细展示每个类别的精确率（查准率）和召回率（查全率）
❓ 为什么这样做？
- 只看准确率会骗人！比如90%的人健康，模型全猜"健康"也能得90分，但对真正的患者毫无帮助
- 召回率（Recall） 对糖尿病预测特别重要：漏诊（把患者判为健康）比误诊（把健康人判为患者）后果更严重
- ROC AUC能综合评价模型的稳健性，不受数据不平衡影响
'''



print("\nConfusion Matrix:")
print(pd.crosstab(y_test, y_pred,
      rownames=['Actual'],
      colnames=['Predicted']))
'''
🔍 在做什么？ 用表格展示模型的4种预测结果：
          Predicted
Actual    0    1
0        98   12  # 健康人被正确预测/误诊
1        23   31  # 患者被漏诊/正确预测

❓ 为什么这样做？
- 比数字更直观：一眼看出模型更容易犯哪种错误（比如漏诊23人，误诊12人）
- 帮助医生判断模型实用性：如果漏诊太多，可能需要调整模型
'''


######################################################
######### 六：特征重要性：模型更看重哪些指标？
######################################################
import matplotlib.pyplot as plt
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(10,6))
importances.plot(kind='bar')
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
'''
🔍 在做什么？ 画柱状图展示每个特征对预测的贡献度，比如：
- 血糖（Glucose）可能占30%
- BMI可能占20%
- 年龄可能占15%

❓ 为什么这样做？
- 像医生总结"哪些症状最能判断糖尿病"，帮我们理解模型逻辑
- 发现无用特征：如果某个特征重要性接近0，下次可以不用它（简化模型）
- 医学意义：验证模型学到的规律是否符合医学常识（比如血糖确实是关键指标）
'''

######################################################
######### 七：超参数调优：给模型"调参"
######################################################
from sklearn.model_selection import GridSearchCV
params = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [10, 20, 30]
}

grid = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
grid.fit(X_train_scaled, y_train)
print("\nBest parameters:", grid.best_params_)  # 输出最佳参数
'''
🔍 在做什么？ 自动尝试不同参数组合，找出效果最好的设置，比如：
- max_depth=5 （树深5层）
- min_samples_split=20 （至少20个样本才分支）

❓ 为什么这样做？
- 模型像相机，需要调参数才能拍出好照片：
  - 树太深（ max_depth=10 ）→ 模型太复杂，会记住训练数据的噪音（过拟合）
  - 树太浅（ max_depth=2 ）→ 模型太简单，学不会规律（欠拟合）
- GridSearchCV帮我们自动"试错"，比人工调参更高效
'''

######################################################
######### 八、决策树可视化：让模型"思考过程"看得见
######################################################
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(model,
          feature_names=X.columns,
          class_names=['No Diabetes', 'Diabetes'],
          filled=True,
          rounded=True)
plt.show()
'''
🔍 在做什么？ 画出决策树的结构图，展示模型如何一步步做判断：
如果 Glucose ≤ 127.5 → 检查 BMI
  如果 BMI ≤ 30.4 → 预测：No Diabetes
  如果 BMI > 30.4 → 预测：Diabetes
如果 Glucose > 127.5 → 检查 Age
...

❓ 为什么这样做？
- 把抽象的"AI黑箱"变成看得见的流程图，适合初学者理解
- 医学场景中可解释性很重要：医生需要知道模型为什么做出这个判断
- 发现不合理的决策路径：如果模型用"皮肤厚度"作为最重要指标，可能需要检查数据问题
'''

######################################################
######### 总结
######################################################
'''
1. 明确问题 → 2. 准备数据 → 3. 清洗数据 → 4. 创造特征 → 5. 拆分数据
   ↓          ↓          ↓          ↓          ↓
   预测糖尿病  读取CSV文件  处理0值/缺失值  BMI×年龄等新特征  训练集(80%)+测试集(20%)

6. 选择模型 → 7. 训练模型 → 8. 评估模型 → 9. 优化模型 → 10. 解释模型
   ↓          ↓          ↓          ↓          ↓
   决策树算法  喂数据给模型  准确率/ROC等指标  GridSearchCV调参  特征重要性+树可视化
'''

