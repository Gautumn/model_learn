

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# part2: Basic Data Exploration
# https://www.kaggle.com/code/dansbecker/basic-data-exploration/tutorial
melbourne_file_path = "./melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.describe())
''' 
打印如下，std:标准差
              Rooms         Price  ...    Longtitude  Propertycount
count  13580.000000  1.358000e+04  ...  13580.000000   13580.000000
mean       2.937997  1.075684e+06  ...    144.995216    7454.417378
std        0.955748  6.393107e+05  ...      0.103916    4378.581772
min        1.000000  8.500000e+04  ...    144.431810     249.000000
25%        2.000000  6.500000e+05  ...    144.929600    4380.000000
50%        3.000000  9.030000e+05  ...    145.000100    6555.000000
75%        3.000000  1.330000e+06  ...    145.058305   10331.000000
max       10.000000  9.000000e+06  ...    145.526350   21650.000000

[8 rows x 13 columns]
'''


# part3: Your First Machine Learning Model
# https://www.kaggle.com/code/dansbecker/your-first-machine-learning-model/tutorial

# 选择预测目标
filtered_melbourne_data = melbourne_data.dropna(axis=0)
y = filtered_melbourne_data.Price

# 选择多个特征
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]
print(X.describe())
'''
              Rooms      Bathroom       Landsize     Lattitude    Longtitude
count  13580.000000  13580.000000   13580.000000  13580.000000  13580.000000
mean       2.937997      1.534242     558.416127    -37.809203    144.995216
std        0.955748      0.691712    3990.669241      0.079260      0.103916
min        1.000000      0.000000       0.000000    -38.182550    144.431810
25%        2.000000      1.000000     177.000000    -37.856822    144.929600
50%        3.000000      1.000000     440.000000    -37.802355    145.000100
75%        3.000000      2.000000     651.000000    -37.756400    145.058305
max       10.000000      8.000000  433014.000000    -37.408530    145.526350

'''

print(X.head())
'''
   Rooms  Bathroom  Landsize  Lattitude  Longtitude
0      2       1.0     202.0   -37.7996    144.9984
1      2       1.0     156.0   -37.8079    144.9934
2      3       2.0     134.0   -37.8093    144.9944
3      3       2.0      94.0   -37.7969    144.9969
4      4       1.0     120.0   -37.8072    144.9941
'''

# 定义模型: 决策树模型，将features和target变量进行拟合
melbourne_model = DecisionTreeRegressor(random_state = 1)

# 捕获特征数据
melbourne_model.fit(X, y)

# 需要预测市场上的新房，而不是我们已经有价格的房屋。但是，我们将对训练数据的前几行进行预测，以了解 predict 函数的工作原理。
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
'''
Making predictions for the following 5 houses:
   Rooms  Bathroom  Landsize  Lattitude  Longtitude
0      2       1.0     202.0   -37.7996    144.9984
1      2       1.0     156.0   -37.8079    144.9934
2      3       2.0     134.0   -37.8093    144.9944
3      3       2.0      94.0   -37.7969    144.9969
4      4       1.0     120.0   -37.8072    144.9941
The predictions are
[1480000. 1035000. 1465000.  850000. 1600000.]
'''

