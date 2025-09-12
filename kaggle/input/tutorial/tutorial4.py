
# part4: Model Validation 模型验证，也就是预测的准确性
# https://www.kaggle.com/code/dansbecker/model-validation
'''
许多人在测量预测准确性时犯了一个巨大的错误。他们使用训练数据进行预测，并将这些预测与训练数据中的目标值进行比较。您稍后将看到此方法的问题以及如何解决它，但让我们先考虑一下如何执行此作。

您首先需要将模型质量总结为易于理解的方式。如果您比较 10,000 套房屋的预测和实际房屋价值，您可能会发现好预测和坏预测的混合。查看 10,000 个预测值和实际值的列表是毫无意义的。我们需要将其总结为一个指标。

总结模型质量的指标有很多，但我们将从一个称为平均绝对误差 （也称为 MAE）的指标开始。让我们从最后一个词错误开始分解这个指标。

使用 MAE 指标，我们取每个误差的绝对值。这会将每个错误转换为正数。然后我们取这些绝对误差的平均值。这是我们衡量模型质量的标准。用简单的英语来说，它可以说是：
    On average, our predictions are off by about X.
    平均而言，我们的预测偏离了大约 X。

要计算 MAE，我们首先需要一个模型。它内置在下面的隐藏单元格中，您可以通过单击代码按钮来查看该单元格。

'''

# Data Loading Code Hidden Here
import pandas as pd

# Load data
melbourne_file_path = './melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(X, y)

# 计算MAE
from sklearn.metrics import mean_absolute_error
predicted_home_prices = melbourne_model.predict(X)
# 输出：434.71594577146544
print(mean_absolute_error(y, predicted_home_prices))

'''
The measure we just computed can be called an "in-sample" score. We used a single "sample" of houses for both building the model and evaluating it. Here's why this is bad.
我们刚刚计算的度量可以称为“样本内”分数。我们使用单个房屋“样本”来构建模型和评估模型。这就是为什么这很糟糕。

Imagine that, in the large real estate market, door color is unrelated to home price.
想象一下，在大型房地产市场中，门的颜色与房价无关。

However, in the sample of data you used to build the model, all homes with green doors were very expensive. The model's job is to find patterns that predict home prices, so it will see this pattern, and it will always predict high prices for homes with green doors.
但是，在用于构建模型的数据样本中，所有带有绿色门的房屋都非常昂贵。该模型的工作是找到预测房价的模式，因此它会看到这种模式，并且它总是会预测绿色门的房屋的高价。

Since this pattern was derived from the training data, the model will appear accurate in the training data.
由于此模式是从训练数据派生而来的，因此模型在训练数据中将显得准确。

But if this pattern doesn't hold when the model sees new data, the model would be very inaccurate when used in practice.
但是，如果当模型看到新数据时这种模式不成立，那么该模型在实践中使用时将非常不准确。

Since models' practical value come from making predictions on new data, we measure performance on data that wasn't used to build the model. The most straightforward way to do this is to exclude some data from the model-building process, and then use those to test the model's accuracy on data it hasn't seen before. This data is called validation data.
由于模型的实用价值来自对新数据的预测，因此我们衡量未用于构建模型的数据的性能。最直接的方法是从模型构建过程中排除一些数据，然后使用这些数据来测试模型在以前从未见过的数据上的准确性。此数据称为验证数据 。

The scikit-learn library has a function train_test_split to break up the data into two pieces. We'll use some of that data as training data to fit the model, and we'll use the other data as validation data to calculate mean_absolute_error.
scikit-learn 库有一个函数 train_test_split 将数据分成两部分。我们将使用其中一些数据作为训练数据来拟合模型，我们将使用其他数据作为验证数据来计算 mean_absolute_error。

'''


from sklearn.model_selection import train_test_split
# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
# 输出：261986.9186571982
print(mean_absolute_error(val_y, val_predictions))

'''
Your mean absolute error for the in-sample data was about 500 dollars. Out-of-sample it is more than 250,000 dollars.
样本内数据的平均绝对误差约为 500 美元。样本外超过 250,000 美元。

This is the difference between a model that is almost exactly right, and one that is unusable for most practical purposes. 
As a point of reference, the average home value in the validation data is 1.1 million dollars. 
So the error in new data is about a quarter of the average home value.
这就是几乎完全正确的模型与无法用于大多数实际目的的模型之间的区别。作为参考，验证数据中的平均房屋价值为 110 万美元。因此，新数据中的误差约为平均房屋价值的四分之一。

There are many ways to improve this model, such as experimenting to find better features or different model types.
有许多方法可以改进此模型，例如尝试查找更好的特征或不同的模型类型。
'''