# part6: Random Forests  随机森林
# https://www.kaggle.com/code/dansbecker/random-forests

'''
Introduction  介绍¶
Decision trees leave you with a difficult decision. A deep tree with lots of leaves will overfit because each prediction is coming from historical data from only the few houses at its leaf. But a shallow tree with few leaves will perform poorly because it fails to capture as many distinctions in the raw data.
决策树让您面临艰难的决定。具有大量叶子的深树将过度拟合，因为每个预测仅来自其叶子处的几座房屋的历史数据。但是，叶子很少的浅树性能会很差，因为它无法在原始数据中捕获尽可能多的区别。

Even today's most sophisticated modeling techniques face this tension between underfitting and overfitting. But, many models have clever ideas that can lead to better performance. We'll look at the random forest as an example.
即使是当今最复杂的建模技术也面临着欠拟合和过拟合之间的这种紧张关系。但是，许多模型都有聪明的想法，可以带来更好的性能。我们将以随机森林为例。

The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. It generally has much better predictive accuracy than a single decision tree and it works well with default parameters. If you keep modeling, you can learn more models with even better performance, but many of those are sensitive to getting the right parameters.
随机森林使用许多树，它通过平均每个组件树的预测来做出预测。它通常比单个决策树具有更好的预测准确性，并且适用于默认参数。如果您继续建模，您可以学习更多性能更好的模型，但其中许多模型对于获得正确的参数很敏感。
'''

import pandas as pd

# Load data
melbourne_file_path = './melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
# Filter rows with missing values
melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

'''
We build a random forest model similarly to how we built a decision tree in scikit-learn - this time using the RandomForestRegressor class instead of DecisionTreeRegressor.
我们构建随机森林模型，类似于我们在 scikit-learn 中构建决策树的方式 - 这次使用 RandomForestRegressor 类而不是 DecisionTreeRegressor。

'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))


'''
Conclusion  结论¶
There is likely room for further improvement, but this is a big improvement over the best decision tree error of 250,000. There are parameters which allow you to change the performance of the Random Forest much as we changed the maximum depth of the single decision tree. But one of the best features of Random Forest models is that they generally work reasonably even without this tuning.
可能还有进一步改进的空间，但这比 250,000 的最佳决策树误差有了很大的改进。有一些参数允许您更改随机森林的性能，就像我们更改了单个决策树的最大深度一样。但随机森林模型的最佳功能之一是，即使没有这种调整，它们通常也能正常工作。
'''