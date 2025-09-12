# part5: Underfitting and Overfitting (欠拟合和过拟合)
# https://www.kaggle.com/code/dansbecker/underfitting-and-overfitting


'''
Experimenting With Different Models¶
试验不同的模型 ¶
Now that you have a reliable way to measure model accuracy, you can experiment with alternative models and see which gives the best predictions. But what alternatives do you have for models?
现在，您有了一种可靠的方法来衡量模型准确性，您可以试验替代模型，看看哪个模型给出了最好的预测。但是，您有哪些型号选择呢？

You can see in scikit-learn's documentation that the decision tree model has many options (more than you'll want or need for a long time). The most important options determine the tree's depth. Recall from the first lesson in this course that a tree's depth is a measure of how many splits it makes before coming to a prediction. This is a relatively shallow tree
您可以在 scikit-learn 的文档中看到，决策树模型有很多选项（比您长时间想要或需要的要多）。最重要的选项决定了树的深度。回想一下本课程的第一节课，树的深度是衡量它在进行预测之前进行了多少次拆分的量度。这是一棵相对较浅的树

Depth 2 Tree

In practice, it's not uncommon for a tree to have 10 splits between the top level (all houses) and a leaf. As the tree gets deeper, the dataset gets sliced up into leaves with fewer houses. If a tree only had 1 split, it divides the data into 2 groups. If each group is split again, we would get 4 groups of houses. Splitting each of those again would create 8 groups. If we keep doubling the number of groups by adding more splits at each level, we'll have
2
10
 groups of houses by the time we get to the 10th level. That's 1024 leaves.
在实践中，一棵树在顶层（所有房屋）和一片叶子之间有 10 次分裂的情况并不少见。随着树的深入，数据集被切成房屋较少的叶子。如果一棵树只有 1 个拆分，它会将数据分为 2 组。如果每组再次拆分，我们将得到 4 组宫位。再次拆分每个组将创建 8 个组。如果我们通过在每个级别添加更多的拆分来继续增加组的数量，那么当我们到达第 10 级时，我们将拥有
2^10 一组房屋。那是 1024 片叶子。

When we divide the houses amongst many leaves, we also have fewer houses in each leaf. Leaves with very few houses will make predictions that are quite close to those homes' actual values, but they may make very unreliable predictions for new data (because each prediction is based on only a few houses).
当我们在许多叶子中划分房屋时，每片叶子中的房屋也会减少。房屋很少的叶子将做出非常接近这些房屋实际值的预测，但它们可能会对新数据做出非常不可靠的预测（因为每个预测仅基于少数房屋）。

This is a phenomenon called overfitting, where a model matches the training data almost perfectly, but does poorly in validation and other new data. On the flip side, if we make our tree very shallow, it doesn't divide up the houses into very distinct groups.
这是一种称为过拟合的现象，即模型与训练数据几乎完美匹配，但在验证和其他新数据方面表现不佳。另一方面，如果我们把树做得很浅，它不会把房子分成非常不同的组。

At an extreme, if a tree divides houses into only 2 or 4, each group still has a wide variety of houses. Resulting predictions may be far off for most houses, even in the training data (and it will be bad in validation too for the same reason). When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data, that is called underfitting.
在极端情况下，如果一棵树只将房屋分成 2 或 4 个，则每组仍然有各种各样的房屋。对于大多数房屋来说，结果预测可能相差甚远，即使在训练数据中也是如此（出于同样的原因，验证也会很糟糕）。当模型无法捕获数据中的重要区别和模式时，即使在训练数据中也表现不佳，这称为欠拟合。

Since we care about accuracy on new data, which we estimate from our validation data, we want to find the sweet spot between underfitting and overfitting. Visually, we want the low point of the (red) validation curve in the figure below.
由于我们关心新数据的准确性（我们从验证数据中估计），因此我们希望找到欠拟合和过拟合之间的最佳平衡点。从视觉上看，我们想要下图中（红色）验证曲线的低点。

underfitting_overfitting

Example  示例 ¶
There are a few alternatives for controlling the tree depth, and many allow for some routes through the tree to have greater depth than other routes. But the max_leaf_nodes argument provides a very sensible way to control overfitting vs underfitting. The more leaves we allow the model to make, the more we move from the underfitting area in the above graph to the overfitting area.
有几种方法可以控制树深度，许多方法允许通过树的某些路径比其他路径具有更大的深度。但是 max_leaf_nodes 论点提供了一种非常明智的方法来控制过拟合与欠拟合。我们允许模型制作的叶子越多，我们从上图中的欠拟合区域移动到过拟合区域的次数就越多。

We can use a utility function to help compare MAE scores from different values for max_leaf_nodes:
我们可以使用效用函数来帮助比较 max_leaf_nodes 不同值的 MAE 分数：
'''


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# Data Loading Code Runs At This Point
import pandas as pd

# Load data
melbourne_file_path = './melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
# Filter rows with missing values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

'''
输出：
Max leaf nodes: 5  		 Mean Absolute Error:  347380
Max leaf nodes: 50  		 Mean Absolute Error:  258171
Max leaf nodes: 500  		 Mean Absolute Error:  243495
Max leaf nodes: 5000  		 Mean Absolute Error:  254983

Of the options listed, 500 is the optimal number of leaves.
在列出的选项中，500 是最佳叶数。

'''


'''
Here's the takeaway: Models can suffer from either:
要点如下：模型可能会遭受以下任一情况：

> Overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions, or
过拟合：捕获将来不会再次出现的虚假模式，导致预测不准确，或者
> Underfitting: failing to capture relevant patterns, again leading to less accurate predictions.
欠拟合：未能捕获相关模式，再次导致预测不准确。

We use validation data, which isn't used in model training, to measure a candidate model's accuracy. This lets us try many candidate models and keep the best one.
我们使用模型训练中未使用的验证数据来衡量候选模型的准确性。这让我们可以尝试许多候选模型并保留最好的模型。
'''
