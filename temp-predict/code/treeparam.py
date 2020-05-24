# -*- encoding: utf-8 -*-
"""
@File    : treeparam.py
@Time    : 2020/5/24 10:46
@Author  : 王一宁
@Email   : wyn_365@163.com
参数调优
"""
import pandas as pd

# Read in data as a dataframe
features = pd.read_csv('D:\\APP\\PythonCharm2018\\workspace\\temp-predict\data\\temps_extended.csv')

# 编码
# One Hot Encoding
features = pd.get_dummies(features)

# Extract features and labels
labels = features['actual']
features = features.drop('actual', axis = 1)

# List of features for later use
feature_list = list(features.columns)

# Convert to numpy arrays
import numpy as np

features = np.array(features)
labels = np.array(labels)

# Training and Testing Sets
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                            test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

print('{:0.1f} years of data in the training set'.format(train_features.shape[0] / 365.))
print('{:0.1f} years of data in the test set'.format(test_features.shape[0] / 365.))

# 选取特征 95% of total importance
important_feature_names = ['temp_1', 'average', 'ws_1', 'temp_2', 'friend', 'year']

# Find the columns of the most important features
important_indices = [feature_list.index(feature) for feature in important_feature_names]

# Create training and testing sets with only the important features
important_train_features = train_features[:, important_indices]
important_test_features = test_features[:, important_indices]

# Sanity check on operations
print('Important train features shape:', important_train_features.shape)
print('Important test features shape:', important_test_features.shape)

# Use only the most important features
train_features = important_train_features[:]
test_features = important_test_features[:]

# Update feature list for visualizations
feature_list = important_feature_names[:]


# 开始调参
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state = 42)

from pprint import pprint

# 打印所有参数
pprint(rf.get_params())


# 随机查找
from sklearn.model_selection import RandomizedSearchCV

# 建立树的个数 200 210 220
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# 最大特征的选择方式
max_features = ['auto', 'sqrt']
# 树的最大深度
max_depth = [int(x) for x in np.linspace(10, 20, num = 2)]
max_depth.append(None)
# 节点最小分裂所需样本个数
min_samples_split = [2, 5, 10]
# 叶子节点最小样本数，任何分裂不能让其子节点样本数少于此值
min_samples_leaf = [1, 2, 4]
# 样本采样方法
bootstrap = [True, False]

# 参数空间 决定结果的好坏
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
# 随机选择最合适的参数组合 回归算法
rf = RandomForestRegressor()

# 随机100次 cv交叉验证 评估方法 打印指令 随机数字 所有CPU执行
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 100, scoring='neg_mean_absolute_error',
                              cv = 3, verbose=2, random_state=42, n_jobs=-1)

# 执行寻找操作 建立100个模型
rf_random.fit(train_features, train_labels)

# 最好的参数 输出！
rf_random.best_params_


print("========================================================")

# 参数评估
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape

    print('平均气温误差.',np.mean(errors))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

# 老模型 默认参数
base_model = RandomForestRegressor( random_state = 42)
base_model.fit(train_features, train_labels)
evaluate(base_model, test_features, test_labels)

# 最优的参数
best_random = rf_random.best_estimator_
evaluate(best_random, test_features, test_labels)