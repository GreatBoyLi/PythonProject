from liwp import litorch as liwp
import pandas as pd
import torch
import torch.nn as nn


train_data = pd.read_csv(liwp.download('kaggle_house_train'))
test_data = pd.read_csv(liwp.download('kaggle_house_test'))

# 去除第一列
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# 获取数值类型的特征
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# 对数值类型的特征进行标准化，减均值除方差
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 将缺失值填充为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# 将非数值转换为指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)

print(all_features.MSSubClass.values)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

loss = nn.MSELoss()
in_features = train_features.shape[1]


def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net
