import numpy as np
import math
import random
# from sklearn.model_selection import train_test_split
#
#
# # data = np.loadtxt('wine.data', delimiter=',')
# # # print(data)
# # # print(type(data))
# # X = data[:, 1:]  # 取所有行，除最后一列外的所有列作为特征
# # y = data[:, 0]   # 取所有行，最后一列作为标签
# # # print(X.shape)
# # # print(y.shape)
# # # print(np.unique(y))
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
#
#
#
#
# def normal_distribution(x, mean, std): # std (float): 标准差 σ
#     coefficient = 1 / (math.sqrt(2 * math.pi) * std)
#     exponent = math.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
#     return coefficient * exponent
#
# def trainPbmodel_X(feats):
#     N,D = np.shape(feats)
#
#     model = {}
#     for d in range(D):
#         data = feats[:, d].tolist()
#         mean = np.mean(data)
#         std = np.std(data)
#         keys = set(data) # ???
#         model[d] = {}
#         for key in keys:
#             model[d][key] = normal_distribution(data[key], mean, std)
#     return model
#
# def trainPbmodel(datas, labs):
#     model = {}
#     keys = set(labs)
#     for key in keys:
#         Pbmodel_Y = labs.count(key)/len(labs)
#
#         index = np.where(np.array(labs) == key)[0].tolist() # 获取标签为y的位次
#         feats = np.array(datas)[index] # 获取标签为y的训练集
#
#         Pbmodel_X = trainPbmodel_X(feats)
#
#         model[key] = {}
#         model[key]["PY"] = Pbmodel_Y
#         model[key]["PX"] = Pbmodel_X
#     return model
#
# # 改
# def getPbfromModel(feat, model, keys):
#     results = {}
#     eps = 0.00001
#     for key in keys:
#         PY = model.get(key,eps).get("PY")
#         model_X = model.get(key,eps).get("PX") # 如果键值不存在则返回无穷小
#         list_px = [] ###
#         for d in range(len(feat)):
#             pb = model_X.get(d,eps).get(feat[d],eps)
#             list_px.append(pb)
#         result = np.log(PY) + np.sum(np.log(list_px)) ###
#         results[key] = result
#     return results
#
# with open("wine.data",'r') as f:
#     lines = f.read().splitlines()
# dataSet = [line.split('\t') for line in lines]

# 训练
# random.seed(42)
# test_size = 0.2
# test_count = int(len(dataSet) * test_size)
# indices = list(range(len(dataSet)))
# random.shuffle(indices)
# train_indices = indices[:-test_count]
# test_indices = indices[-test_count:]
#
# train_data = [dataSet[i] for i in train_indices]
# test_data = [dataSet[i] for i in test_indices]
#
#
# datas = [i[1:] for i in train_data]
# labs = [i[0] for i in train_data]
# datas = [i[1:] for i in train_data]
# labs = [i[0] for i in train_data]
#
# keys = set(labs)
# model = trainPbmodel(datas, labs)
# print(model)

# 测试
# for line in
def trainPbmodel_X(feats):
    """
    Train feature-wise Gaussian models for continuous data.
    """
    N, D = feats.shape
    model = {}
    for d in range(D):  # Loop through features
        feature_data = feats[:, d]
        mean = np.mean(feature_data)
        std = np.std(feature_data) if np.std(feature_data) > 0 else 1e-6  # Avoid zero std
        model[d] = {"mean": mean, "std": std}
    return model

def trainPbmodel(datas, labs):
    """
    Train the Naive Bayes model for the given data and labels.
    """
    model = {}
    classes = np.unique(labs)
    for cls in classes:
        indices = np.where(labs == cls)[0]
        feats = datas[indices]
        model[cls] = {
            "PY": len(indices) / len(labs),
            "PX": trainPbmodel_X(feats)
        }
    return model

def normal_distribution(x, mean, std):
    coefficient = 1 / (math.sqrt(2 * math.pi) * std)
    exponent = math.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
    return coefficient * exponent

def getPbfromModel(feat, model, keys):
    results = {}
    eps = 1e-6
    for key in keys:
        PY = model[key]["PY"]
        PX = model[key]["PX"]
        log_prob = np.log(PY)
        for d, x in enumerate(feat):
            mean = PX[d]["mean"]
            std = PX[d]["std"]
            prob = normal_distribution(x, mean, std)
            log_prob += np.log(prob + eps)  # Avoid log(0)
        results[key] = log_prob
    return results

import numpy as np

# Generate some sample data for testing
np.random.seed(42)

# Create a toy dataset with two classes (0 and 1)
# Class 0: Mean = 2.0, Std = 0.5
# Class 1: Mean = 5.0, Std = 1.0
class_0 = np.random.normal(2.0, 0.5, (50, 2))  # 50 samples, 2 features
class_1 = np.random.normal(5.0, 1.0, (50, 2))  # 50 samples, 2 features

# Combine the data and labels
data = np.vstack((class_0, class_1))
labels = np.array([0] * 50 + [1] * 50)

# Shuffle the dataset
indices = np.random.permutation(len(data))
data = data[indices]
labels = labels[indices]

# Split into training and test sets
train_data = data[:80]
train_labels = labels[:80]
test_data = data[80:]
test_labels = labels[80:]

# Train the Naive Bayes model
model = trainPbmodel(train_data, train_labels)

# Test the model on test data
correct_predictions = 0
for i, feat in enumerate(test_data):
    probabilities = getPbfromModel(feat, model, keys=np.unique(labels))
    predicted_class = max(probabilities, key=probabilities.get)
    true_class = test_labels[i]
    if predicted_class == true_class:
        correct_predictions += 1
    print(f"Sample {i}: True Class = {true_class}, Predicted Class = {predicted_class}, Probabilities = {probabilities}")

# Calculate accuracy
accuracy = correct_predictions / len(test_data)
print(f"\nTest Accuracy: {accuracy:.2f}")

