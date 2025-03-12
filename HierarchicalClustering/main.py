# %% [markdown]
# 聚类层次
# 
# 
# ## 实验要求
# - 数据集 <br/>
# 生成 2000 个样例，每个样例的前 3 列表示特征，第 4 列表示标签
# - 基本要求
#     - 绘制聚类前后样本分布情况
#     - 实现 single-linkage 层次聚类算法
#     - 实现 complete-linkage 层次聚类算法
# <br/>
# - 中级要求 <br/>
# 实现 average-linkage 层次聚类算法，绘制样本分布图
# <br/>
# - 提高要求<br/>
# 对比上述三种算法，给出结论。

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def random_x(cnt1, cnt2, cnt3, name): 
    cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    a1 = np.random.multivariate_normal((1, 1 ,1), cov, cnt1) #满足均值矢量
    a2 = np.random.multivariate_normal((4, 4 ,4), cov, cnt2)
    a3 = np.random.multivariate_normal((8, 1 ,1), cov, cnt3)
    colors0 = '#000000'
    colors1 = '#00CED1' #点的颜色
    colors2 = '#DC143C'
    area = np.pi   # 点面积
    fig = plt.figure()
    ax = Axes3D(fig)
    x = a1[:, 0]
    y = a1[:, 1] 
    z = a1[:, 2]
    ax.scatter(x, y, z, c = colors0, s = area)
    x = a2[:, 0]
    y = a2[:, 1] 
    z = a2[:, 2]
    ax.scatter(x, y, z, c = colors1, s = area)
    x = a3[:, 0]
    y = a3[:, 1] 
    z = a3[:, 2]
    ax.scatter(x, y, z, c = colors2, s = area)

    ls = []
    result = []
    for i in range(cnt1):
        ls.append(a1[i])
        result.append(1)
    for i in range(cnt2):
        ls.append(a2[i])
        result.append(2)
    for i in range(cnt3):
        ls.append(a3[i])
        result.append(3)
    plt.figure(num = name) 
    return ls, result



# %%
def single_linkage(kind1, kind2, data):
    min = float("inf")
    for i in range(len(kind1)):
        for j in range(len(kind2)):
            t = np.linalg.norm(data[kind1[i]] - data[kind2[j]])
            if t < min:
                min = t
    return min



# %%
def complete_linkage(kind1, kind2, data):
    max = float("-inf")
    for i in range(len(kind1)):
        for j in range(len(kind2)):
            t = np.linalg.norm(data[kind1[i]] - data[kind2[j]])
            if t > max:
                max = t
    return max


# %%
def average_linkage(kind1, kind2, data):
    avg = 0.0
    for i in range(len(kind1)):
        for j in range(len(kind2)):
            avg += np.linalg.norm(data[kind1[i]] - data[kind2[j]]) 
    return avg / (len(kind1) * len(kind2))



# %%
def distance(data):
    ls = [ [] for i in range(len(data))]
    for i in range(len(data)):
        for j in range(len(data)):
            if i >= j:
                ls[i].append(float("inf"))
            else:
                ls[i].append(np.linalg.norm(data[i] - data[j]))
    return ls   



# %%
def Clustering(data, dis, func, num = 3):
    category = []
    for i in range(len(data)):
         temp = []
         temp.append(i)
         category.append(temp)
    while len(category) > num:
         t = np.asarray(dis)
         w = t.shape[1]
         position = t.argmin()                         
         row, col = position // w, position % w  #找到矩阵中的最小值
         #更改分类中的内容
         if row > col:
              temp = category.pop(row)
              temp = temp + category[col]
              category[col] = temp
         else:
              temp = category.pop(col)
              temp = temp + category[row]
              category[row] = temp
         if func == 0:
              if row > col:
                   dis.pop(row)
                   for i in range(len(dis)):
                         dis[i].pop(row)
                   for i in range(col + 1, len(dis)):
                         dis[col][i] = single_linkage(category[col], category[i], data)
                   for i in range(0, col - 1):
                         dis[i][col] = single_linkage(category[col], category[i], data)
              else:
                   dis.pop(col)
                   for i in range(len(dis)):
                         dis[i].pop(col)
                   for i in range(row + 1, len(dis)):
                         dis[row][i] = single_linkage(category[row], category[i], data)
                   for i in range(0, row - 1):
                         dis[i][row] = single_linkage(category[row], category[i], data)
         elif func == 1:
              if row > col:
                   dis.pop(row)
                   for i in range(len(dis)):
                         dis[i].pop(row)
                   for i in range(col + 1, len(dis)):
                         dis[col][i] = complete_linkage(category[col], category[i], data)
                   for i in range(0, col - 1):
                         dis[i][col] = complete_linkage(category[col], category[i], data)
              else:
                   dis.pop(col)
                   for i in range(len(dis)):
                         dis[i].pop(col)
                   for i in range(row + 1, len(dis)):
                         dis[row][i] = complete_linkage(category[row], category[i], data)
                   for i in range(0, row - 1):
                         dis[i][row] = complete_linkage(category[row], category[i], data)
         elif func == 2:
              if row > col:
                   dis.pop(row)
                   for i in range(len(dis)):
                         dis[i].pop(row)
                   for i in range(col + 1, len(dis)):
                         dis[col][i] = average_linkage(category[col], category[i], data)
                   for i in range(0, col - 1):
                         dis[i][col] = average_linkage(category[col], category[i], data)
              else:
                   dis.pop(col)
                   for i in range(len(dis)):
                         dis[i].pop(col)
                   for i in range(row + 1, len(dis)):
                         dis[row][i] = average_linkage(category[row], category[i], data)
                   for i in range(0, row - 1):
                         dis[i][row] = average_linkage(category[row], category[i], data)
    return category

# %%
def show(result, data):
    x = [[] for i in range(len(result))]
    y = [[] for i in range(len(result))]
    z = [[] for i in range(len(result))]
    for i in range(len(result)):
        for j in range(len(result[i])):
            x[i].append(data[result[i][j]][0])
            y[i].append(data[result[i][j]][1])
            z[i].append(data[result[i][j]][2])
    colors0 = '#000000'
    colors1 = '#00CED1' #点的颜色
    colors2 = '#DC143C'
    area = np.pi   # 点面积
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x[0], y[0], z[0], c = colors0, s = area)
    ax.scatter(x[1], y[1], z[1], c = colors1, s = area)
    ax.scatter(x[2], y[2], z[2], c = colors2, s = area)



# %%
data2, k2 = random_x(666, 667, 667, "散点图")
dis2 = distance(data2)
result2 = Clustering(data2, dis2, 0)
show(result2, data2)

# %%
data3, k3 = random_x(666, 667, 667, "散点图")
dis3 = distance(data3)
result3 = Clustering(data3, dis3, 1)
show(result3, data3)


# %%
data1, k1 = random_x(666, 667, 667, "散点图")
dis1 = distance(data1)
result1 = Clustering(data1, dis1, 2)
show(result1, data1)



