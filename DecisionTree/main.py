# %% [markdown]
# # 决策树
#  
# 
# ## 实验要求
# - 基本要求
#     1. 基于 Watermelon-train1数据集（只有离散属性），构造ID3决策树；
#     2. 基于构造的 ID3 决策树，对数据集 Watermelon-test1进行预测，输出分类精度；
# - 中级要求
#     1. 对数据集Watermelon-train2，构造C4.5或者CART决策树，要求可以处理连续型属性；
#     2. 对测试集Watermelon-test2进行预测，输出分类精度；


# %%
import numpy as np 
import pandas as pd
import math
import copy
def readfile(name):
    df = pd.read_csv(name, error_bad_lines = False, encoding = 'gbk')
    temp = np.array(df).tolist()
    for i in temp:
        i.pop(0)
    return temp

train1 = readfile("Watermelon-train1.csv")
train2 = readfile("Watermelon-train2.csv")
test1 = readfile("Watermelon-test1.csv")
test2 = readfile("Watermelon-test2.csv")



# %%
def information(data):
    dic = {}
    for i in data:
        current = i[-1] #取出最后的结果
        if current not in dic.keys(): 
            dic[current] = 1 #创建一个新的类别
        else:
            dic[current] += 1 #原有类别+1
    result = 0.0
    for key in dic:
        prob = float(dic[key]) / len(data)
        result -= prob * math.log(prob,2) 
    return result, dic



# %%
def split(data, index, kind):
    ls = []
    for temp in data:
        if temp[index] == kind:
            t = temp[0: index]
            t = t + temp[index + 1: ]
            ls.append(t)
    return ls



# %%
def chooseBestFeatureToSplit(data):
    base, mm= information(data) #原始的信息熵
    best = 0
    bestindex = -1
    for i in range(len(data[0]) - 1):
        #抽取该列数据的所有信息
        ls = [index[i] for index in data]
        feture = set(ls) 
        #计算该列的信息增益
        temp = 0.0
        for value in feture:
            datatemp = split(data, i, value)
            prob = len(datatemp) / float(len(data))
            t, mm = information(datatemp)
            temp += prob * t
        infoGain = base - temp
        #根据信息增益挑选 
        if infoGain > best:
            best = infoGain
            bestindex = i
    return bestindex 



# %%
def classify1(data, labels):
    typelist = [index[-1] for index in data] #取出该数据集的分类
    nothing, typecount = information(data) #计算出类别种类以及数量
    if len(typecount) == 1: #如果只有一个类别
        return typelist[0]
    bestindex = chooseBestFeatureToSplit(data)  # 最优划分属性的索引
    bestlabel = labels[bestindex]
    Tree = {bestlabel: {}}
    temp = labels[:]
    del (temp[bestindex])  # 已经选择的特征不再参与分类，将该类别删除
    feature = [example[bestindex] for example in data]
    unique = set(feature)  # 该属性所有可能取值，也就是节点的分支
    for i in unique:  
        temp = temp[:] 
        Tree[bestlabel][i] = classify1(split(data, bestindex, i), temp)
    return Tree



# %%
def run1(testdata, tree, labels):
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    featIndex = labels.index(firstStr)
    result = ''
    for key in list(secondDict.keys()): 
         if testdata[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':  # 该分支不是叶子节点，递归
                result = run1(testdata, secondDict[key], labels)
            else:
                result = secondDict[key]
    return result

def getresult(train, test, labels):
    ls = []
    tree = classify1(train, labels)
    print("生成决策树如下:", tree)
    for index in test:
        ls.append(run1(index, tree, labels))
    return ls

labels1 = ['色泽', '根蒂', '敲声', '纹理', '好瓜']
result1 = getresult(train1, test1, labels1)



# %%
def simrate(data, predict):
    num = 0
    for i in range(len(data)):
        if data[i][-1] == predict[i]:
            num +=1
    return format(num / len(data), '.2%')
print("ID3分类器对于test1数据集的准确率是：", simrate(test1, result1))





# %%
def Division(data):
    ls = data[:]
    ls.sort()
    result = []
    for i in range(len(ls) - 1):
        result.append((data[i + 1] + data[i]) / 2)
    return result



# %%
def split2(data, index, kind, method):
    ls = []
    if method == 0:
        for temp in data:
            if temp[index] <= kind:
                t = temp[0 : index]
                t = t + temp[index + 1 : ]
                ls.append(t)
    else:
        for temp in data:
            if temp[index] > kind:
                t = temp[0 : index]
                t = t + temp[index + 1 : ]
                ls.append(t)
    return ls



# %%
def chooseBestFeatureToSplit2(data):
    base, mm= information(data) #原始的信息熵
    info = []
    for j in range(len(data[0]) - 1):
        dic = {}
        for i in data:
            current = i[j] #取出最后的结果
            if current not in dic.keys(): 
                dic[current] = 1 #创建一个新的类别
            else:
                dic[current] += 1 #原有类别+1
        result = 0.0
        for key in dic:
            prob = float(dic[key]) / len(data)
            result -= prob * math.log(prob,2) 
        info.append(result)
    best = 0
    bestindex = -1
    bestpartvalue = None #如果是离散值，使用该值进行分割
    for i in range(len(data[0]) - 1):
        #抽取该列数据的所有信息
        ls = [index[i] for index in data]
        feture = set(ls) 
        #计算该列的信息增益
        temp = 0.0
        if type(ls[0]) == type("a"):#判断是否时离散的
            for value in feture:
                datatemp = split(data, i, value)
                prob = len(datatemp) / float(len(data))
                t, mm = information(datatemp)
                temp += prob * t
        else:
            ls.sort()
            min = float("inf")
            for j in range(len(ls) - 1):
                part = (ls[j + 1] + ls[j]) / 2 #计算划分点
                left = split2(data, i, part, 0)
                right = split2(data, i, part, 1)
                temp1, useless = information(left)
                temp2, useless = information(right)
                temp = len(left) / len(data) * temp1 + len(right) / len(data) * temp2
                if temp < min:
                    min = temp
                    bestpartvalue = part
            temp = min
        infoGain = base - temp
        #根据信息增益挑选 
        if info[i] != 0:
            if infoGain / info[i] >= best:
                best = infoGain / info[i]
                bestindex = i
    return bestindex, bestpartvalue




# %%
def classify2(data, labels):
    typelist = [index[-1] for index in data] #取出该数据集的分类
    nothing, typecount = information(data) #计算出类别种类以及数量
    if typecount == len(typelist): #如果只有一个类别
        return typelist[0]
    bestindex, part = chooseBestFeatureToSplit2(data)  # 最优划分属性的索引
    if bestindex == -1:
        return "是"
    if type([t[bestindex] for t in data][0]) == type("a"):#判断是否时离散的
        bestlabel = labels[bestindex]
        Tree = {bestlabel: {}}
        temp = copy.copy(labels)
        feature = [example[bestindex] for example in data]
        del (temp[bestindex])  # 已经选择的特征不再参与分类，将该类别删除
        unique = set(feature)  # 该属性所有可能取值，也就是节点的分支
        for i in unique:  
            s = temp[:] 
            Tree[bestlabel][i] = classify2(split(data, bestindex, i), s)
    else: #连续的变量
        bestlabel = labels[bestindex] + "<" + str(part)
        Tree = {bestlabel: {}}
        temp = labels[:]
        del(temp[bestindex])
        leftdata = split2(data, bestindex, part, 0)
        Tree[bestlabel]["是"] = classify2(leftdata, temp)
        rightdata = split2(data, bestindex, part, 1)
        Tree[bestlabel]["否"] = classify2(rightdata, temp)
    return Tree



# %%
def run2(data, tree, labels):
    firstStr = list(tree.keys())[0]  # 根节点
    firstLabel = firstStr
    t = str(firstStr).find('<') #查看是否是连续型
    if t > -1:  # 如果是连续型的特征
        firstLabel = str(firstStr)[ : t]
    secondDict = tree[firstStr]
    featIndex = labels.index(firstLabel)  # 跟节点对应的属性
    result = ''
    for key in list(secondDict.keys()):  # 对每个分支循环
        if type(data[featIndex]) == type("a"):
            if data[featIndex] == key:  # 测试样本进入某个分支
                if type(secondDict[key]).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    result = run2(data, secondDict[key], labels)
                else:  # 如果是叶子， 返回结果
                    result = secondDict[key]
        else:
            value = float(str(firstStr)[t + 1 : ])
            if data[featIndex] <= value:
                if type(secondDict['是']).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    result = run2(data, secondDict['是'], labels)
                else:  # 如果是叶子， 返回结果
                    result = secondDict['是']
            else:
                if type(secondDict['否']).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    result = run2(data, secondDict['否'], labels)
                else:  # 如果是叶子， 返回结果
                    result = secondDict['否']
    return result

def getresult2(train, test, labels):
    ls = []
    tree = classify2(train, labels)
    print("生成决策树如下:", tree)
    for index in test:
        ls.append(run2(index, tree, labels))
    return ls

labels2 = ["色泽", "根蒂", "敲声", "纹理", "密度", "好瓜"]
result2 = getresult2(train2, test2, labels2)


# %%
print("C4.5 accuracy：", simrate(test2, result2))




