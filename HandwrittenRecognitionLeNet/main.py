import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch.nn.functional as F
from torchinfo import summary
import os
from datetime import datetime

# 记录开始时间
start_time = datetime.now()
# 检查 MPS
device = torch.device("mps")
print(torch.backends.mps.is_available())

# 下载路径
data_path = os.path.join(os.getcwd(), "data")

# 预处理
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),  # 将图片大小调整为 32x32
    torchvision.transforms.ToTensor()  # 转换为 Tensor
])

# 加载 MNIST 数据集
train_ds = torchvision.datasets.MNIST(data_path, train=True, transform=transform, download=True)
test_ds = torchvision.datasets.MNIST(data_path, train=False, transform=transform, download=True)


batch_size = 32

# 数据加载器
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

# 查看一个批次的数据
imgs, labels = next(iter(train_dl))
print("Batch shape:", imgs.shape)

# 显示样本图片
plt.figure(figsize=(20, 5))
for i, img in enumerate(imgs[:20]):
    npimg = np.squeeze(img.numpy())
    plt.subplot(2, 10, i + 1)
    plt.imshow(npimg, cmap=plt.cm.binary)
    plt.axis('off')
plt.show()

# 定义模型
num_classes = 10


import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 输入 16 个 5x5 的特征图，输出 120
        self.fc2 = nn.Linear(120, 84)  # 输出 84
        self.fc3 = nn.Linear(84, 10)  # 输出 10 个类别（MNIST）

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 2x2 的最大池化
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)  # 展平
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# 初始化
model = LeNet().to(device)
summary(model)

# 损失函数 优化器
loss_fn = nn.CrossEntropyLoss()
learn_rate = 1e-2
opt = torch.optim.SGD(model.parameters(), lr=learn_rate)

# 训练
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, train_acc = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()

    return train_acc / size, train_loss / num_batches

# 测试
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, test_acc = 0, 0

    with torch.no_grad():
        for imgs, target in dataloader:
            imgs, target = imgs.to(device), target.to(device)
            target_pred = model(imgs)
            loss = loss_fn(target_pred, target)
            test_loss += loss.item()
            test_acc += (target_pred.argmax(1) == target).type(torch.float).sum().item()

    return test_acc / size, test_loss / num_batches

# 训练
epochs = 8
train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):

    epoch_train_acc, epoch_train_loss = train(train_dl, model, loss_fn, opt)
    epoch_test_acc, epoch_test_loss = test(test_dl, model, loss_fn)

    train_acc.append(epoch_train_acc)
    train_loss.append(epoch_train_loss)
    test_acc.append(epoch_test_acc)
    test_loss.append(epoch_test_loss)

    template = 'Epoch:{:2d}, Train_acc:{:.1f}%, Train_loss:{:.3f}, Test_acc:{:.1f}%, Test_loss:{:.3f}'
    print(template.format(epoch + 1, epoch_train_acc * 100, epoch_train_loss, epoch_test_acc * 100, epoch_test_loss))

print('Done')

# 绘制 准确率损失


plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

epochs_range = range(epochs)
plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc, label='Training Accuracy')
plt.plot(epochs_range, test_acc, label='Test Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, test_loss, label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

end_time = datetime.now()
elapsed_time = end_time - start_time

print(f"程序运行时间: {elapsed_time}")