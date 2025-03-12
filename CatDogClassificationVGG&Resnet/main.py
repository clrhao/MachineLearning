import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import device
import matplotlib.pyplot as plt
from PIL import Image
from Resnet50 import Resnet, convolutional_block, identity_block  # 引入自定义网络模块
import VGG16
import torchvision.models as models

# 数据预处理：包括图像尺寸调整，转换为Tensor，归一化
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG16 的输入大小是 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化，虽然没有预训练模型，也可以用这个标准
])

# 加载数据集
train_data = datasets.ImageFolder('data', transform=transform)  # 加载数据集的根目录，cat和dog会自动作为子类
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 选择设备（MPS 或 CPU）
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')



# 实例化模型
# model = VGG16(num_classes=2)
#model = models.vgg16(pretrained=True)
model = models.resnet50(pretrained=True)
# model = Resnet(convolutional_block, identity_block)

model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 用于记录每轮的loss和accuracy
epoch_losses = []
epoch_accuracies = []

# 训练模型
num_epochs = 10  # 你可以根据需要调整训练的轮数

for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_losses.append(running_loss / len(train_loader))
    epoch_accuracies.append(100 * correct / total)

    # 打印每个 epoch 的训练损失和准确度
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# # 保存训练好的模型
# torch.save(model.state_dict(), 'vgg16.pth')

# 绘制损失和准确度曲线
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, epoch_losses, label='Training Loss', color='blue')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, epoch_accuracies, label='Training Accuracy', color='green')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True)

# 显示
plt.tight_layout()
plt.show()
