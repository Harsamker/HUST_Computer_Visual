import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import random
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset

BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCH = 30
NUMWORKER = 8


# 数据集类 - 创建成对的MNIST图像
class PairedMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.pairs = self._create_pairs()

    def _create_pairs(self):
        label_to_indices = {label: [] for label in range(10)}
        for idx, (_, label) in enumerate(self.mnist_dataset):
            label_to_indices[label].append(idx)

        pairs = []
        for idx in tqdm(range(len(self.mnist_dataset) // 10), desc="Creating pairs"):
            _, label = self.mnist_dataset[idx]

            # 同一标签的成对图像
            same_label_indices = label_to_indices[label]
            pair_idx = random.choice(same_label_indices)
            pairs.append((idx, pair_idx, 1))

            # 不同标签的成对图像
            different_label_indices = label_to_indices[random.choice([l for l in range(10) if l != label])]
            pair_idx = random.choice(different_label_indices)
            pairs.append((idx, pair_idx, 0))
        return pairs

    def __getitem__(self, index):
        idx1, idx2, same_class = self.pairs[index]
        img1, _ = self.mnist_dataset[idx1]
        img2, _ = self.mnist_dataset[idx2]
        # 将两个图像沿通道维度堆叠
        paired_image = torch.cat((img1, img2), 0)  # 结果形状为 [2, 28, 28]
        return paired_image, same_class

    def __len__(self):
        return len(self.pairs)


# 修改后的ResNet模型
class ModifiedResNet(nn.Module):
    def __init__(self, block, layers):
        super(ModifiedResNet, self).__init__()
        self.inplanes = 64  # 初始值为64，与self.conv1的输出通道数一致
        self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 数据转换
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# 创建成对的数据集
train_paired_dataset = PairedMNIST(train_dataset)
test_paired_dataset = PairedMNIST(test_dataset)

# 创建数据加载器
train_loader = DataLoader(dataset=train_paired_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(dataset=test_paired_dataset,    batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# 模型初始化
model = ModifiedResNet(BasicBlock, [2, 4, 4, 2]).to(DEVICE)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)
# 添加学习率调度器
# step 10, x0.2
#scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
# 测试循环
test_loss_list = []
test_acc_list = []
# 记录训练指标的列表
train_loss_list = []
train_acc_list = []

for epoch in range(EPOCH):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCH}", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.size(0)

        # 更新进度条的描述
        progress_bar.set_description(
            f"Epoch {epoch + 1}/{EPOCH} [Train Loss: {train_loss / total:.4f}, Train Acc: {100. * correct / total:.2f}%]")

    #scheduler.step()

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / total

    # 记录训练指标
    train_loss_list.append(train_loss)
    train_acc_list.append(train_accuracy)

    print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # 测试部分
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            # 在这里应用softmax
            outputs_softmax = F.softmax(outputs, dim=1)
            test_loss += criterion(outputs_softmax, labels).item()
            pred = outputs_softmax.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    # 记录测试指标
    test_loss_list.append(test_loss)
    test_acc_list.append(test_accuracy)

    print(f"Epoch {epoch + 1}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


# 保存模型
torch.save(model.state_dict(), "paired_mnist_resnet_model.pth")

plt.figure(figsize=(15, 8))

# 显示损失
plt.subplot(2, 1, 1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.title('Train and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 显示准确率
plt.subplot(2, 1, 2)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()  # 调整布局，避免重叠
plt.show()
