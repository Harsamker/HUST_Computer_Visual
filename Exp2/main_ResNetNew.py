import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS=4
LearnRate = 0.01
EPOCH=10
# 定义ResNet基本块
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.match_dimensions = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.relu_match = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.match_dimensions(residual)

        out += residual
        out = self.relu_match(out)

        return out

# 定义ResNet网络
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset=datasets.MNIST(root="./data", train=False,download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True)


net = ResNet(ResNetBlock, [2, 2, 2, 2], num_classes=10).to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LearnRate)

# 记录每个epoch的损失和准确率
test_losses=[]
test_accuracies=[]

# 记录每个epoch的损失和准确率
train_loss_list = []
train_acc_list = []


for epoch in range(EPOCH):  # 做5个epoch的训练，可以根据需要调整
    process_bar = tqdm(enumerate(train_loader, 0), desc=f"Epoch {epoch + 1}", unit="batch", total=len(train_loader))
    #running_loss = 0.0
    mini_batch_loss_train = []  # 用于记录每一轮mini-batch的训练集损失
    for i, data in process_bar:
        net.train()  # 切换到训练模式
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        '''
        running_loss += loss.item()
        mini_batch_loss_train.append(running_loss / 100)
        running_loss = 0.0
    plt.plot(mini_batch_loss_train, label=f'Train Epoch {epoch + 1}')
    plt.xlabel('Mini-batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Mini-batch Training Loss - Epoch {epoch + 1}')
    plt.show()
    net.eval()
    '''
    total_train_loss = 0.0
    correct_train = 0
    total_train = 0

    # 计算训练集损失和准确率
    with torch.no_grad():
        for train_data in train_loader:
            train_images, train_labels = train_data
            train_images, train_labels = train_images.to(DEVICE), train_labels.to(DEVICE)
            train_outputs = net(train_images)
            train_loss = criterion(train_outputs, train_labels)
            total_train_loss += train_loss.item()
            _, train_predict = torch.max(train_outputs.data, 1)
            total_train += train_labels.size(0)
            correct_train += (train_predict == train_labels).sum().item()

    train_loss_epoch = total_train_loss / len(train_loader)
    train_acc_epoch = 100 * correct_train / total_train

    train_loss_list.append(train_loss_epoch)
    train_acc_list.append(train_acc_epoch)

    # 计算测试集损失和准确率
    total_test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data
            test_images, test_labels = test_images.to(DEVICE), test_labels.to(DEVICE)
            test_outputs = net(test_images)
            test_loss = criterion(test_outputs, test_labels)
            total_test_loss += test_loss.item()
            _, test_predicted = torch.max(test_outputs.data, 1)
            total_test += test_labels.size(0)
            correct_test += (test_predicted == test_labels).sum().item()

    test_loss_epoch = total_test_loss / len(test_loader)
    test_acc_epoch = 100 * correct_test / total_test

    test_losses.append(test_loss_epoch)
    test_accuracies.append(test_acc_epoch)
    print(f"Epoch {epoch + 1}: Train Loss: {train_loss_epoch:.6f}, Train Accuracy: {train_acc_epoch:.2f}%")
    print(f"Epoch {epoch + 1}: Test Loss: {test_loss_epoch:.6f}, Test Accuracy: {test_acc_epoch:.2f}%")

print(f"Accuracy on the test set: {test_acc_epoch:.2f}%")
torch.save(net, r"Exp2/model/NewModel.pth")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.show()

