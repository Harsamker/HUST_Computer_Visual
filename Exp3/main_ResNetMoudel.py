import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn.functional as F
from functools import partial
import torch.optim.lr_scheduler as lr_scheduler

BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8
LearnRate = 0.01
EPOCH = 30


class ResNetModel(ResNet):
    def __init__(self, block, layers, num_classes=10):
        super(ResNetModel, self).__init__(block, layers)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


ResNet18 = ResNetModel(BasicBlock, [1, 1, 1, 1], num_classes=10).to(DEVICE)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)
test_loss_list = []
test_acc_list = []
train_loss_list = []
train_acc_list = []

if __name__ == "__main__":
    # 用BasicBlock训练
    ResNet18 = ResNetModel(BasicBlock, [1, 1, 1, 1], num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ResNet18.parameters(), lr=LearnRate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(EPOCH):
        process_bar = tqdm(
            enumerate(train_loader, 0),
            desc=f"Epoch {epoch + 1}",
            unit="batch",
            total=len(train_loader),
        )
        for i, data in process_bar:
            ResNet18.train()
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = ResNet18(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()  # 在每个epoch结束时调整学习率
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for train_data in train_loader:
                train_images, train_labels = train_data
                train_images, train_labels = train_images.to(DEVICE), train_labels.to(
                    DEVICE
                )
                train_outputs = ResNet18(train_images)
                train_loss = criterion(train_outputs, train_labels)
                total_train_loss += train_loss.item()
                _, train_predicted = torch.max(train_outputs.data, 1)
                total_train += train_labels.size(0)
                correct_train += (train_predicted == train_labels).sum().item()

        train_loss_epoch = total_train_loss / len(train_loader)
        train_acc_epoch = 100 * correct_train / total_train

        train_loss_list.append(train_loss_epoch)
        train_acc_list.append(train_acc_epoch)

        total_test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for test_data in test_loader:
                test_images, test_labels = test_data
                test_images, test_labels = test_images.to(DEVICE), test_labels.to(
                    DEVICE
                )
                test_outputs = ResNet18(test_images)
                test_loss = criterion(test_outputs, test_labels)
                total_test_loss += test_loss.item()
                _, test_predicted = torch.max(test_outputs.data, 1)
                total_test += test_labels.size(0)
                correct_test += (test_predicted == test_labels).sum().item()

        test_loss_epoch = total_test_loss / len(test_loader)
        test_acc_epoch = 100 * correct_test / total_test

        test_loss_list.append(test_loss_epoch)
        test_acc_list.append(test_acc_epoch)
        print(
            f"Epoch {epoch + 1}: Train Loss: {train_loss_epoch:.6f}, Train Accuracy: {train_acc_epoch:.2f}%"
        )
        print(
            f"Epoch {epoch + 1}: Test Loss: {test_loss_epoch:.6f}, Test Accuracy: {test_acc_epoch:.2f}%"
        )
    print(f"Final Test Acc: {test_acc_epoch:.2f}%")
    print(f"Final Train Acc: {train_acc_epoch:.2f}%")
    torch.save(ResNet18, r"Exp3/model/NewModel.pth")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(test_loss_list, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label="Train Accuracy")
    plt.plot(test_acc_list, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.show()
