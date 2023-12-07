import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import os

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

PYDEVD_DISABLE_FILE_VALIDATION = 1
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock


"""
train.csv 标签+数据
test.csv 数据，标签输出到sample_submisson.csv
"""
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
EPOCHS = 2
BATCH_SIZE = 256
NUM_WORKER = 4
learning_rate = 0.01
momentum = 0.5
log_interval = 10

csv_reader = pd.read_csv(r"Exp2/data/train.csv").values.astype("float32")
test_reader, train_reader = train_test_split(csv_reader, train_size=0.2, random_state=1)


class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = torch.tensor(
            data[:, 1:].reshape(-1, 1, 28, 28), dtype=torch.float32
        )
        self.labels = torch.tensor(data[:, 0], dtype=torch.long)
        self.transform = transform

        # 计算均值和标准差
        self.mean, self.std = self.calculate_mean_std()

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

    def calculate_mean_std(self):
        num_of_pixels = self.data.numel()
        total_sum = self.data.sum()
        mean = total_sum / num_of_pixels

        sum_of_squared_error = ((self.data - mean).pow(2)).sum()
        std = torch.sqrt(sum_of_squared_error / num_of_pixels)

        return mean, std

    def normalize(self, x):
        return (x - self.mean) / self.std

    def transform(self, x):
        return self.normalize(x)


# 使用TensorDataset
train_dataset = torch.utils.data.TensorDataset(
    train_reader[:, 1:].reshape(-1, 1, 28, 28), train_reader[:, 0]
)
test_dataset = torch.utils.data.TensorDataset(
    test_reader[:, 1:].reshape(-1, 1, 28, 28), test_reader[:, 0]
)

# DataLoader优化
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKER,
    pin_memory=True,
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKER, pin_memory=True
)


class ResNetModel(ResNet):
    def __init__(self):
        super(ResNetModel, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Linear(512, 10)

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


if __name__ == "__main__":
    net = ResNetModel()
    lossF = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    # 存储训练过程
    history = {
        "Train Loss": [],
        "Train Accuracy": [],
        "Test Loss": [],
        "Test Accuracy": [],
    }
    for epoch in range(1, EPOCHS + 1):
        processBar = tqdm(train_loader, unit="step")
        net.train(True)
        for step, (trainImgs, labels) in enumerate(processBar):
            trainImgs = trainImgs.to(DEVICE)
            labels = labels.to(DEVICE)

            net.zero_grad()
            outputs = net(trainImgs)
            loss = lossF(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = torch.sum(predictions == labels) / labels.shape[0]
            loss.backward()

            optimizer.step()
            processBar.set_description(
                "[%d/%d] Train_Loss: %.4f, Train_Acc: %.2f"
                % (epoch, EPOCHS, loss.item(), accuracy.item() * 100)
            )

            if step == len(processBar) - 1:
                correct, totalLoss = 0, 0
                net.train(False)
                for testImgs, labels in test_loader:
                    testImgs = testImgs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = net(testImgs)
                    loss = lossF(outputs, labels)
                    predictions = torch.argmax(outputs, dim=1)

                    totalLoss += loss
                    correct += torch.sum(predictions == labels)
                testAccuracy = correct / (BATCH_SIZE * len(test_loader))
                testLoss = totalLoss / len(test_loader)
                history["Test Loss"].append(testLoss.item())
                history["Test Accuracy"].append(testAccuracy.item())
                history["Train Loss"].append(loss.item())
                history["Train Accuracy"].append(accuracy.item())
                processBar.set_description(
                    "[%d/%d] Train_Loss: %.4f, Train Acc: %.2f, Test Loss: %.4f, Test Acc: %.2f"
                    % (
                        epoch,
                        EPOCHS,
                        loss.item(),
                        accuracy.item() * 100,
                        testLoss.item(),
                        testAccuracy.item() * 100,
                    )
                )
        processBar.close()
    torch.save(net, r"Exp2/model/model.pth")
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["Train Loss"], label="Train Loss")
    plt.plot(history["Test Loss"], label="Test Loss")
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.xticks(np.arange(1, EPOCHS + 1, 1))  # 显示整数Epoch

    plt.subplot(1, 2, 2)
    plt.plot(history["Train Accuracy"], label="Train Accuracy")
    plt.plot(history["Test Accuracy"], label="Test Accuracy", color="red")
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.xticks(np.arange(1, EPOCHS + 1, 1))  # 显示整数Epoch

    plt.tight_layout()
    plt.show()
