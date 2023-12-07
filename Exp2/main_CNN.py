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

"""
train.csv 标签+数据
test.csv 数据，标签输出到sample_submisson.csv
"""
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 256
NUM_WORKER = 4
learning_rate = 0.01
momentum = 0.5
log_interval = 10

csv_reader = pd.read_csv(r"data/train.csv").values.astype("float32")
test_reader, train_reader = train_test_split(csv_reader, train_size=0.2, random_state=1)


def normalcalcu(data):
    num_of_pixels = len(data) * 28 * 28
    total_sum = 0
    for batch in data:
        total_sum += batch.sum()
    mean = total_sum / num_of_pixels
    sum_of_squared_error = 0
    for batch in data:
        sum_of_squared_error += ((batch - mean).pow(2)).sum()
    std = torch.sqrt(sum_of_squared_error / num_of_pixels)
    return (mean, std)


class MyDataset(Dataset):
    """
    读取数据、初始化数据
    """

    def __init__(self, data, transform=None):
        # plt.imshow(train_data[1000])
        # plt.show()
        (data_set, data_labels) = self.load_data(data)
        self.data_set = data_set
        self.data_labels = data_labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data_set[index], int(self.data_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data_set)

    """
    load_data也是我们自定义的函数，用途：显示数据集中的数据图片数据+标签label
    """

    def load_data(self, data):
        data_data = data[:, 1:]
        data_label = data[:, 0]
        data_data = data[:, 1:]
        data_label = data[:, 0]
        data_data = torch.tensor(data_data).reshape(len(data_data), 1, 28, 28)
        data_label = torch.tensor(data_label)
        (mean, std) = normalcalcu(data_data)
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.Normalize(mean=mean, std=std)]
        )
        data_data = transform(data_data)
        return (data_data, data_label)


train_dataset = MyDataset(train_reader)
train_loader = Data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER
)
test_dataset = MyDataset(test_reader)
test_loader = Data.DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKER
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    # https://blog.csdn.net/peacefairy/article/details/108020179 归一化
    net = Net()
    lossF = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    # 存储训练过程
    history = {"Test Loss": [], "Test Accuracy": []}
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
                "[%d/%d] Loss: %.4f, Acc: %.4f"
                % (epoch, EPOCHS, loss.item(), accuracy.item())
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
                processBar.set_description(
                    "[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f"
                    % (
                        epoch,
                        EPOCHS,
                        loss.item(),
                        accuracy.item(),
                        testLoss.item(),
                        testAccuracy.item(),
                    )
                )
        processBar.close()
    matplotlib.pyplot.plot(history["Test Loss"], label="Test Loss")
    matplotlib.pyplot.legend(loc="best")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.xlabel("Epoch")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.show()

    # 对测试准确率进行可视化
    matplotlib.pyplot.plot(history["Test Accuracy"], color="red", label="Test Accuracy")
    matplotlib.pyplot.legend(loc="best")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.xlabel("Epoch")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.show()
