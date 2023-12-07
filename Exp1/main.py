import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

csv_reader = pd.read_csv(r"Exp1/dataset.csv")
train_acc_list = []
test_acc_list = []
for i in range(0, 3):
    csv_reader = csv_reader.sample(frac=1).reset_index(drop=True)
    train_data, test_data = train_test_split(csv_reader, test_size=0.1, random_state=42)
    train_feature = train_data.drop("label", axis=1).values
    train_label = train_data["label"].values
    test_feature = test_data.drop("label", axis=1).values
    test_label = test_data["label"].values
    scaler = StandardScaler()
    train_feature = scaler.fit_transform(train_feature)
    test_feature = scaler.transform(test_feature)
    train_feature = torch.FloatTensor(train_feature)
    train_label = torch.LongTensor(train_label - 1)  # 从1~4映射到0~3
    test_feature = torch.FloatTensor(test_feature)
    test_label = torch.LongTensor(test_label - 1)
    train_dataset = TensorDataset(train_feature, train_label)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 一次取32个
    model = nn.Sequential(
        nn.Linear(2, 128),  # 入口为特征数据维数
        nn.ReLU(),
        nn.Linear(128, 4),  # 出口为标签个数
    )

    learning_rate = 0.001
    epochs = 50

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loss_list = []
    test_loss_list = []

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_loss_list.append(loss.item())
        model.eval()
        with torch.no_grad():
            test_result = model(test_feature)
            test_loss = criterion(test_result, test_label)
            test_loss_list.append(test_loss.item())
        print(
            f"Epoch:{epoch + 1}/{epochs}, Training Loss: {loss.item()}, Test Loss: {test_loss.item()}"
        )

    model.eval()
    with torch.no_grad():
        train_outputs = model(train_feature)
        predicted_train_labels = torch.argmax(train_outputs, axis=1)

    train_acc = torch.sum(predicted_train_labels == train_label).item() / len(
        train_label
    )
    train_acc_list.append(train_acc)
    print(f"Train Accuracy: {train_acc * 100:.2f}%")

    model.eval()
    with torch.no_grad():
        test_result = model(test_feature)
        predicted_label = torch.argmax(test_result, axis=1)

    test_acc = torch.sum(predicted_label == test_label).item() / len(test_label)
    test_acc_list.append(test_acc)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    plt.plot(range(1, epochs + 1), train_loss_list, label="Training Loss")
    plt.plot(range(1, epochs + 1), test_loss_list, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

final_train_acc = sum(train_acc_list) / len(train_acc_list)
final_test_acc = sum(test_acc_list) / len(test_acc_list)
print(f"Final Train Accuracy: {final_train_acc * 100:.2f}%")
print(f"Final Test Accuracy: {final_test_acc * 100:.2f}%")
