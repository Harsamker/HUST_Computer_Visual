import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn.functional as F
import torch.nn as nn


BATCH_SIZE = 256
NUM_WORKER = 4
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# 定义你的模型类
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


# 读取测试集数据
test_csv_path = r"Exp2/data/test.csv"
test_data = pd.read_csv(test_csv_path).values.astype("float32")

# 创建测试集的 Dataset 和 DataLoader
test_dataset = MyDataset(test_data)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKER, pin_memory=True
)

# 加载已经训练好的模型
net = ResNetModel()
net.load_state_dict(torch.load(r"Exp2/model/model.pth"))  # 请替换为你实际保存的模型文件

# 准备模型进行测试
net.eval()

# 存储测试集的预测标签
test_predictions = []

# 开始模型的测试阶段
with torch.no_grad():
    for testImgs, _ in test_loader:
        testImgs = testImgs.to(DEVICE)
        outputs = net(testImgs)
        predictions = torch.argmax(outputs, dim=1)
        test_predictions.extend(predictions.cpu().numpy())

# 将预测结果输出到 CSV 文件（predict.csv）
predict_df = pd.DataFrame(
    {"ImageId": range(1, len(test_predictions) + 1), "Label": test_predictions}
)
predict_df.to_csv(r"Exp2/data/predict.csv", index=False)
