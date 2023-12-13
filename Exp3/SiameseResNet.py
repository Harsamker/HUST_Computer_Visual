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

BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8
LearnRate = 0.01
EPOCH = 30


class SiameseResNet(ResNet):
    def __init__(self, block, layers, num_classes=1):
        super(SiameseResNet, self).__init__(block, layers)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class SiameseMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset, split="train", transform=None):
        self.mnist_dataset = mnist_dataset
        self.split = split
        self.transform = transform
        self.labels = self.mnist_dataset.targets
        self.indices = list(range(len(self.mnist_dataset)))
        random.seed(42)
        random.shuffle(self.indices)
        split_size = int(0.1 * len(self.indices))
        if split == "train":
            self.indices = self.indices[:-split_size]
        elif split == "test":
            self.indices = self.indices[-split_size:]

    def __getitem__(self, index):
        img1, label1 = self.mnist_dataset[self.indices[index]]
        is_same_class = random.choice([True, False])
        if is_same_class:
            # Choose another image from the same class
            indices = [i for i, label in enumerate(self.labels) if label == label1]
            index2 = random.choice(indices)
        else:
            # Choose another image from a different class
            indices = [i for i, label in enumerate(self.labels) if label != label1]
            index2 = random.choice(indices)

        img2, label2 = self.mnist_dataset[index2]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, int(is_same_class)

    def __len__(self):
        return len(self.indices)


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

mnist_train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
mnist_test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

siamese_train_dataset = SiameseMNISTDataset(
    mnist_train_dataset, split="train", transform=transform
)
siamese_test_dataset = SiameseMNISTDataset(
    mnist_test_dataset, split="test", transform=transform
)

siamese_train_loader = DataLoader(
    siamese_train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
siamese_test_loader = DataLoader(
    siamese_test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

siameseresnet = SiameseResNet(BasicBlock, [1, 1, 1, 1], num_classes=1).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(siameseresnet.parameters(), lr=LearnRate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练过程
for epoch in range(EPOCH):
    process_bar = tqdm(
        enumerate(siamese_train_loader, 0),
        desc=f"Epoch {epoch + 1}",
        unit="batch",
        total=len(siamese_train_loader),
    )
    for i, data in process_bar:
        siameseresnet.train()
        img1, img2, target = data
        img1, img2, target = img1.to(DEVICE), img2.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = siameseresnet(img1, img2)
        loss = criterion(output, target.float().view(-1, 1))
        loss.backward()
        optimizer.step()
    scheduler.step()

# 测试过程
siameseresnet.eval()
correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(siamese_test_loader):
        img1, img2, target = data
        img1, img2, target = img1.to(DEVICE), img2.to(DEVICE), target.to(DEVICE)
        output = siameseresnet(img1, img2)
        predictions = (output > 0.5).float()
        correct += (predictions == target.float().view(-1, 1)).sum().item()
        total += target.size(0)

accuracy = correct / total * 100
print(f"Test Accuracy: {accuracy:.2f}%")
