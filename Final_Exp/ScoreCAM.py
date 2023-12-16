import torch
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F

# 加载模型
model = torch.load(r"Exp4\torch_alex.pth")
model.eval()

def preprocess_image(img_path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(img_path)
    image = transform(image).unsqueeze(0)
    return image

class ScoreCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = []
        self.model.eval()

        # 注册钩子
        self.target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        self.activations.append(output.detach())

    def forward(self, input_tensor, class_idx=None):
        # 清除历史激活
        self.activations.clear()

        # 正向传播获取特征图
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # 获取目标层的激活
        activation = self.activations[-1]
        _, nc, h, w = activation.shape

        # 为每个通道生成一个分数
        scores = torch.zeros((nc, ), dtype=torch.float32).to(input_tensor.device)

        for i in range(nc):
            # 获取当前通道的特征图
            saliency_map = activation[:, i, :, :]

            # 归一化
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

            # 调整 saliency_map 的维度以符合 interpolate 函数的要求
            saliency_map = saliency_map.unsqueeze(0)

            # 将特征图调整为输入张量的大小
            saliency_map = F.interpolate(saliency_map, input_tensor.shape[2:], mode='bilinear', align_corners=False)

            # 移除增加的维度以匹配输入张量的维度
            saliency_map = saliency_map.squeeze(0)

            # 使用特征图作为掩码加权原始图像
            weighted_input = input_tensor * saliency_map

            # 重新进行前向传播
            score = self.model(weighted_input)
            scores[i] = score[0, class_idx]

        # 修改权重计算
        weights = scores / scores.sum()
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam

    def generate_heatmap(self, cam):
        cam = cam.cpu().detach().numpy()
        cam = np.uint8(255 * cam)
        cam = cam[0][0]
        return cam

def apply_scorecam(image_path, model, target_layer, target_class):
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(image_path)

    scorecam = ScoreCAM(model=model, target_layer=target_layer)
    saliency_map = scorecam.forward(input_tensor, class_idx=target_class)
    heatmap = scorecam.generate_heatmap(saliency_map)

    # 显示结果
    plt.imshow(rgb_img, alpha=0.5)
    plt.imshow(heatmap.squeeze(), cmap='hot', alpha=0.5)
    plt.show()

# 使用 ScoreCAM 进行可视化
image_path = r"Exp4\data4\both.jpg"
target_layer = model.features[10]  # 使用第十层 (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
target_class = 0  # 目标类别为狗

apply_scorecam(image_path, model, target_layer, 0)
apply_scorecam(image_path, model, target_layer, 1)
