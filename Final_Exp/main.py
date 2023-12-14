import torch
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

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

import torch
import torch.nn.functional as F
import numpy as np

class GradCAMpp:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = []
        self.activations = []

        # 注册钩子
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def _find_3rd_order_grad(self, grads):
        # 计算三阶梯度，用于GradCAM++的计算
        alpha_numer = grads.pow(2)
        alpha_denom = grads.pow(2).mul(2) + \
                      self.activations[-1].pow(2).mul(grads.pow(3))
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_numer.div(alpha_denom + 1e-7)
        positive_gradients = F.relu(grads)
        weights = (alpha * positive_gradients).sum(dim=(2, 3), keepdim=True)
        return weights

    def forward(self, input_tensor, class_idx=None):
        self.gradients.clear()
        self.activations.clear()

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = np.argmax(output.cpu().data.numpy())

        # 创建目标类别的one-hot编码
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][class_idx] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(input_tensor.device) * output)

        # 反向传播
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        # 计算权重
        weights = self._find_3rd_order_grad(self.gradients[-1])
        activation = self.activations[-1]
        b, k, u, v = activation.size()

        # 权重和特征图相乘，然后求和
        saliency_map = (weights*activation).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min + 1e-7)
        return saliency_map

    def generate_heatmap(self, saliency_map):
        # 将saliency_map转换为numpy数组，并可视化为热力图
        saliency_map = saliency_map.cpu().data.numpy()
        heatmap = np.uint8(255 * saliency_map[0][0])
        return heatmap
import torch
import torch.nn.functional as F

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

        # 生成加权的特征图
        weights = F.softmax(scores, dim=0).view(nc, 1, 1)
        cam = (weights * activation).sum(dim=0, keepdim=True)
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

    def _find_3rd_order_grad(self, grads):
        # 调整计算方式
        alpha_numer = grads.pow(2)
        alpha_denom = grads.pow(2) + self.activations[-1].pow(2).mul(grads.pow(3))
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_numer.div(alpha_denom + 1e-7)
        positive_gradients = F.relu(grads)
        weights = (alpha * positive_gradients).sum(dim=(2, 3), keepdim=True)
        return weights

def apply_gradcampp(image_path, model, target_layer, target_class):
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(image_path)

    # 实例化 GradCAMpp
    grad_cam_pp = GradCAMpp(model=model, target_layer=target_layer)

    # 使用 GradCAMpp
    saliency_map = grad_cam_pp.forward(input_tensor, class_idx=target_class)
    heatmap = grad_cam_pp.generate_heatmap(saliency_map)

    # 返回 GradCAM++ 的结果
    return rgb_img, heatmap.squeeze()


image_paths = [r"Exp4\data4\dog.jpg", r"Exp4\data4\cat.jpg", r"Exp4\data4\both.jpg"]
target_layer = model.features[10]  # 使用第十层 (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
target_class_list = [0, 1]
type_map = {0: "Cat", 1: "Dog"}

num_rows = len(image_paths)
num_cols = len(target_class_list) + 1
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

for i, img_path in enumerate(image_paths):
    # 显示原图
    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    axes[i, 0].imshow(rgb_img)
    axes[i, 0].set_title(f"IMG{i} Original")
    axes[i, 0].axis("off")

    for j, target_class in enumerate(target_class_list):
        rgb_img, grad_cam_pp_image = apply_gradcampp(img_path, model, target_layer, target_class)

        # 显示 GradCAM++ 的结果
        axes[i, j + 1].imshow(rgb_img, alpha=0.5)  # 显示原图作为背景
        axes[i, j + 1].imshow(grad_cam_pp_image, cmap='hot', alpha=0.5)  # 叠加 GradCAM++ 热图
        axes[i, j + 1].set_title(f"IMG{i} GradCAM++ ({type_map[target_class]})")
        axes[i, j + 1].axis("off")

plt.tight_layout()
plt.show()


