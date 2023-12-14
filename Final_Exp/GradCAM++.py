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
        alpha_denom = grads.pow(2).mul(2) + self.activations[-1].pow(3).mul(grads.pow(3))
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
        one_hot = one_hot.to(input_tensor.device)
        
        # 反向传播
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)

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
    
def apply_gradcampp(image_path, model, target_layer, target_class):
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(image_path)

    # 实例化 GradCAMpp
    grad_cam_pp = GradCAMpp(model=model, target_layer=target_layer)

    # 使用 GradCAMpp
    saliency_map = grad_cam_pp.forward(input_tensor, class_idx=target_class)
    heatmap = grad_cam_pp.generate_heatmap(saliency_map)

    # 显示 GradCAM++ 的结果
    plt.imshow(rgb_img, alpha=0.5)
    plt.imshow(heatmap.squeeze(), cmap='hot', alpha=0.5)
    plt.show()

# 使用 GradCAM++ 可视化
image_path = r"Exp4\data4\both.jpg"
target_layer = model.features[10]  # 使用第十层 (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
target_class = 0  # 目标类别为狗

apply_gradcampp(image_path, model, target_layer, 0)
apply_gradcampp(image_path, model, target_layer, 1)
