import torch
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 预处理图像的函数
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
def apply_custom_layer_cam(image_path, model, target_layer, target_class):
    # 读取图像并进行预处理
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    original_image = np.copy(rgb_img)
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(image_path)

    # 前向传播
    model.zero_grad()
    output = model(input_tensor)
    class_activation = output[:, target_class]

    # 反向传播
    class_activation.backward()

    # 获取类激活对特征图的梯度
    gradients = target_layer.gradients  # 假设梯度已经从钩子函数中获取
    activations = target_layer.activations  # 假设激活已经从钩子函数中获取

    # 逐元素乘积
    grad_activations = gradients * activations

    # 将逐元素乘积后的结果进行ReLU操作，并在通道维度上求和来生成热图
    cam = torch.relu(torch.sum(grad_activations, dim=1)).squeeze(0)

    # 将热图转换为numpy格式并进行归一化处理
    cam = cam.cpu().detach().numpy()
    cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    cam = np.maximum(cam, 0)  # ReLU
    cam_normalized = cam / np.max(cam)  # 归一化

    # 创建彩色热图
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_normalized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # 将彩色热图叠加在原始图像上
    superimposed_img = heatmap_colored * 0.4 + original_image * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img



def apply_custom_grad_cam(image_path, model, target_layer, target_class):
    # 读取图像并进行预处理
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    original_image = np.copy(rgb_img)
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(image_path)

    # 前向传播
    model.zero_grad()
    output = model(input_tensor)
    class_activation = output[:, target_class]

    # 反向传播
    class_activation.backward()

    # 获取类激活对特征图的梯度
    gradients = target_layer.gradients  # 假设梯度已经从钩子函数中获取

    # 沿通道方向池化梯度
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # 获取目标层的激活
    activations = target_layer.activations  # 假设激活已经从钩子函数中获取

    # 将通道按梯度加权
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    # 对激活通道进行平均
    heatmap = torch.mean(activations, dim=1).squeeze()
    # 对热图进行 ReLU
    heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
    # 对热图进行归一化
    heatmap /= np.max(heatmap)
    # 从热图创建图像
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # 使用红色表示高热度，蓝色表示低热度的彩色热图
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # 将彩色热图叠加在原始图像上
    superimposed_img = heatmap_colored * 0.4 + original_image * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img



# Hooks to capture the gradients and activations
def get_activations_hook(module, input, output):
    target_layer.activations = output

def get_gradients_hook(module, input_grad, output_grad):
    target_layer.gradients = output_grad[0]

# 加载模型
model = torch.load("Exp4/torch_alex.pth")
model.eval()

# Add hooks to the target layer
target_layer = model.features[10]  # Selecting the 10th layer as target layer
target_layer.activations = None
target_layer.gradients = None
target_layer.register_forward_hook(get_activations_hook)
target_layer.register_backward_hook(get_gradients_hook)

# 图像路径和类别设置
image_paths = ["Exp4/data4/dog.jpg", "Exp4/data4/cat.jpg", "Exp4/data4/both.jpg"]
target_class_list = [0, 1]
type_map = {0: "Cat", 1: "Dog"}

# 可视化
num_rows = len(image_paths)
num_cols = len(target_class_list) * 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

for i, img_path in enumerate(image_paths):
    for j, target_class in enumerate(target_class_list):
        grad_cam_image = apply_custom_grad_cam(img_path, model, target_layer, target_class)
        layer_cam_image = apply_custom_layer_cam(img_path, model, target_layer, target_class)  # 这个函数需要你根据LayerCAM的用法来实现

        # 显示 Grad-CAM
        axes[i, j * 2].imshow(grad_cam_image)
        axes[i, j * 2].set_title(f"IMG{i} Grad-CAM ({type_map[target_class]})")
        axes[i, j * 2].axis("off")

        # 显示 Layer-CAM
        axes[i, j * 2 + 1].imshow(layer_cam_image)
        axes[i, j * 2 + 1].set_title(f"IMG{i} Layer-CAM ({type_map[target_class]})")
        axes[i, j * 2 + 1].axis("off")

plt.tight_layout()
plt.show()
