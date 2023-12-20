import torch
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F
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

def apply_gradcam_plusplus(image_path, model, target_layer, target_class):
    # 读取图像并进行预处理
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    original_image = np.copy(rgb_img)
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(image_path)

    # 前向传播
    model.zero_grad()
    output = model(input_tensor)
    score = output[:, target_class]

    # 反向传播
    score.backward(retain_graph=True)

    # 获取目标层的激活和梯度
    gradients = target_layer.gradients  # 梯度
    activations = target_layer.activations  # 激活
    b, k, u, v = gradients.size()

    alpha_numer = gradients.pow(2)
    # 对梯度进行三次方并与激活相乘，然后在通道上求和，避免维度不匹配
    alpha_denom = gradients.pow(2).mul(2) + \
                  activations.mul(gradients.pow(3)).sum(axis=1, keepdim=True)
    # 确保分母不为零
    alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
    alphas = alpha_numer.div(alpha_denom + 1e-7)

    # ReLU操作以确保梯度为正
    positive_gradients = F.relu(score.exp() * gradients)
    # 在通道上进行全局平均池化，得到权重
    weights = (alphas * positive_gradients).sum(axis=(2, 3), keepdim=True)

    # 生成GradCAM++图
    gradcam_plusplus = (weights * activations).sum(dim=1, keepdim=True)
    gradcam_plusplus = F.relu(gradcam_plusplus)
    gradcam_plusplus = F.interpolate(gradcam_plusplus, (224, 224), mode='bilinear', align_corners=False)

    # 归一化并去掉梯度计算，然后转换为numpy格式进行可视化
    gradcam_plusplus = gradcam_plusplus / torch.max(gradcam_plusplus)
    gradcam_plusplus = gradcam_plusplus.detach().cpu().squeeze().numpy()  # 修改这一行

    # 创建热图
    heatmap = np.uint8(255 * gradcam_plusplus)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 叠加热图到原始图像
    superimposed_img = heatmap * 0.4 + original_image * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img



def apply_scorecam(image_path, model, target_layer, target_class):
    # 预处理图像
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)
    original_image = np.copy(img)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input_tensor = preprocess_image(image_path)

    # 获取激活图
    model.zero_grad()
    output = model(input_tensor)
    score = output[0][target_class]
    target_layer.activations = None
    target_layer.register_forward_hook(get_activations_hook)
    output = model(input_tensor)
    activations = target_layer.activations.squeeze().cpu().detach().numpy()
    b, k, u, v = activations.shape

    # 生成Score-CAM图
    score_cam = np.zeros(target_layer.activations.shape[2:], dtype=np.float32)
    for i in range(k):
        saliency_map = cv2.resize(activations[i], (224, 224))
        if np.max(saliency_map) - np.min(saliency_map) != 0:
            saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))
        input_tensor = preprocess_image(image_path)
        input_tensor[0, 0] += saliency_map
        input_tensor[0, 1] += saliency_map
        input_tensor[0, 2] += saliency_map
        output = model(input_tensor)
        score = output[0][target_class]
        score_cam += score.cpu().detach().numpy() * saliency_map

    # 归一化
    score_cam = np.maximum(score_cam, 0)
    score_cam = cv2.resize(score_cam, (224, 224))
    score_cam = score_cam - np.min(score_cam)
    score_cam = score_cam / np.max(score_cam)

    # 叠加热力图
    heatmap = np.uint8(255 * score_cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = heatmap * 0.4 + original_image * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img



#钩子函数
def get_activations_hook(module, input, output):
    target_layer.activations = output

def get_gradients_hook(module, input_grad, output_grad):
    target_layer.gradients = output_grad[0]

# 加载模型
model = torch.load("Exp4/torch_alex.pth")
model.eval()
target_layer = model.features[10] 
target_layer.activations = None
target_layer.gradients = None
target_layer.register_forward_hook(get_activations_hook)
target_layer.register_backward_hook(get_gradients_hook)
image_paths = ["Exp4/data4/dog.jpg", "Exp4/data4/cat.jpg", "Exp4/data4/both.jpg"]
target_class_list = [0, 1]
type_map = {0: "Cat", 1: "Dog"}

# 可视化
num_rows = len(image_paths)
num_cols = len(target_class_list) * 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

for i, img_path in enumerate(image_paths):
    for j, target_class in enumerate(target_class_list):
        grad_campp_image = apply_gradcam_plusplus(img_path, model, target_layer, target_class)
        score_cam_image = apply_scorecam(img_path, model, target_layer, target_class)

        # 显示 Grad-CAM
        axes[i, j * 2].imshow(grad_campp_image)
        axes[i, j * 2].set_title(f"IMG{i} Grad-CAM++ ({type_map[target_class]})")
        axes[i, j * 2].axis("off")

        # 显示 Layer-CAM
        axes[i, j * 2 + 1].imshow(score_cam_image)
        axes[i, j * 2 + 1].set_title(f"IMG{i} ScoreCAM ({type_map[target_class]})")
        axes[i, j * 2 + 1].axis("off")

plt.tight_layout()
plt.show()
