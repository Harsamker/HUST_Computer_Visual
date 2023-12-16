import torch
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 加载模型
model = torch.load(r"Exp4\torch_alex.pth")
model.eval()

# 图像预处理
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(img_path)
    image = transform(image).unsqueeze(0)
    return image

def apply_scorecam(image_path, model, target_layer, target_class):
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    input_tensor = preprocess_image(image_path)

    # 获取目标层
    target_layers = [target_layer]

    # 设置前向钩子修改模型输出
    def forward_hook(module, input, output):
        return torch.softmax(output, dim=1)[:, target_class].unsqueeze(0)

    hook_handle = model.fc.register_forward_hook(forward_hook)

    # 实例化 ScoreCAM
    cam = ScoreCAM(model=model, target_layers=target_layers)

    # 获取 ScoreCAM 热图
    grayscale_cam = cam(input_tensor=input_tensor)

    # 移除钩子
    hook_handle.remove()

    # 叠加热图到原图
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.axis('off')
    plt.show()

# 使用 ScoreCAM 进行可视化
image_path = r"Exp4\data4\both.jpg"
target_layer = model.features[10]  # 获取第十层 Conv2d 作为目标层

target_class = 0  # 目标类别

apply_scorecam(image_path, model, target_layer, target_class)
