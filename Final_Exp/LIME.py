import torch
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries

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
    image = Image.open(img_path).convert("RGB")  # 确保图像是 RGB 格式
    image = transform(image).unsqueeze(0)
    return image

def apply_lime(image_path, model, num_samples=1000):
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]  # 读取图像时确保是 RGB 格式
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(image_path)

    # 定义LIME解释器
    explainer = lime_image.LimeImageExplainer()

    # 生成解释
    explanation = explainer.explain_instance(rgb_img,  # 传递 RGB 图像
                                             model,  # 直接传递模型对象
                                             top_labels=1,
                                             num_samples=num_samples)

    # 获取解释后的图像
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    img_boundry = mark_boundaries(temp / 2 + 0.5, mask)

    # 显示 LIME 的结果
    plt.imshow(img_boundry)
    plt.show()

# 使用 LIME 进行解释
image_path = r"Exp4\data4\both.jpg"
apply_lime(image_path, model)
