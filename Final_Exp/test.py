import os
import numpy as np
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F

# 加载模型
model = torch.load(r"Exp4\torch_alex.pth")
model.eval()

# 图像预处理函数
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.fromarray(img).convert('RGB')
    image = transform(image).unsqueeze(0)  # 正确地添加批处理维度
    return image
def batch_predict(imgs):
    model.eval()
    batch = torch.stack([preprocess_image(img) for img in imgs], dim=0)
    batch = batch.squeeze(1)  # 确保批处理张量是四维的
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    with torch.no_grad():
        logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


# 选择一个图像进行解释
image_paths = [r"Exp4\data4\dog.jpg", r"Exp4\data4\cat.jpg", r"Exp4\data4\both.jpg"]
img_path = image_paths[0]  # 选择第一个图像
original_image = Image.open(img_path).convert('RGB')
img = np.array(original_image)
# 使用 LIME 进行图像解释
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    np.array(original_image),  # 使用原始图像的 NumPy 数组
    batch_predict,  # 直接传递 batch_predict 函数
    top_labels=5, 
    num_samples=1000  # 图像样本数量
)

# 可视化 LIME 解释
fig, axes = plt.subplots(1, len(explanation.top_labels), figsize=(15, 5))
axes[0].imshow(original_image)
axes[0].set_title('Original Image')
for i, label_id in enumerate(explanation.top_labels):
    temp, mask = explanation.get_image_and_mask(label_id, positive_only=True, hide_rest=False)
    axes[i + 1].imshow(mark_boundaries(temp / 2 + 0.5, mask))
    axes[i + 1].set_title(f'LIME Explanation: {label_id}')
plt.show()
