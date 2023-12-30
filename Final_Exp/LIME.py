import torch
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.segmentation import slic, quickshift
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

device = "cpu"


# 预处理图像的函数
def preprocess_image(img_np):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    img_tensor = transform(img_pil).unsqueeze(0)
    return img_tensor


# LIME解释方法
def lime_explain(image_path, model, target_class, num_samples=1000, num_segments=200):
    # 读取并预处理图像
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # 取消归一化
    img = np.float32(cv2.resize(original_image, (224, 224)))
    # 两种不同的生成超像素的策略
    superpixels = quickshift(img, kernel_size=2, max_dist=150, ratio=0.1)
    # superpixels = quickshift(img, kernel_size=4, max_dist=150, ratio=0.1)
    # superpixels = slic(img, n_segments=num_segments, compactness=10, sigma=1)
    num_superpixels = np.unique(superpixels).shape[0]

    # 生成扰动图像
    perturbations = np.random.binomial(1, 0.5, size=(num_samples, num_superpixels))
    preds = []
    for pert in tqdm(perturbations, desc="Generating perturbations"):
        perturbed_img = img.copy()
        for i in range(num_superpixels):
            if pert[i] == 0 and np.any(superpixels == i):
                perturbed_img[superpixels == i] = np.mean(img[superpixels == i], axis=0)
        perturbed_tensor = preprocess_image(perturbed_img).to(device)
        model.zero_grad()
        output = model(perturbed_tensor)
        preds.append(output.cpu().detach().numpy()[0, target_class])

    # 训练线性模型
    preds = np.array(preds)
    model_linear = LinearRegression().fit(perturbations, preds)

    # 获取权重并映射到超像素
    coef = model_linear.coef_
    explanation = np.zeros(superpixels.shape)
    for i in range(num_superpixels):
        if np.any(superpixels == i):
            explanation[superpixels == i] = coef[i]

    # 归一化
    max_value = explanation.max()
    min_value = explanation.min()
    if max_value != min_value:
        explanation = (explanation - min_value) / (max_value - min_value)
    else:
        explanation.fill(0)  # 如果最大值等于最小值，则将所有值设置为0.5，或者根据实际情况选择一个合适的默认值

    # 创建热图
    heatmap = np.uint8(255 * explanation)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 叠加热图到原始图像
    superimposed_img = heatmap * 0.4 + original_image * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img


# 加载模型
model = torch.load("Exp4/torch_alex.pth")
model.eval()

# 类别映射
typemap = {0: "Cat", 1: "Dog"}

# 图像路径
image_paths = ["Exp4/data4/dog.jpg", "Exp4/data4/cat.jpg", "Exp4/data4/both.jpg"]

# 可视化LIME解释
fig, axes = plt.subplots(len(image_paths), len(typemap), figsize=(10, 15))  # 3行2列的子图
numsample = 2000
for i, img_path in enumerate(image_paths):
    for j, target_class in enumerate(typemap):
        lime_image = lime_explain(img_path, model, target_class, num_samples=numsample)
        axes[i, j].imshow(lime_image)
        axes[i, j].axis("off")
        axes[i, j].set_title(
            f"{image_paths[i].split('/')[-1]}: {typemap[target_class]}"
        )
fig.suptitle("LIME Explanation", fontsize=16)
plt.tight_layout()
plt.tight_layout()
plt.show()
