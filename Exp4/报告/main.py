import torch
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 加载模型
model = torch.load(r"Exp4\torch_alex.pth")
print(model)
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


def apply_cam(image_path, model, target_layer, target_class):
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(image_path)
    grad_cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=False)
    layer_cam = LayerCAM(model=model, target_layers=[target_layer], use_cuda=False)
    target_category = lambda x: x[target_class]

    # Grad-CAM
    grayscale_cam_grad = grad_cam(input_tensor=input_tensor, targets=[target_category])
    grad_cam_image = show_cam_on_image(rgb_img, grayscale_cam_grad[0, :], use_rgb=True)

    # Layer-CAM
    grayscale_cam_layer = layer_cam(
        input_tensor=input_tensor, targets=[target_category]
    )
    layer_cam_image = show_cam_on_image(
        rgb_img, grayscale_cam_layer[0, :], use_rgb=True
    )

    return grad_cam_image, layer_cam_image


image_paths = [r"Exp4\data4\dog.jpg", r"Exp4\data4\cat.jpg", r"Exp4\data4\both.jpg"]
target_layer = model.features[
    10
]  # 第十层 (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
target_class_list = [0, 1]
type_map = {0: "Cat", 1: "Dog"}


num_rows = len(image_paths)
num_cols = len(target_class_list) * 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

for i, img_path in enumerate(image_paths):
    for j, target_class in enumerate(target_class_list):
        grad_cam_image, layer_cam_image = apply_cam(
            img_path, model, target_layer, target_class
        )
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
