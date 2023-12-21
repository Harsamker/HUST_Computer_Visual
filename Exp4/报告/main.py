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

def get_activate_hook(module, input, output):
    target_layer.activations = output

def get_grad_hook(module, input, output):
    target_layer.gradients = output[0]
    
def layercam(image_path, model, target_layer, target_class):

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
    gradients = target_layer.gradients 
    activations = target_layer.activations 
    grad_activations = gradients * activations
    cam = torch.relu(torch.sum(grad_activations, dim=1)).squeeze(0)
    cam = cam.cpu().detach().numpy()
    cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    cam = np.maximum(cam, 0)
    cam_normalized = cam / np.max(cam)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_normalized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    img = heatmap_colored * 0.4 + original_image * 0.6
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img



def gradcam(image_path, model, target_layer, target_class):
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
    gradients = target_layer.gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = target_layer.activations
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    img = heatmap_colored * 0.4 + original_image * 0.6
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# 加载模型
model = torch.load("Exp4/torch_alex.pth")
model.eval()
target_layer = model.features[10] 
target_layer.activations = None
target_layer.gradients = None
target_layer.register_forward_hook(get_activate_hook)
target_layer.register_backward_hook(get_grad_hook)
image_paths = ["Exp4/data4/dog.jpg", "Exp4/data4/cat.jpg", "Exp4/data4/both.jpg"]
target_class_list = [0, 1]
type_map = {0: "Cat", 1: "Dog"}


num_rows = len(image_paths)
num_cols = len(target_class_list) * 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

for i, img_path in enumerate(image_paths):
    for j, target_class in enumerate(target_class_list):
        grad_cam_image = gradcam(img_path, model, target_layer, target_class)
        layer_cam_image = layercam(img_path, model, target_layer, target_class)

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
