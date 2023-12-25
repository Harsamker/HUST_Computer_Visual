import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(img_path)
    image = transform(image).unsqueeze(0)
    return image

def get_activations_hook(module, input, output):
    target_layer.activations = output

def visualize_feature_maps(image_path, model, target_layer):
    input_tensor = preprocess_image(image_path)
    model(input_tensor)
    feature_maps = target_layer.activations.cpu().detach().numpy()[0]
    # 256 = 16 * 16
    num_cols = 16
    num_rows = 16
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8), squeeze=False)
    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            ax = axes[i, j]
            ax.imshow(feature_maps[idx], cmap='gray')
            ax.axis('off')
    plt.suptitle(image_path+" each Conv2D Channels Outputs")
    plt.subplots_adjust(wspace=0.1, hspace=0.01)
    plt.show()

model = torch.load("Exp4/torch_alex.pth")
model.eval()
target_layer = model.features[10]
target_layer.register_forward_hook(get_activations_hook)
image_paths = ["Exp4/data4/dog.jpg", "Exp4/data4/cat.jpg", "Exp4/data4/both.jpg"]
for img_path in image_paths:
    visualize_feature_maps(img_path, model, target_layer)
