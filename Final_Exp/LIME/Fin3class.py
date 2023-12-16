import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from sklearn.linear_model import LinearRegression
from skimage.segmentation import quickshift, mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
import concurrent.futures
from matplotlib.colors import ListedColormap

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(img):
    if isinstance(img, str):  
        img_pil = Image.open(img).convert('RGB')
    else:  
        img_pil = Image.fromarray((img * 255).astype(np.uint8)).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img_pil).to(device)
    return img_tensor.unsqueeze(0)

def create_perturbation(img_np, segments, num_segments, idx):
    perturbed_img = img_np.copy()
    mask = np.random.binomial(1, 0.5, num_segments)
    for (i, val) in enumerate(mask):
        if val == 0:
            perturbed_img[segments == i] = 0
    return perturbed_img

def create_perturbations_parallel(img_np, segments, num_segments, num_samples):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        perturbed_images = list(tqdm(executor.map(create_perturbation, [img_np]*num_samples, [segments]*num_samples, [num_segments]*num_samples, range(num_samples)),
                                     desc="Creating perturbations total{}", unit="image"))

    return np.array(perturbed_images)

# 在predict_perturbations函数中
def predict_perturbations(model, perturbed_images):
    preds = []
    for perturbed_img in tqdm(perturbed_images, desc="Predicting perturbations", unit="image"):
        perturbed_tensor = preprocess_image(perturbed_img)
        with torch.no_grad():
            preds.append(model(perturbed_tensor).cpu().numpy())
    preds_array = np.array(preds)
    print("Predictions shape:", preds_array.shape)
    return preds_array

def train_linear_model(segments, perturbed_images, preds):
    coefficients = np.zeros(segments.max() + 1)

    for i in tqdm(range(segments.max() + 1), desc="Training linear model", unit="segment"):
        mask = segments == i
        X = (perturbed_images[:, mask].mean(axis=(1, 2)) > 0).astype(int).reshape(-1, 1)

        if np.sum(X) > 0:
            Y = preds  # 直接使用 preds 数组
            lin_reg = LinearRegression().fit(X.reshape(-1, 1), Y)
            coefficients[i] = lin_reg.coef_[0]
        else:
            coefficients[i] = 0

    return coefficients


from keras.applications.inception_v3 import preprocess_input, decode_predictions

def main():
    img_path = "Final_Exp/image/car1.jpg"
    num_samples = 100

    model = inception_v3(pretrained=True).to(device)
    model.eval()

    # 预处理图像
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    img_tensor = preprocess_image(img_path)
    img_np_tensor = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

    # 进行预测并获取前三个类别
    preds = model(img_tensor).detach().cpu().numpy()
    top_preds = np.argsort(-preds[0])[:3]
    
    # 解码前三个预测类别
    decoded_preds = decode_predictions(preds)[0]
    top_labels = [label for _, label, _ in decoded_preds[:3]]

    segments = quickshift(img_np_tensor, kernel_size=4, max_dist=200, ratio=0.2)
    num_segments = np.unique(segments).shape[0]

    plt.figure(figsize=(12, 5))
    for i, label in enumerate(top_labels[:3]):
        subplot_index = i + 1
        target_class = top_preds[i]
        perturbed_images = create_perturbations_parallel(img_np_tensor, segments, num_segments, num_samples)
        preds = predict_perturbations(model, perturbed_images)
        target_preds = preds.squeeze(axis=1)[:, target_class]
        coefficients = train_linear_model(segments, perturbed_images, target_preds)

        visualize_lime(img_np_tensor, segments, coefficients, img_path, label, subplot_index)

    plt.show()
    
def visualize_lime(img_np, segments, coefficients, img_path, target_class, subplot_index):
    mask = np.zeros(segments.shape)
    for i in range(segments.max() + 1):
        if coefficients[i] > 0:
            mask[segments == i] = coefficients[i]

     # Convert the original image to uint8 type if necessary
    superimposed_img = np.uint8(img_np)

    # Ensure the heatmap is also uint8 type
    heatmap = np.uint8(255 * mask)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    transparent_heatmap = np.zeros((img_np.shape[0], img_np.shape[1], 4), dtype=np.uint8)
    transparent_heatmap[:, :, :3] = heatmap
    transparent_heatmap[:, :, 3] = (mask > 0) * 255  # Only important areas are non-transparent

    # Resize the heatmap to match the original image size
    transparent_heatmap_resized = cv2.resize(transparent_heatmap[:, :, :3], (superimposed_img.shape[1], superimposed_img.shape[0]))

    # Blend the heatmap with the original image
    alpha = 0.4
    overlay = cv2.addWeighted(superimposed_img, 1-alpha, transparent_heatmap_resized, alpha, 0)

    # 在子图上显示结果
    plt.subplot(1, 3, subplot_index)
    plt.imshow(overlay)
    plt.title(f"Class {target_class}")
    plt.axis('off')
    
if __name__ == "__main__":
    main()
