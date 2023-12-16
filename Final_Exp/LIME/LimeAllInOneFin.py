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
'''
def create_perturbation(img_np, segments, num_segments, idx):
    perturbed_img = img_np.copy()
    mask = np.random.binomial(1, 0.5, num_segments)
    for (i, val) in enumerate(mask):
        if val == 0:
            perturbed_img[segments == i] = 0
    return perturbed_img'''
def create_perturbation(img_np, segments, num_segments, idx):
    perturbed_img = img_np.copy()
    mask = np.random.binomial(1, 0.5, num_segments).astype(bool)  # 使用二项分布生成掩码
    perturbed_img[~mask[segments]] = 0  # 关闭被掩码遮住的区域
    return perturbed_img

def create_perturbations_parallel(img_np, segments, num_segments, num_samples):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        perturbed_images = list(tqdm(executor.map(create_perturbation, [img_np]*num_samples, [segments]*num_samples, [num_segments]*num_samples, range(num_samples)),
                                     desc="Creating perturbations total{}", unit="image"))

    return np.array(perturbed_images) 

def predict_perturbations(model, perturbed_images):
    preds = []
    for perturbed_img in tqdm(perturbed_images, desc="Predicting perturbations", unit="image"):
        perturbed_tensor = preprocess_image(perturbed_img)
        with torch.no_grad():
            preds.append(model(perturbed_tensor).cpu().numpy())
    return np.array(preds)

def train_linear_model(segments, perturbed_images, preds):
    coefficients = np.zeros(segments.max() + 1)

    for i in tqdm(range(segments.max() + 1), desc="Training linear model", unit="segment"):
        mask = segments == i
        X = (perturbed_images[:, mask].mean(axis=(1, 2)) > 0).astype(int).reshape(-1, 1)

        if np.sum(X) > 0:
            Y = preds[:, 0]
            lin_reg = LinearRegression().fit(X.reshape(-1, 1), Y)
            coefficients[i] = lin_reg.coef_[0]
        else:
            coefficients[i] = 0

    return coefficients

def main():
    img_path = "Final_Exp\image\Dog.jpg"
    num_samples = 1000

    model = inception_v3(pretrained=True).to(device)
    model.eval()

    original_image = preprocess_image(img_path)
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)

    img_tensor = preprocess_image(img_path)
    img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    segments = quickshift(img_np, kernel_size=4, max_dist=200, ratio=0.2)
    num_segments = np.unique(segments).shape[0]

    perturbed_images = create_perturbations_parallel(img_np, segments, num_segments, num_samples)
    preds = predict_perturbations(model, perturbed_images)

    original_pred = model(original_image).detach().cpu().numpy()
    argmax_index = int(np.argmax(original_pred))

    coefficients = train_linear_model(segments, perturbed_images, preds)

    temp, mask = np.copy(perturbed_images[0]), np.zeros(perturbed_images[0].shape[:2])
    temp = temp.astype(np.uint8)

    mask_min = np.min(mask)
    mask_max = np.max(mask)
    mask_range = mask_max - mask_min

    if mask_range > 0:
        mask = (mask - mask_min) / mask_range * 255
        mask = mask.astype(np.uint8)

    mask = mask.astype(int)

    for i in range(segments.max() + 1):
        if coefficients[i] > 0:
            mask[segments == i] = coefficients[i]

    alpha = 0.4

    temp = temp.astype(np.uint8)

    lime_overlay = mark_boundaries(temp, mask.astype(int), outline_color=(1, 1, 0))

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lime_overlay_with_border = lime_overlay.copy()
    cv2.drawContours(lime_overlay_with_border, contours, -1, (0, 255, 0), 2)

    lime_overlay_with_border = (lime_overlay_with_border - lime_overlay_with_border.min()) / (lime_overlay_with_border.max() - lime_overlay_with_border.min()) * 255
    lime_overlay_with_border = lime_overlay_with_border.astype(np.uint8)

    img_np_array = np.array(img)
    lime_overlay_resized = cv2.resize(lime_overlay_with_border, (img_np_array.shape[1], img_np_array.shape[0]))

    blended_image = cv2.addWeighted(img_np_array, alpha, lime_overlay_resized, 1 - alpha, 30)

    cmap = ListedColormap(['black', 'darkorange', 'darkred'])

    plt.imshow(blended_image, cmap=cmap)
    plt.title('Blended Image with Border (Original + LIME)')
    plt.show()

if __name__ == "__main__":
    main()
