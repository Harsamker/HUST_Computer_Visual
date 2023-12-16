import matplotlib as mpl
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
    mask = np.random.binomial(1, 0.5, num_segments).astype(bool)  # 使用二项分布生成掩码
    perturbed_img[~mask[segments]] = 0  # 关闭被掩码遮住的区域
    return perturbed_img


def create_perturbations_parallel(img_np, segments, num_segments, num_samples):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        perturbed_images = list(tqdm(executor.map(lambda x: create_perturbation(img_np, segments, num_segments, x), range(num_samples)),
                                     desc="Creating perturbations total{}", unit="image"))

    return np.array(perturbed_images)


# 在predict_perturbations函数中
def predict_perturbations(model, perturbed_images):
    preds = []
    for perturbed_img in tqdm(perturbed_images, desc="Predicting perturbations", unit="image"):
        perturbed_tensor = preprocess_image(perturbed_img).to(device)  # 将扰动图片移到 GPU
        with torch.no_grad():
            preds.append(model(perturbed_tensor).cpu().numpy())
    preds_array = np.array(preds)
    print("Predictions shape:", preds_array.shape)
    return preds_array

def train_linear_model(segments, avg_activation_per_segment, preds):
    coefficients = np.zeros(segments.max() + 1)

    for i in tqdm(range(segments.max() + 1), desc="Training linear model", unit="segment"):
        X = avg_activation_per_segment[i, :].reshape(-1, 1)
        if X.sum() > 0:
            Y = preds
            lin_reg = LinearRegression(n_jobs=-1).fit(X, Y)
            coefficients[i] = lin_reg.coef_[0]
        else:
            coefficients[i] = 0

    return coefficients


from keras.applications.inception_v3 import preprocess_input, decode_predictions

import matplotlib.gridspec as gridspec

def batch_predict(model, img_np, segments, num_segments, num_samples_per_batch, num_batches):
    all_preds = []
    avg_activation_per_segment = np.zeros((num_segments, num_batches * num_samples_per_batch))

    for batch_idx in tqdm(range(num_batches), desc="Batch predictions"):
        # 创建当前批次的扰动图像
        perturbed_images = create_perturbations_parallel(img_np, segments, num_segments, num_samples_per_batch)

        # 对当前批次的扰动图像进行预测
        batch_preds = []
        for perturbed_img in perturbed_images:
            perturbed_tensor = preprocess_image(perturbed_img).to(device)
            with torch.no_grad():
                batch_pred = model(perturbed_tensor).cpu().numpy()
                batch_preds.append(batch_pred)

                # 更新每个区域的平均激活
                for i in range(num_segments):
                    avg_activation_per_segment[i, batch_idx * num_samples_per_batch:(batch_idx + 1) * num_samples_per_batch] = (perturbed_img[segments == i].mean(axis=(0, 1)) > 0).astype(int)

        all_preds.extend(batch_preds)

        # 清除当前批次的扰动图像，释放内存
        del perturbed_images

    return np.array(all_preds), avg_activation_per_segment


def main():
    img_path = "Final_Exp/image/car1.jpg"
    num_samples = 100  # 总样本数
    num_samples_per_batch = 500  # 每个批次的样本数
    num_batches = num_samples // num_samples_per_batch  # 总批次数

    model = inception_v3(pretrained=True).to(device)
    model.eval()

    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    img_tensor = preprocess_image(img_path)
    img_np_tensor = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

    preds = model(img_tensor).detach().cpu().numpy()
    top_preds = np.argsort(-preds[0])[:3]
    decoded_preds = decode_predictions(preds)[0]
    top_labels = [label for _, label, _ in decoded_preds[:3]]

    segments = quickshift(img_np_tensor, kernel_size=4, max_dist=200, ratio=0.2)
    num_segments = np.unique(segments).shape[0]

    # 分批预测所有扰动图像
    preds, avg_activation_per_segment = batch_predict(model, img_np_tensor, segments, num_segments, num_samples_per_batch, num_batches)

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 4, width_ratios=[3, 3, 3, 0.1])

    ax0 = plt.subplot(gs[1, 0])
    ax0.imshow(img_np)
    ax0.set_title("Original Image")
    ax0.axis('off')

    for i, label in enumerate(top_labels[:3]):
        subplot_index = i + 1
        target_class = top_preds[i]
        target_preds = preds.squeeze(axis=1)[:, target_class]
        coefficients = train_linear_model(segments, avg_activation_per_segment, target_preds)  # 使用新参数
        ax = plt.subplot(gs[1, subplot_index])
        visualize_lime(img_np, segments, coefficients, label, ax)

    cax = plt.subplot(gs[:, -1])
    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=0, vmax=1)
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')
    cb.set_label('Importance')

    plt.tight_layout()
    plt.show()

def visualize_lime(img_np, segments, coefficients, target_class, ax):
    mask = np.zeros(segments.shape)
    for i, coef in enumerate(coefficients):
        if coef > 0:
            mask[segments == i] = coef
    mask = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))
    heatmap = np.uint8(255 * mask)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 1, heatmap, 0.5, 0)
    ax.imshow(overlay)
    ax.set_title(f"Class: {target_class}")
    ax.axis('off')

if __name__ == "__main__":
    main()