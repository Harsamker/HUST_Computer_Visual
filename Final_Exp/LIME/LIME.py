import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from sklearn.linear_model import LinearRegression
from skimage.segmentation import quickshift
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm.auto import tqdm
import matplotlib.colors as mcolors
from keras.applications.inception_v3 import decode_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def predict(model, imgs):
    model.eval()
    with torch.no_grad():
        preds = []
        for img in imgs:
            img_tensor = preprocess_image(Image.fromarray(img))
            pred = model(img_tensor)
            preds.append(pred.cpu().numpy())
    return np.array(preds)
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).to(device)
    return img_tensor.unsqueeze(0)
from concurrent.futures import ThreadPoolExecutor

def create_perturbations(img_np, segments, num_perturb):
    perturbed_images = []
    weights = []
    def create_perturbation(i):
        perturbed_img = img_np.copy()
        active_pixels = np.random.choice([False, True], len(np.unique(segments)))
        for segment_id, active in enumerate(active_pixels):
            perturbed_img[segments == segment_id] = 0 if not active else perturbed_img[segments == segment_id]
        return perturbed_img, active_pixels
    #开多线程加速
    with ThreadPoolExecutor(max_workers=None) as executor:
        results = list(tqdm(executor.map(create_perturbation, range(num_perturb)), total=num_perturb, desc="Creating Perturbations"))
        for perturbed_img, active_pixels in results:
            perturbed_images.append(perturbed_img)
            weights.append(active_pixels)
            
    return np.array(perturbed_images), np.array(weights)

if __name__ == "__main__":
    img_path = r"Final_Exp\data4\both.jpg"
    model = inception_v3(pretrained=True).to(device)
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    img_tensor = preprocess_image(img)
    segments = quickshift(img_np, kernel_size=4, max_dist=200, ratio=0.2)
    
    num_perturb = 10000
    perturbed_images, weights = create_perturbations(img_np, segments, num_perturb)
    
    preds = []
    for i in tqdm(range(perturbed_images.shape[0]), desc="Predicting", leave=
                  False):
        pred = predict(model, [perturbed_images[i]])
        preds.append(pred[0])
    preds = np.array(preds)
    
    original_pred = predict(model, [img_np])[0]

    top_preds = np.argsort(-original_pred[0])[:5]
    decoded_preds = decode_predictions(original_pred)[0]
    top_labels = [label for _, label, _ in decoded_preds[:5]]

    fig, axes = plt.subplots(1, 6, figsize=(20, 5))
    axes[0].imshow(img_np)
    axes[0].axis('off')
    axes[0].set_title('Original Image')

    # 自定义颜色映射，使得低贡献度区域透明
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_map", [(0, "blue"), (0.5, "white"), (1, "red")]
    )

    for idx, class_idx in enumerate(top_preds):
        class_preds = preds[:, 0, class_idx]
        lin_reg = LinearRegression()
        lin_reg.fit(weights, class_preds)
        coef = lin_reg.coef_
        exp_img = np.zeros(segments.shape)
        masks = np.zeros(segments.shape)
        for segment_id, w in enumerate(coef):
            if w > 0:
                exp_img[segments == segment_id] = w
                masks[segments == segment_id] = 1 
            else:
                masks[segments == segment_id] = 0
        exp_img = (exp_img - exp_img.min()) / (exp_img.max() - exp_img.min())
        heatmap = cv2.applyColorMap(np.uint8(255 * exp_img), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        superimposed_img = np.where(masks[:, :, np.newaxis], heatmap * 0.4 + img_np * 0.6, img_np)
        axes[idx + 1].imshow(superimposed_img.astype('uint8'))
        axes[idx + 1].axis('off')
        axes[idx + 1].set_title(f"Class: {top_labels[idx]}")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation='horizontal', pad=0.05, ax=axes.ravel().tolist(), aspect=40)
    cbar.set_label('Heatmap Intensity')
    plt.suptitle('LIME', fontsize=12)
    plt.show()