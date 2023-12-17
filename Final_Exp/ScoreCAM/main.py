import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import cv2
import torch.nn.functional as F
from keras.applications.inception_v3 import decode_predictions

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

class ScoreCAM:
    def __init__(self, model, target_layer, resize_size=(224, 224)):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.hooks = []
        self.resize_size = resize_size  # 特征图的下采样尺寸

        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = F.interpolate(output.detach(), self.resize_size, mode='bilinear', align_corners=False)

        for module in self.model.named_modules():
            if module[0] == self.target_layer:
                self.hooks.append(module[1].register_forward_hook(forward_hook))

    def generate_heatmap(self, input_tensor, class_idx, batch_size=16):
        # 获取特征图
        _ = self.model(input_tensor)
        feature_maps = self.feature_maps

        # 初始化热力图
        heatmap = torch.zeros(input_tensor.shape[2:], device=device)
        num_batches = (feature_maps.shape[1] + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, feature_maps.shape[1])
            batch_heatmap = torch.zeros(input_tensor.shape[2:], device=device)

            for i in range(batch_start, batch_end):
                mask = feature_maps[:, i, :, :]
                mask = F.interpolate(mask.unsqueeze(0), input_tensor.shape[2:], mode='bilinear', align_corners=False)
                mask = mask - mask.min()
                mask = mask / (mask.max() + 1e-8)

                masked_input = input_tensor * mask
                output = self.model(masked_input)
                score = F.softmax(output, dim=1)[0, class_idx]
                batch_heatmap += score * mask.squeeze(0).squeeze(0)

                del mask, masked_input, output
            heatmap += batch_heatmap
            del batch_heatmap
            torch.cuda.empty_cache()

        heatmap = heatmap.cpu().detach().numpy()
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        return heatmap


    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
            

if __name__ == "__main__":
    img_path = r"Final_Exp\image\img1.jpg"

    model = inception_v3(pretrained=True).to(device)
    model.eval()
    img_tensor = preprocess_image(img_path)

    preds = model(img_tensor).detach().cpu().numpy()
    top_preds = np.argsort(-preds[0])[:5]
    decoded_preds = decode_predictions(preds)[0]
    top_labels = [label for _, label, _ in decoded_preds[:5]]

    target_layer = 'Mixed_7c.branch_pool.conv'
    scorecam = ScoreCAM(model, target_layer)

    fig, axes = plt.subplots(1, 6, figsize=(24, 5)) 

    img_np = np.array(Image.open(img_path))
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    cmap = mcolors.LinearSegmentedColormap.from_list("custom_map", ["blue", "white", "red"])

    for i, class_idx in enumerate(top_preds):
        heatmap = scorecam.generate_heatmap(img_tensor, class_idx)
        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        superimposed_img = heatmap_color * 0.4 + img_np

        axes[i + 1].imshow(superimposed_img / 255)
        axes[i + 1].set_title(f"Class: {top_labels[i]}")
        axes[i + 1].axis('off')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation='horizontal', pad=0.05, ax=axes.ravel().tolist(), aspect=40)
    cbar.set_label('Heatmap Intensity')
    plt.suptitle('ScoreCAM', fontsize=12)
    plt.show()

    scorecam.clear_hooks()
