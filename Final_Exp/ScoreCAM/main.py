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
import torch.nn.functional as F

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

class ScoreCAM:
    def __init__(self, model, target_layer, resize_size=(224, 224)):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.hooks = []
        self.resize_size = resize_size  # 新增：特征图的下采样尺寸

        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = F.interpolate(output.detach(), self.resize_size, mode='bilinear', align_corners=False)

        for module in self.model.named_modules():
            if module[0] == self.target_layer:
                self.hooks.append(module[1].register_forward_hook(forward_hook))

    def generate_heatmap(self, input_tensor, batch_size=16):
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
                class_idx = output.argmax(dim=1).item()
                score = F.softmax(output, dim=1)[0, class_idx]
                batch_heatmap += score * mask.squeeze(0).squeeze(0)

                # 清除不再需要的变量
                del mask, masked_input, output

            # 累加每批次的结果
            heatmap += batch_heatmap

            # 清除每批次的热力图和释放缓存
            del batch_heatmap
            torch.cuda.empty_cache()

        # 归一化最终热力图
        heatmap = heatmap.cpu().detach().numpy()
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        return heatmap


    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
def main():
    img_path = "Final_Exp\image\car2.jpg"

    # 加载模型和预处理图像
    model = inception_v3(pretrained=True).to(device)
    model.eval()
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess_image(img_path)

    
    target_layer = 'Mixed_7c.branch_pool.conv'  # 指定目标层，适用于 Inception V3
    
    # 初始化 ScoreCAM 对象
    scorecam = ScoreCAM(model, target_layer)

    # 生成热力图
    heatmap = scorecam.generate_heatmap(img_tensor)

    # 自定义颜色映射：蓝色到红色
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_map", ["blue", "white", "red"])

    # 将热力图叠加到原始图像上
    img_np = np.array(img)
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed_img = heatmap_color * 0.4 + img_np

    # 使用 gridspec 创建图像和颜色条的布局
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(superimposed_img / 255)
    ax0.set_title('ScoreCAM')
    ax0.axis('off')

    # 添加颜色条
    ax1 = fig.add_subplot(gs[1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, cax=ax1, orientation='horizontal')

    plt.show()

    # 清理
    scorecam.clear_hooks()

if __name__ == "__main__":
    main()

