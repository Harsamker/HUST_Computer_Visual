import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch.nn.functional as F
import matplotlib.colors as mcolors

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
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None
        self.hooks = []

        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        for module in self.model.named_modules():
            if module[0] == self.target_layer:
                self.hooks.append(module[1].register_forward_hook(forward_hook))
                self.hooks.append(module[1].register_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        output[:, class_idx].backward()

        gradients = self.gradients
        feature_maps = self.feature_maps
        weights = F.adaptive_avg_pool2d(gradients, 1)

        gcam = torch.mul(feature_maps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(gcam, input_tensor.shape[2:], mode='bilinear', align_corners=False)

        heatmap = gcam.squeeze().cpu().numpy()
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        return heatmap

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()



def main():
    img_path = "Final_Exp\image\car&man.jpg"

    # 加载模型和预处理图像
    model = inception_v3(pretrained=True).to(device)
    model.eval()
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess_image(img_path)

    # 初始化 GradCAM++ 对象
    target_layer = 'Mixed_7c.branch_pool.conv'
    gradcam = GradCAMPlusPlus(model, target_layer)

    # 生成热力图
    heatmap = gradcam.generate_heatmap(img_tensor, None)

    # 自定义颜色映射：蓝色到红色
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_map", ["blue", "white", "red"])

    # 将热力图叠加到原始图像上
    img_np = np.array(img)
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)  # OpenCV默认为BGR，转换为RGB
    superimposed_img = heatmap_color * 0.4 + img_np

    # 创建图像和颜色条的显示
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(superimposed_img / 255)
    ax.set_title('GradCAM++')
    ax.axis('off')

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation='horizontal', pad=0.05, ax=ax)
    cbar.set_label('Heatmap Intensity')

    plt.show()

    # 清理
    gradcam.clear_hooks()

if __name__ == "__main__":
    main()
