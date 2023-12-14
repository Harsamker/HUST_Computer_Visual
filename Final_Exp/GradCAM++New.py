import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.inception_v3(pretrained=True).to(device)
model.eval()


# 图像预处理
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(img_path)
    image = transform(image).unsqueeze(0)
    return image


# 预处理输入图像
img_path = r"Final_Exp\image\Dog.jpg"  # 替换为你的图像路径
input_tensor = preprocess_image(img_path).to(device)


class GradCAMpp:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer_name = target_layer
        self.model.eval()

        # 注册钩子
        self.hook_registered = False
        self.target_layer_activations = None
        self.target_layer_gradients = None
        self.register_hooks()

    def find_target_layer(self, model, target_layer_name):
        for name, module in model.named_modules():
            if name == target_layer_name:
                return module
        return None

    def register_hooks(self):
        # 找到目标层
        self.target_layer = self.find_target_layer(self.model, self.target_layer_name)

        if self.target_layer is None:
            raise ValueError(f"Target layer {self.target_layer_name} not found in the model")

        # 注册钩子
        if not self.hook_registered:
            self.target_layer.register_forward_hook(self.forward_hook)
            self.hook_registered = True

    def forward_hook(self, module, input, output):
        self.target_layer_activations = output

    def get_target_layer_output(self, input_tensor):
        # 执行正向传播以触发钩子
        _ = self.model(input_tensor)
        return self.target_layer_activations

    def __call__(self, input_tensor, target_category=None):
        # 正向传播
        output = self.model(input_tensor)
        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())

        # 获取目标层的输出
        target_layer_output = self.get_target_layer_output(input_tensor)

        # 获取梯度
        self.model.zero_grad()
        gradient = torch.autograd.grad(outputs=output, inputs=input_tensor, grad_outputs=torch.ones_like(output),
                                       retain_graph=True, create_graph=True)[0].cpu().data.numpy()

        # 获取激活和权重
        activations = target_layer_output.cpu().data.numpy()[0, :]
        weights = self.compute_weights(gradient, activations)

        # 计算加权组合的特征图
        cam = np.ones(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (299, 299))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam

    def compute_weights(self, gradients, activations):
        alpha_numer = gradients ** 2
        alpha_denom = gradients ** 2 * 2 + activations * gradients ** 3
        alpha_denom = np.where(alpha_denom != 0, alpha_denom, np.ones(alpha_denom.shape))
        alphas = alpha_numer / alpha_denom

        weights = np.maximum(gradients, 0) * alphas
        weights = np.sum(weights, axis=(1, 2))

        return weights


# 创建 GradCAM++ 对象
grad_cam_pp = GradCAMpp(model, 'Mixed_7c')  # 'Mixed_7c' 是 Inception v3 的一个层名

# 应用 GradCAM++
cam = grad_cam_pp(input_tensor)

# 将热图叠加到原始图像上
img = cv2.imread(img_path, 1)
img = np.float32(cv2.resize(img, (299, 299))) / 255
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img

# 显示图像
plt.imshow(superimposed_img[..., ::-1])
plt.show()
