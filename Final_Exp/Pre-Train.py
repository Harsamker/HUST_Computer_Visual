import torch
from torchvision import transforms
from torch.autograd import Variable
from torchvision.models import Inception3
from PIL import Image
from torchray.attribution.grad_cam import grad_cam
import matplotlib.pyplot as plt

# 定义 Inception3 模型
inet_model = Inception3()

# 读取图像
image_path = r"Final_Exp\data4\dog.jpg"
img = Image.open(image_path).convert('RGB')

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)  # 添加 batch 维度

# 将输入数据转换为 PyTorch 的 Variable
input_variable = Variable(input_batch, requires_grad=True)

# 使用 torchray 进行图像解释
cam = grad_cam(inet_model, input_variable, target_layer='Mixed_7c', target=torch.argmax(inet_model(input_variable)).item())

# 将图像和热力图叠加在一起
cam_pp_image = transforms.ToPILImage()(cam)
overlay = Image.blend(img, cam_pp_image.convert("RGB"), alpha=0.7)

# 显示原始图像、热力图和叠加图像
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img)
axes[0].set_title("Original Image")

axes[1].imshow(cam_pp_image, cmap='jet')
axes[1].set_title("GradCAM++ Heatmap")

axes[2].imshow(overlay)
axes[2].set_title("Overlay")

plt.show()
