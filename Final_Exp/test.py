import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import lime

# Load pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)
model.eval()


# Grad-CAM implementation for 9 channels
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_map = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer = self.model._modules.get(self.target_layer)
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)

        self.handles = [forward_handle, backward_handle]

    def generate_heatmap(self, input_image, target_class):
        self.model.zero_grad()

        output = self.model(input_image)
        score = output[0, target_class]

        score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.feature_map, dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.detach().numpy()[0, 0, :, :]
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam


def main():
    # Load and preprocess input image
    image_path = r"Final_Wxp\image\car&man.jpg"
    image = Image.open(image_path).convert("RGB")  # Assuming the image is RGB
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Convert 9-channel image to a 3-channel image for Grad-CAM
    image = image.convert("RGB")

    input_image = preprocess(image).unsqueeze(0)

    # Specify the target layer for Grad-CAM
    target_layer = "layer4"

    # Specify the target class index (replace with the correct index based on your task)
    target_class = 0  # Replace with the correct index

    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)

    # Generate heatmap for the specified target class
    heatmap = grad_cam.generate_heatmap(input_image, target_class)

    # Resize the heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (3000, 2000))

    # Apply color map and ensure heatmap has the same number of channels as the original image
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.merge([heatmap, heatmap, heatmap])

    # Overlay the heatmap on the original image
    overlaid_image = cv2.addWeighted(np.array(image), 0.7, heatmap, 0.3, 0)

    # Display the original image
    cv2.imshow("Original Image", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    # Display the overlaid image
    cv2.imshow("Grad-CAM Heatmap", overlaid_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
