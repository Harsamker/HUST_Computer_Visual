import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2


class ScoreCAM:
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
    # Load pre-trained model
    model = models.resnet50(pretrained=True)
    target_layer = "layer4"
    score_cam = ScoreCAM(model, target_layer)

    # Load and preprocess input image
    image_path = r"Final_Wxp\image\car&man.jpg"
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    input_image = preprocess(image).unsqueeze(0)

    # Classify the image
    with torch.no_grad():
        output = model(input_image)

    # Get the predicted class
    _, predicted_class = torch.max(output, 1)

    # Generate heatmap for the predicted class
    heatmap = score_cam.generate_heatmap(input_image, predicted_class.item())

    # Resize the heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (3000, 2000))

    # Normalize the heatmap and ensure it has the same number of channels as the original image
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.merge([heatmap, heatmap, heatmap])

    # Overlay the heatmap on the original image
    # Overlay the heatmap on the original image with adjusted weights
    overlaid_image = np.clip(
        np.array(image) * 0.7 + heatmap[:, :, :3] * 0.3, 0, 255
    ).astype(np.uint8)

    import matplotlib.pyplot as plt

    # Display the original image
    plt.imshow(image)
    plt.title("Original Image")
    plt.show()

    # Display the overlaid image
    plt.imshow(overlaid_image)
    plt.title("Overlaid Image")
    plt.show()


if __name__ == "__main__":
    main()
