import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


def flatten_and_normalize(images, scaler=None):
    flattened_images = np.array([img.flatten() for img in images])

    if scaler is None:
        scaler = StandardScaler()
        # 在每个图像上进行标准化
        flattened_images = np.array(
            [scaler.fit_transform(img.reshape(1, -1))[0] for img in images]
        )

    return flattened_images, scaler


def extract_patches(image, patch_size=(5, 5), stride=(2, 2)):
    patches = []
    for i in range(0, image.shape[0] - patch_size[0] + 1, stride[0]):
        for j in range(0, image.shape[1] - patch_size[1] + 1, stride[1]):
            patch = image[i : i + patch_size[0], j : j + patch_size[1]]
            patches.append(patch)
    return np.array(patches)


def generate_perturbed_patches(
    original_patches, num_samples=1000, perturbation_scale=0.1, random_state=None
):
    random_state = check_random_state(random_state)
    perturbed_patches = original_patches + random_state.normal(
        scale=perturbation_scale, size=original_patches.shape
    )
    return perturbed_patches


def lime_explanation(original_image, perturbed_patches, model, scaler):
    flat_original, _ = flatten_and_normalize([original_image], scaler)
    flat_perturbed, _ = flatten_and_normalize(perturbed_patches, scaler)

    original_prediction = model.predict(flat_original)[0]
    perturbed_predictions = model.predict(flat_perturbed)

    weights = np.exp(-np.linalg.norm(flat_perturbed - flat_original, axis=1))

    interpretable_model = LogisticRegression(max_iter=1000)
    interpretable_model.fit(
        flat_perturbed, perturbed_predictions, sample_weight=weights
    )

    feature_importances = interpretable_model.coef_[0]

    return feature_importances


def visualize_heatmap(image, importance_mask, patch_size=(5, 5), stride=(2, 2)):
    heatmap = np.zeros_like(image)
    idx = 0
    for i in range(0, image.shape[0] - patch_size[0] + 1, stride[0]):
        for j in range(0, image.shape[1] - patch_size[1] + 1, stride[1]):
            heatmap[i : i + patch_size[0], j : j + patch_size[1]] = importance_mask[idx]
            idx += 1

    heatmap /= heatmap.max()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap="hot", interpolation="nearest")
    plt.title("Heatmap of Important Regions")

    plt.show()


# 加载手写数字数据库MNIST
digits = datasets.load_digits()
images = digits.images
labels = digits.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# 展示一个图像示例
plt.imshow(X_train[0], cmap="gray")
plt.title(f"Digit: {y_train[0]}")
plt.show()

# 将图像展平并标准化
flattened_images, scaler = flatten_and_normalize(X_train)

# 训练逻辑回归模型
model = LogisticRegression(max_iter=1000)
model.fit(flattened_images, y_train)

# 预测并评估模型
flattened_test_images, _ = flatten_and_normalize(X_test, scaler)
predictions = model.predict(flattened_test_images)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of the Logistic Regression model on the test set: {accuracy:.2%}")

# 生成扰动图像块
patch_size = (5, 5)
stride = (2, 2)
patches = extract_patches(X_test[0], patch_size=patch_size, stride=stride)
perturbed_patches = generate_perturbed_patches(patches)

# 使用LIME解释图像
feature_importances = lime_explanation(X_test[0], perturbed_patches, model, scaler)

# 可视化热力图
visualize_heatmap(X_test[0], feature_importances, patch_size=patch_size, stride=stride)
