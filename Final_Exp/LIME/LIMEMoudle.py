import os
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# 定义InceptionV3模型
inet_model = InceptionV3()

# 读取图像
image_path = os.path.join("./", r"Final_Exp\image\car2.jpg")
img = image.load_img(image_path, target_size=(299, 299))

# 将图像转换为numpy数组
x = image.img_to_array(img)

# 在第0维度上添加一个维度，以匹配模型的输入要求
x = np.expand_dims(x, axis=0)

# 预处理输入图像
x = preprocess_input(x)

# 进行预测
predictions = inet_model.predict(x)

# 解码预测结果
decoded_predictions = decode_predictions(predictions)

# 输出预测结果
for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
    print(f"{i + 1}: {label}{imagenet_id}({score:.2f})")

# 使用LIME进行图像解释
explainer = lime_image.LimeImageExplainer()

# 注意：由于LIME在解释图像时可能需要大量样本，这里只使用一个样本。在实际应用中，你可能需要更多样本以获得更好的解释。
explanation = explainer.explain_instance(
    x[0], inet_model.predict, top_labels=5, num_samples=1000
)
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# 使用plt.subplots创建一个包含三个子图的图表
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

# 显示原始图像
axes[0].imshow(img)
axes[0].set_title("Original Image")

# 显示LIME解释的图像（前三个标签）
for i, label_id in enumerate(explanation.top_labels[:3]):
    temp, mask = explanation.get_image_and_mask(
        label_id, positive_only=True, num_features=5, hide_rest=False
    )
    lime_img = mark_boundaries(temp / 2 + 0.5, mask)
    
    # 在子图中显示LIME解释的图像
    axes[i + 1].imshow(lime_img)
    axes[i + 1].set_title(f"LIME Explanation: {i}")

# 显示图表
plt.show()
