import os
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input,
    decode_predictions,
)
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# 定义InceptionV3模型
inet_model = InceptionV3()

# 使用LIME自带的示例图像
img = lime_image.load_image(pos=0)

# 在第0维度上添加一个维度，以匹配模型的输入要求
x = np.expand_dims(img, axis=0)

# 预处理输入图像
x = preprocess_input(x)

# 进行预测
predictions = inet_model.predict(x)

# 解码预测结果
decoded_predictions = decode_predictions(predictions)

# 输出预测结果
for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
    print(f"{i + 1}: {label} ({score:.2f})")

# 获取概率最高的标签索引
top_label_index = np.argmax(predictions[0])

# 使用LIME进行图像解释
explainer = lime_image.LimeImageExplainer()

# 注意：由于LIME在解释图像时可能需要大量样本，这里只使用一个样本。在实际应用中，你可能需要更多样本以获得更好的解释。
explanation = explainer.explain_instance(
    np.array(x[0]), inet_model.predict, top_labels=[top_label_index], num_samples=1000
)

# 显示原始图像和LIME解释的图像在同一张图表中
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 显示原始图像
axes[0].imshow(img)
axes[0].set_title("Original Image")

# 显示LIME解释的图像
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False
)
axes[1].imshow(mark_boundaries(temp / 2 + 0.5, mask))
axes[1].set_title("LIME Explanation")

plt.show()
