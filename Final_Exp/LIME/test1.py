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

# 读取图像
image_path = os.path.join("./", r"Final_Exp\image\car1.jpg")
img = image.load_img(image_path, target_size=(474, 266))

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
    print(f"{i + 1}: {label} ({score:.2f})")

# 使用LIME进行图像解释
explainer = lime_image.LimeImageExplainer()

# 注意：由于LIME在解释图像时可能需要大量样本，这里只使用一个样本。在实际应用中，你可能需要更多样本以获得更好的解释。
explanation = explainer.explain_instance(
    x[0], inet_model.predict, top_labels=5, num_samples=1000
)

# 显示原始图像
plt.imshow(img)
plt.title("Original Image")
plt.show()

# 显示LIME解释的图像
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False
)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.title("LIME Explanation")
plt.show()
