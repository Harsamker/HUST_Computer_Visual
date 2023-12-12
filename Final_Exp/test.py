import os
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input,
    decode_predictions,
)

# 定义InceptionV3模型
inet_model = InceptionV3()

# 读取图像
image_path = os.path.join("./", "your_image.jpg")  # 将 "your_image.jpg" 替换为你的图像路径
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
    print(f"{i + 1}: {label} ({score:.2f})")
