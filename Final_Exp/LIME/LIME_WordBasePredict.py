import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt


class LimeExplainer:
    def __init__(self, model, num_samples=500, random_state=None):
        self.model = model
        self.num_samples = num_samples
        self.random_state = check_random_state(random_state)

    def generate_samples(self, instance, num_features):
        samples = self.random_state.normal(0, 1, size=(self.num_samples, num_features))
        return instance + samples

    def explain_instance(self, instance, num_features=5, num_inferences=100):
        samples = self.generate_samples(instance, num_features)

        # Get predictions for the generated samples
        predictions = self.model.predict_proba(samples)

        # Calculate distances between instance and generated samples
        distances = np.linalg.norm(samples - instance, axis=1)

        # Weight predictions by distance
        distances_exp = np.exp(-distances).reshape(-1, 1)
        weighted_predictions = distances_exp * predictions

        # Fit a linear model to the weighted predictions
        interpretable_model = LogisticRegression()
        interpretable_model.fit(samples, np.argmax(weighted_predictions, axis=1))

        # Get coefficients of the linear model as feature importances
        feature_importances = interpretable_model.coef_[0]

        return feature_importances


def main():
    # 示例：使用一个简单的分类模型
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # 加载数据集
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 训练分类模型
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 需要解释的实例
    instance_to_explain = X_test[0]

    # 创建LIME解释器
    lime_explainer = LimeExplainer(model)

    # 解释实例
    num_features = 4  # 根据你的数据集而定
    feature_importances = lime_explainer.explain_instance(
        instance_to_explain, num_features
    )

    # 可视化特征重要性
    plt.bar(range(num_features), feature_importances)
    plt.xticks(
        range(num_features), ["Feature {}".format(i) for i in range(num_features)]
    )
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("LIME Feature Importances")
    plt.show()


if __name__ == "__main__":
    main()
