import copy
from functools import partial

import numpy as np
import sklearn
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
from skimage.segmentation import quickshift
from tqdm.auto import tqdm


class ImageExplanation(object):
    def __init__(self, image, segments):
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}

    def get_image_and_mask(
        self,
        label,
        positive_only=True,
        negative_only=False,
        hide_rest=False,
        num_features=5,
        min_weight=0.0,
    ):
        if label not in self.local_exp:
            raise KeyError("Label not in explanation")
        if positive_only & negative_only:
            raise ValueError(
                "Positive_only and negative_only cannot be true at the same time."
            )
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp if x[1] > 0 and x[1] > min_weight][:num_features]
        if negative_only:
            fs = [x[0] for x in exp if x[1] < 0 and abs(x[1]) > min_weight][
                :num_features
            ]
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = -1 if w < 0 else 1
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
            return temp, mask


class SimpleImageExplainer(object):
    def __init__(self, random_state=None):
        self.random_state = check_random_state(random_state)

    def explain_instance(
        self,
        image,
        classifier_fn,
        num_samples=1000,
        batch_size=10,
        distance_metric="cosine",
        random_seed=None,
        progress_bar=True,
    ):
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        segments = quickshift(
            image, kernel_size=4, max_dist=200, ratio=0.2, random_seed=random_seed
        )

        fudged_image = image.copy()

        data, labels = self.data_labels(
            image,
            fudged_image,
            segments,
            classifier_fn,
            num_samples,
            batch_size=batch_size,
            progress_bar=progress_bar,
        )

        distances = sklearn.metrics.pairwise_distances(
            data, data[0].reshape(1, -1), metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)
        (
            ret_exp.intercept[1],
            ret_exp.local_exp[1],
            ret_exp.score[1],
            ret_exp.local_pred[1],
        ) = self.explain_instance_with_data(
            data,
            labels,
            distances,
            1,
            100,
            model_regressor=None,
            feature_selection="auto",
        )
        return ret_exp

    def data_labels(
        self,
        image,
        fudged_image,
        segments,
        classifier_fn,
        num_samples,
        batch_size=10,
        progress_bar=True,
    ):
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features).reshape(
            (num_samples, n_features)
        )
        labels = []
        data[0, :] = 1
        imgs = []
        rows = tqdm(data) if progress_bar else data
        for row in rows:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
        return data, np.array(labels)

    def explain_instance_with_data(
        self,
        data,
        labels,
        distances,
        label,
        num_features,
        model_regressor=None,
        feature_selection="auto",
    ):
        # Implement your explanation method here
        # This is a placeholder for simplicity
        intercept = 0
        local_exp = [(i, 0.1) for i in range(num_features)]
        score = 0.8
        local_pred = np.dot(data, local_exp)

        return intercept, local_exp, score, local_pred


# Example usage:
# classifier_fn is the predict_proba function of your classifier
# Modify this function based on your classifier
def classifier_fn(images):
    # This is a placeholder, replace it with your actual classifier prediction logic
    return np.random.rand(len(images), 1)


# Example usage of the simple explainer
explainer = SimpleImageExplainer(random_state=42)
image_instance = np.random.rand(64, 64, 3)  # Replace with your actual image
explanation = explainer.explain_instance(image_instance, classifier_fn)

# Access the explanation results
print("Intercept:", explanation.intercept[1])
print("Local explanation:", explanation.local_exp[1])
print("Score:", explanation.score[1])
print("Local prediction:", explanation.local_pred[1])
