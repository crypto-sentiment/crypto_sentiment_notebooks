from importlib import import_module
from typing import Any, Callable, Dict
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from sklearn.calibration import CalibrationDisplay


def build_object(
    object_cfg: Dict[str, Any], is_hugging_face: bool = False, **kwargs: Dict[str, Any]
) -> Callable:
    """Build object from config.

    Config should have the following construction:

    class: <class name>
    params:
        <param name>: val

    params - constructor parameters.
    Also if kwargs is passed they will be added as constructor parameters.

    :param object_cfg: object config
    :param is_hugging_face: whether object is hugging face model, defaults to False
    :raises ValueError: if config doesn't contain the class key
    :return: created object
    """
    if "class" not in object_cfg.keys():
        raise ValueError("class key schould be in config")

    if "params" in object_cfg.keys():
        params = object_cfg["params"]

        for key, val in params.items():
            kwargs[key] = val
    else:
        params = {}

    if is_hugging_face:
        return get_instance(object_cfg["class"]).from_pretrained(**kwargs)

    return get_instance(object_cfg["class"])(**kwargs)


def get_instance(object_path: str) -> Callable:
    """Return object instance.

    :param object_path: instance name, for example transformers.DistilBertTokenizerFast
    :return: object instance
    """
    module_path, class_name = object_path.rsplit(".", 1)
    module = import_module(module_path)

    return getattr(module, class_name)


def get_probs_gt_by_class(
    val_predict_probs: np.ndarray, val_labels: np.ndarray
) -> Dict[str, Any]:
    val_predict = val_predict_probs.argmax(-1)

    ohe = OneHotEncoder(sparse=False)
    ohe_val_labels = ohe.fit_transform(np.array(val_labels).reshape(-1, 1))

    label_names = ("negative", "neutral", "positive")
    label_idxs = (0, 1, 2)
    probs_gt_by_class = {}

    for label_name, label_idx in zip(label_names, label_idxs):
        # select predictions for particular label
        predicted_label_idx = val_predict == label_idx

        # get predicted probs for particular label
        predicted_label_probs = val_predict_probs[predicted_label_idx][:, label_idx]

        # get gt labels (1, 0)
        gt_labels = ohe_val_labels[:, label_idx][predicted_label_idx]

        probs_gt_by_class[f"{label_name}_probs"] = predicted_label_probs
        probs_gt_by_class[f"gt_{label_name}"] = gt_labels

    return probs_gt_by_class


def plot_calibration_curves(probs_gt_by_class: Dict[str, Any], n_bins: int = 10):

    f, ax = plt.subplots(1, 1, tight_layout=True, figsize=(10, 6))

    CalibrationDisplay.from_predictions(
        probs_gt_by_class["gt_negative"],
        probs_gt_by_class["negative_probs"],
        n_bins=n_bins,
        name="negative class",
        ax=ax,
    )
    CalibrationDisplay.from_predictions(
        probs_gt_by_class["gt_neutral"],
        probs_gt_by_class["neutral_probs"],
        n_bins=n_bins,
        name="neutral class",
        ax=ax,
    )
    CalibrationDisplay.from_predictions(
        probs_gt_by_class["gt_positive"],
        probs_gt_by_class["positive_probs"],
        n_bins=n_bins,
        name="positive class",
        ax=ax,
    )

    return f


def plot_distributions(probs_gt_by_class: Dict[str, Any]):
    f, ax = plt.subplots(1, 3, tight_layout=True, figsize=(15, 4))

    ax[0].hist(probs_gt_by_class["negative_probs"], color="blue")
    ax[1].hist(probs_gt_by_class["neutral_probs"], color="orange")
    ax[2].hist(probs_gt_by_class["positive_probs"], color="green")
    ax[0].set(title="negative", xlabel="Mean predicted probability", ylabel="Count")
    ax[1].set(title="neutral", xlabel="Mean predicted probability", ylabel="Count")
    ax[2].set(title="positive", xlabel="Mean predicted probability", ylabel="Count")

    return f
