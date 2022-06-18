from pathlib import Path
from typing import Any, Dict

import eli5
import numpy as np
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline


def get_notebooks_root() -> Path:
    """
    Return a Path object pointing to the project root directory.
    :return: Path
    """
    return Path(__file__).parent.parent


def load_adv_validation_params() -> Dict[str, Any]:
    """
    Loads adversarial validation configuration params defined in the `config.yaml` file.
    :return: a nested dictionary corresponding to the `config.yaml` file.
    """
    project_root: Path = get_notebooks_root()
    with open(project_root / "adversarial_data_drift_utils" / "config.yml") as f:
        params: Dict[str, Any] = yaml.load(f, Loader=yaml.FullLoader)
    return params


def plot_cv_results(scores: np.ndarray, num_sigma: int = 2, bins: int = 10):
    """
    Visualizes cross-validation results as a histogram with standard deviations.

    :param scores: a Numpy array with cross-valdiation scores
    :param num_sigma: the number of standard deviations to show on the plot
    :param bins: the number of bins for the histplot

    """
    sns.set()

    mean = scores.mean()
    lower, upper = mean - num_sigma * scores.std(), mean + num_sigma * scores.std()

    sns.histplot(scores, bins=bins, kde=True)
    plt.vlines(
        x=[mean], ymin=0, ymax=10, label="mean", linestyles="dashed", color="black"
    )
    plt.vlines(
        x=[lower, upper],
        ymin=0,
        ymax=10,
        color="red",
        label=f"+/- {num_sigma} std",
        linestyles="dashed",
    )
    plt.title("Adversarial validation ROC AUC scores")
    plt.legend()


def show_weights(model: Pipeline, n_top: int = 20):
    """
    :param model: sklearn pipeline defined in model.py
    :param n_top: the number of coefficients to show for each class
    """
    return eli5.show_weights(
        estimator=model,
        feature_names=list(model.named_steps["tfidf"].get_feature_names_out()),
        top=(n_top, n_top),
        target_names=["old_data", "new_data"],
    )
