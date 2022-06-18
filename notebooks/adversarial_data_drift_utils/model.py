from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


def initialize_model(model_params: Dict[str, Any]) -> Pipeline:
    """
    Initializes the model, an Sklearn Pipeline with two steps: tf-idf and logreg.
    :param model_params: a dictionary read from the `config.yml` file, section "model"
    :return: an Sklearn Pipeline object
    """

    # TODO define a model wrapper class instead
    tf_idf_params = model_params["tfidf"]
    logreg_params = model_params["logreg"]

    # initialize TfIdf, logreg, and the Pipeline with the params from a config file
    # TODO support arbitrary params, not only the listed ones.
    text_transformer = TfidfVectorizer(
        stop_words=tf_idf_params["stop_words"],
        ngram_range=eval(tf_idf_params["ngram_range"]),
        lowercase=bool(tf_idf_params["lowercase"]),
        analyzer=tf_idf_params["analyzer"],
        max_features=int(tf_idf_params["max_features"]),
    )

    logreg = LogisticRegression(
        C=int(logreg_params["C"]),
        solver=logreg_params["solver"],
        random_state=int(logreg_params["random_state"]),
        max_iter=int(logreg_params["max_iter"]),
        n_jobs=int(logreg_params["n_jobs"]),
        fit_intercept=bool(logreg_params["fit_intercept"]),
    )

    model = Pipeline([("tfidf", text_transformer), ("logreg", logreg)])

    return model


def train_model(
    model_params: Dict[str, Any],
    train_texts: List[str],
    train_targets: List[int],
) -> Pipeline:
    """
    Trains the model defined in model.py
    :param train_texts: a list of texts to train the model, the model is an sklearn Pipeline
                        with tf-idf as a first step, so raw texts can be fed into the model
    :param train_targets: a list of targets (ints)
    :param model_params: a dictionary with model parameters, see the "model" section of the `config.yaml` file
    :param cross_val_params: a dictionary with cross-validation parameters,
                             see the "cross_validation" section of the `config.yaml` file
    :return: model – the trained model, an Sklearn Pipeline object
    """

    model = initialize_model(model_params=model_params)
    model.fit(X=train_texts, y=train_targets)

    return model


def run_cross_validation(
    model_params: Dict[str, Any],
    cross_val_params: Dict[str, Any],
    train_texts: List[str],
    train_targets: List[int],
) -> np.ndarray:
    """
    Trains the model defined in model.py
    :param train_texts: a list of texts to train the model, the model is an sklearn Pipeline
                        with tf-idf as a first step, so raw texts can be fed into the model
    :param train_targets: a list of targets (ints)
    :param model_params: a dictionary with model parameters, see the "model" section of the `config.yaml` file
    :param cross_val_params: a dictionary with cross-validation parameters,
                             see the "cross_validation" section of the `config.yaml` file
    :return: cv_results – a NumPy array
    """

    model = initialize_model(model_params=model_params)

    skf = RepeatedStratifiedKFold(
        n_splits=cross_val_params["cv_n_splits"],
        n_repeats=cross_val_params["cv_n_repeats"],
        random_state=cross_val_params["cv_random_state"],
    )

    cv_results = cross_val_score(
        estimator=model,
        X=train_texts,
        y=train_targets,
        scoring=cross_val_params["cv_scoring"],
        cv=skf,
        n_jobs=cross_val_params["cv_n_jobs"],
    )

    return cv_results
