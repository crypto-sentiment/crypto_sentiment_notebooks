from typing import Any, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import List


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
        multi_class=logreg_params["multi_class"],
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
    Trains the model defined in model.py with the optional flag to add cross-validation.
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
