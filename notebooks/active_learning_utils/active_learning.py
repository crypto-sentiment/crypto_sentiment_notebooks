from typing import Dict, Any, List, Tuple, cast
import pandas as pd
from copy import deepcopy

from .pipeline import SentimentPipeline
from .train import predict, train_pl_model
import numpy as np
from torch.utils.data import DataLoader
from .dataset import prepare_dataset, build_dataloaders
from scipy.stats import entropy, mode


class ActiveLearning:
    """Active learning base class.

    :param cfg: model config.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Init active learning."""
        self.cfg = cfg

    def query(self, model: SentimentPipeline, pool_data: pd.DataFrame) -> pd.DataFrame:
        """Generate new chunk from pool data base on querying strategy.

        :param model: current model
        :param pool_data: pool data from where sample new chunk
        :return: new chunk of data to be added to the current dataset
        """
        raise NotImplementedError

    def _get_dataloader(self, dataset: pd.DataFrame) -> DataLoader:
        """Create dataloader for a given dataset.

        :param dataset: input dataset
        :return: result dataloder
        """
        dataset = prepare_dataset(
            self.cfg,
            dataset["title"].tolist(),
            dataset["label"].tolist(),
        )

        return DataLoader(dataset, batch_size=self.cfg["val_batch_size"], shuffle=False)

    def stopping_criterion(self, pool_data: pd.DataFrame) -> bool:
        """Implement stopping criterion for active learning loop.

        :param pool_data: current pool data
        :return: whether to stop active learning
        """
        raise NotImplementedError

    def train(
        self,
        init_data: pd.DataFrame,
        val_data: pd.DataFrame,
        pool_data: pd.DataFrame,
    ) -> Tuple[List[int], List[List[float]]]:
        """Run active learning train loop.

        :param init_data: initial data, used to train model for the first time
        :param val_data: validation data
        :param pool_data: pool data from where new chunks will be sampled
        :return: train sizes and validation scores
        """
        self.base_pool_size = len(pool_data)

        train_sizes: List[int] = []
        val_scores: List[List[float]] = []

        current_dataset = deepcopy(init_data)

        while True:

            train_sizes.append(len(current_dataset))

            train_dataloader, val_dataloader = build_dataloaders(
                self.cfg,
                current_dataset["title"],
                current_dataset["label"],
                val_data["title"],
                val_data["label"],
            )

            scores, model = train_pl_model(
                self.cfg, train_dataloader, val_dataloader, return_model=True
            )

            val_scores.append(scores["val_acc"][-1])

            if self.stopping_criterion(pool_data):
                break

            query_instance = self.query(cast(SentimentPipeline, model), pool_data)
            pool_data = pool_data.drop(query_instance.index)

            current_dataset = pd.concat([current_dataset, query_instance], axis=0)

        return train_sizes, val_scores


class RandomSampling(ActiveLearning):
    """Random sampling querying strategy.

    :param cfg: model config
    :param select_top_percent: percent of the pool data to sample on every step, defaults to 0.1
    """

    def __init__(self, cfg: Dict[str, Any], select_top_percent: float = 0.1) -> None:
        """Init random sampling."""
        super().__init__(cfg)

        self.select_top_percent = select_top_percent
        self.query_instances: List[pd.DataFrame] = []

    def stopping_criterion(self, pool_data: pd.DataFrame) -> bool:
        """Implement stopping criterion for active learning loop.

        Process is stopped if pool data doesn't have enough examples to sample.

        :param pool_data: current pool data
        :return: whether to stop active learning
        """
        num_new_samples = int(self.base_pool_size * self.select_top_percent)

        return len(pool_data) < num_new_samples

    def query(self, model: SentimentPipeline, pool_data: pd.DataFrame) -> pd.DataFrame:
        """Select samples to add to the current train dataset.

        New samples are selected randomly.

        :param model: model
        :param pool_data: pool data from where new chunk will be sampled
        :return: selected samples
        """
        num_new_samples = int(self.base_pool_size * self.select_top_percent)
        query_instance = pool_data.sample(
            num_new_samples, random_state=self.cfg["seed"]
        )

        self.query_instances.append(deepcopy(query_instance))

        return query_instance


class LeastConfidenceSampling(ActiveLearning):
    """Least confidence querying strategy.

    :param cfg: model config
    :param select_top_percent: percent of the pool data to sample on every step, defaults to 0.1
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        select_top_percent: float = 0.1,
        num_stochastic_forward_passes: int = 1,
    ) -> None:
        """Init least confidence sampling."""
        super().__init__(cfg)

        self.select_top_percent = select_top_percent
        self.query_instances: List[pd.DataFrame] = []
        self.num_stochastic_forward_passes = num_stochastic_forward_passes

    def stopping_criterion(self, pool_data: pd.DataFrame) -> bool:
        """Implement stopping criterion for active learning loop.

        Process is stopped if pool data doesn't have enough examples to sample.

        :param pool_data: current pool data
        :return: whether to stop active learning
        """
        num_new_samples = int(self.base_pool_size * self.select_top_percent)

        return len(pool_data) < num_new_samples

    def query(self, model: SentimentPipeline, pool_data: pd.DataFrame) -> pd.DataFrame:
        """Select samples to add to the current train dataset.

        New samples are least confident predictions.

        :param model: model
        :param pool_data: pool data from where new chunk will be sampled
        :return: selected samples
        """
        dataloader = self._get_dataloader(pool_data)

        # [num_passes, num_samples, 3]
        prediction_probs = predict(
            model,
            dataloader,
            num_stochastic_forward_passes=self.num_stochastic_forward_passes,
        )

        if self.num_stochastic_forward_passes > 1:
            prediction_probs = prediction_probs.mean(axis=0)

        # [num_samples, 1]
        predicted_label_prob = np.max(prediction_probs.squeeze(), axis=1)

        pool_copy = deepcopy(pool_data)

        pool_copy["predicted_label_prob"] = predicted_label_prob

        num_new_samples = int(self.base_pool_size * self.select_top_percent)
        query_instance = pool_copy.sort_values(
            "predicted_label_prob", ascending=True
        ).head(num_new_samples)

        self.query_instances.append(deepcopy(query_instance))

        query_instance = query_instance.drop(columns=["predicted_label_prob"])

        return query_instance


class EntropySampling(ActiveLearning):
    """Entropy based querying strategy.

    :param cfg: model config
    :param select_top_percent: percent of the pool data to sample on every step, defaults to 0.1
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        select_top_percent: float = 0.1,
        num_stochastic_forward_passes: int = 1,
    ) -> None:
        """Init entropy sampling."""
        super().__init__(cfg)

        self.select_top_percent = select_top_percent
        self.query_instances: List[pd.DataFrame] = []
        self.num_stochastic_forward_passes = num_stochastic_forward_passes

    def stopping_criterion(self, pool_data: pd.DataFrame) -> bool:
        """Implement stopping criterion for active learning loop.

        Process is stopped if pool data doesn't have enough examples to sample.

        :param pool_data: current pool data
        :return: whether to stop active learning
        """
        num_new_samples = int(self.base_pool_size * self.select_top_percent)

        return len(pool_data) < num_new_samples

    def query(self, model: SentimentPipeline, pool_data: pd.DataFrame) -> pd.DataFrame:
        """Select samples to add to the current train dataset.

        New samples are the ones with the highest predictive entropy.

        :param model: model
        :param pool_data: pool data from where new chunk will be sampled
        :return: selected samples
        """
        dataloader = self._get_dataloader(pool_data)

        # [num_passes, num_samples, 3]
        prediction_probs = predict(
            model,
            dataloader,
            num_stochastic_forward_passes=self.num_stochastic_forward_passes,
        )

        if self.num_stochastic_forward_passes > 1:
            prediction_probs = prediction_probs.mean(axis=0)

        # [num_samples, 1]
        pred_entropy = entropy(prediction_probs.squeeze(), axis=1, base=2.0)

        pool_copy = deepcopy(pool_data)

        pool_copy["pred_entropy"] = pred_entropy

        num_new_samples = int(self.base_pool_size * self.select_top_percent)
        query_instance = pool_copy.sort_values("pred_entropy", ascending=False).head(
            num_new_samples
        )

        self.query_instances.append(deepcopy(query_instance))

        query_instance = query_instance.drop(columns=["pred_entropy"])

        return query_instance


class MarginSampling(ActiveLearning):
    """Margin based querying strategy.

    :param cfg: model config
    :param select_top_percent: percent of the pool data to sample on every step, defaults to 0.1
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        select_top_percent: float = 0.1,
        num_stochastic_forward_passes: int = 1,
    ) -> None:
        """Init margin sampling."""
        super().__init__(cfg)

        self.select_top_percent = select_top_percent
        self.query_instances: List[pd.DataFrame] = []
        self.num_stochastic_forward_passes = num_stochastic_forward_passes

    def stopping_criterion(self, pool_data: pd.DataFrame) -> bool:
        """Implement stopping criterion for active learning loop.

        Process is stopped if pool data doesn't have enough examples to sample.

        :param pool_data: current pool data
        :return: whether to stop active learning
        """
        num_new_samples = int(self.base_pool_size * self.select_top_percent)

        return len(pool_data) < num_new_samples

    def query(self, model: SentimentPipeline, pool_data: pd.DataFrame) -> pd.DataFrame:
        """Select samples to add to the current train dataset.

        New samples are the ones with least difference between the highest probability
        and the second highest probability.

        :param model: model
        :param pool_data: pool data from where new chunk will be sampled
        :return: selected samples
        """
        dataloader = self._get_dataloader(pool_data)

        # [num_passes, num_samples, 3]
        prediction_probs = predict(
            model,
            dataloader,
            num_stochastic_forward_passes=self.num_stochastic_forward_passes,
        )

        if self.num_stochastic_forward_passes > 1:
            prediction_probs = prediction_probs.mean(axis=0)

        # [num_samples, 1]
        # Margin between max prob and second max prob
        two_largest_probs = np.sort(prediction_probs.squeeze(), axis=1)[:, -2:]
        margins = np.diff(two_largest_probs, axis=1)

        pool_copy = deepcopy(pool_data)

        pool_copy["margins"] = margins

        num_new_samples = int(self.base_pool_size * self.select_top_percent)
        query_instance = pool_copy.sort_values("margins", ascending=True).head(
            num_new_samples
        )

        self.query_instances.append(deepcopy(query_instance))

        query_instance = query_instance.drop(columns=["margins"])

        return query_instance


class VariationRatioSampling(ActiveLearning):
    """Variation ratio querying strategy.

    :param cfg: model config
    :param select_top_percent: percent of the pool data to sample on every step, defaults to 0.1
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        select_top_percent: float = 0.1,
        num_stochastic_forward_passes: int = 1,
    ) -> None:
        """Init variation ratio sampling."""
        super().__init__(cfg)

        self.select_top_percent = select_top_percent
        self.query_instances: List[pd.DataFrame] = []
        self.num_stochastic_forward_passes = num_stochastic_forward_passes

    def stopping_criterion(self, pool_data: pd.DataFrame) -> bool:
        """Implement stopping criterion for active learning loop.

        Process is stopped if pool data doesn't have enough examples to sample.

        :param pool_data: current pool data
        :return: whether to stop active learning
        """
        num_new_samples = int(self.base_pool_size * self.select_top_percent)

        return len(pool_data) < num_new_samples

    def query(self, model: SentimentPipeline, pool_data: pd.DataFrame) -> pd.DataFrame:
        """Select samples to add to the current train dataset.

        New samples selected based on the following expression: 1 - f_x / T
        f_x - mode count (how many times the most predicted class was predicted)
        T - num_stochastic_forward_passes

        :param model: model
        :param pool_data: pool data from where new chunk will be sampled
        :return: selected samples
        """
        dataloader = self._get_dataloader(pool_data)

        # [num_passes, num_samples, 3]
        prediction_probs = predict(
            model,
            dataloader,
            num_stochastic_forward_passes=self.num_stochastic_forward_passes,
        )

        # [num_passes, num_samples]
        predicted_labels = prediction_probs.argmax(axis=-1)

        # [num_samples]
        modes_count = mode(predicted_labels, axis=0).count.squeeze()

        variation_ratio = 1.0 - modes_count / self.num_stochastic_forward_passes

        pool_copy = deepcopy(pool_data)

        pool_copy["variation_ratio"] = variation_ratio

        num_new_samples = int(self.base_pool_size * self.select_top_percent)
        query_instance = pool_copy.sort_values("variation_ratio", ascending=False).head(
            num_new_samples
        )

        self.query_instances.append(deepcopy(query_instance))

        query_instance = query_instance.drop(columns=["variation_ratio"])

        return query_instance


class BALDSampling(ActiveLearning):
    """Bayesian Active Learning by Disagreement querying strategy.

    :param cfg: model config
    :param select_top_percent: percent of the pool data to sample on every step, defaults to 0.1
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        select_top_percent: float = 0.1,
        num_stochastic_forward_passes: int = 1,
    ) -> None:
        """Init BALD sampling."""
        super().__init__(cfg)

        self.select_top_percent = select_top_percent
        self.query_instances: List[pd.DataFrame] = []
        self.num_stochastic_forward_passes = num_stochastic_forward_passes

    def stopping_criterion(self, pool_data: pd.DataFrame) -> bool:
        """Implement stopping criterion for active learning loop.

        Process is stopped if pool data doesn't have enough examples to sample.

        :param pool_data: current pool data
        :return: whether to stop active learning
        """
        num_new_samples = int(self.base_pool_size * self.select_top_percent)

        return len(pool_data) < num_new_samples

    def query(self, model: SentimentPipeline, pool_data: pd.DataFrame) -> pd.DataFrame:
        """Select samples to add to the current train dataset.

        New exaples are selected to maximize the mutual information.
        Check the details here:
        https://jacobgil.github.io/deeplearning/activelearning#learning-loss-for-active-learning

        :param model: model
        :param pool_data: pool data from where new chunk will be sampled
        :return: selected samples
        """
        dataloader = self._get_dataloader(pool_data)

        # [num_passes, num_samples, 3]
        prediction_probs = predict(
            model,
            dataloader,
            num_stochastic_forward_passes=self.num_stochastic_forward_passes,
        )

        # [num_samples]
        mutual_information = (
            entropy(prediction_probs.mean(axis=0), axis=1, base=2.0)
            - entropy(prediction_probs, axis=-1, base=2.0).sum(axis=0)
            / self.num_stochastic_forward_passes
        )

        pool_copy = deepcopy(pool_data)

        pool_copy["mutual_information"] = mutual_information.squeeze()

        num_new_samples = int(self.base_pool_size * self.select_top_percent)
        query_instance = pool_copy.sort_values(
            "mutual_information", ascending=False
        ).head(num_new_samples)

        self.query_instances.append(deepcopy(query_instance))

        query_instance = query_instance.drop(columns=["mutual_information"])

        return query_instance
