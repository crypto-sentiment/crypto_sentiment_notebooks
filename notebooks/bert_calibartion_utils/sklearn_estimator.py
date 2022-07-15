from typing import Any, Dict
from sklearn.base import BaseEstimator, ClassifierMixin
from .pipeline import SentimentPipeline, MetricTracker
from .utils import build_object
import numpy as np
import torch
from torch import Tensor
from pytorch_lightning import Trainer, seed_everything
from .dataset import FinNewsDataset
from torch.utils.data import DataLoader
import os
from sklearn.utils.multiclass import unique_labels

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BertWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, cfg: Dict[str, Any], load_pretrained: bool = False) -> None:

        self.cfg = cfg
        self.load_pretrained = load_pretrained

        self.tokenizer = build_object(cfg["tokenizer"], is_hugging_face=True)

        if load_pretrained:
            self.model = SentimentPipeline.load_from_checkpoint(cfg["checkpoint"])
        else:
            self.model = SentimentPipeline(cfg)

        self.device = torch.device(cfg["device"])

    def _prepare_data(self, X: np.ndarray) -> Dict[str, Tensor]:
        encodings = self.tokenizer(X, **self.cfg["tokenizer"]["call_params"])

        return {
            key: torch.tensor(val, device=self.device) for key, val in encodings.items()
        }

    @torch.no_grad()
    def _predict(self, X: np.ndarray) -> np.ndarray:
        self.model = self.model.to(self.device)
        self.model.eval()

        data = self._prepare_data(X)

        logits = self.model.model(**data).logits

        return torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = self._predict(X).argmax(-1)

        torch.cuda.empty_cache()

        return pred

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pred_probas = self._predict(X)

        torch.cuda.empty_cache()

        return pred_probas

    def fit(self, X, y):
        seed_everything(self.cfg["seed"])
        self.model = self.model.to(self.device)
        self.model.train()

        metric_tracker = MetricTracker()

        dataset = FinNewsDataset(news=X, labels=y)

        train_dataloader = DataLoader(
            dataset,
            batch_size=self.cfg["train_batch_size"],
            shuffle=True,
            num_workers=self.cfg["num_workers"],
        )

        trainer = Trainer(
            max_epochs=self.cfg["epochs"],
            gpus=1,
            callbacks=[metric_tracker],
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            logger=False,
        )

        trainer.fit(self.model, train_dataloaders=train_dataloader)

        self.classes_ = unique_labels(y)

        torch.cuda.empty_cache()

        return self

    def get_params(self, deep=True):
        return {"cfg": self.cfg, "load_pretrained": self.load_pretrained}
