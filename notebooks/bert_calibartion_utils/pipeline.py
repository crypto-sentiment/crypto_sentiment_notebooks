from typing import Any, Dict

import pytorch_lightning as pl
import torch
from datasets import load_metric
from pytorch_lightning import Callback
from torch import Tensor
from transformers.modeling_outputs import SequenceClassifierOutput

from .utils import build_object


class SentimentPipeline(pl.LightningModule):
    """Class for training text classification models"""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.cfg = cfg
        self.model = build_object(cfg["model"], is_hugging_face=True)
        self.criterion = build_object(cfg["criterion"])
        self.metric = load_metric("accuracy")

        self.tokenizer = build_object(cfg["tokenizer"], is_hugging_face=True)

    def configure_optimizers(self):
        optimizer = build_object(self.cfg["optimizer"], params=self.model.parameters())

        if "T_max" in self.cfg["scheduler"]["params"]:
            self.cfg["scheduler"]["params"][
                "T_max"
            ] = self.trainer.estimated_stepping_batches

        lr_scheduler = build_object(self.cfg["scheduler"], optimizer=optimizer)

        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.cfg["scheduler_interval"],
            "frequency": self.cfg["scheduler_frequency"],
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, batch: Dict[str, Tensor]):
        encodings = self.tokenizer(
            batch["news"], **self.cfg["tokenizer"]["call_params"]
        )

        item = {key: val.to(self.device) for key, val in encodings.items()}
        item["labels"] = batch["labels"].to(self.device)

        return self.model(**item)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

        outputs: SequenceClassifierOutput = self.forward(batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        self.metric.add_batch(predictions=predictions, references=batch["labels"])

        loss = self.criterion(logits, batch["labels"])

        self.log(
            "train_acc",
            self.metric.compute()["accuracy"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )

        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:

        with torch.no_grad():
            outputs = self.forward(batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        self.metric.add_batch(predictions=predictions, references=batch["labels"])

        loss = self.criterion(logits, batch["labels"])

        self.log(
            "val_acc",
            self.metric.compute()["accuracy"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )


class MetricTracker(Callback):
    def __init__(self):
        self.collection = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def _log_metrics(self, trainer, stage: str = "train"):
        for key in (f"{stage}_acc", f"{stage}_loss"):
            self.collection[key].append(trainer.callback_metrics[key].item())

    def on_validation_epoch_end(self, trainer, module):
        self._log_metrics(trainer, "val")

    def on_train_epoch_end(self, trainer, module):
        self._log_metrics(trainer, "train")
