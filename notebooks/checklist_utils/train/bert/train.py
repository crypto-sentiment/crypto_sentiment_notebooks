from typing import Dict, Any, Tuple, Union
from .pipeline import SentimentPipeline, MetricTracker
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
import pandas as pd
from .dataset import build_dataloaders
import numpy as np
import torch


def train_model(
    cfg: Dict[str, Any],
    train_data: pd.DataFrame,
    train_labels: pd.DataFrame,
    val_data: pd.DataFrame,
    val_labels: pd.DataFrame,
    **kwargs
) -> Dict[str, Any]:

    train_dataloader, val_dataloader = build_dataloaders(
        cfg, train_data, train_labels, val_data, val_labels
    )

    return train_pl_model(cfg, train_dataloader, val_dataloader, **kwargs)


def train_pl_model(
    cfg: Dict[str, Any],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    return_predictions: bool = False,
    return_model: bool = False,
) -> Dict[str, Any]:

    seed_everything(cfg["seed"])

    pipeline = SentimentPipeline(cfg)

    metric_tracker = MetricTracker()

    trainer = Trainer(
        max_epochs=cfg["epochs"],
        gpus=1,
        callbacks=[metric_tracker],
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(
        pipeline, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    output: Dict[str, Any] = {"scores": metric_tracker.collection}

    if return_predictions:
        batch_logits = trainer.predict(pipeline, val_dataloader)

        logits = torch.cat([p.logits for p in batch_logits])

        pred_labels = torch.argmax(logits, dim=-1).numpy()

        output["pred_labels"] = pred_labels

    if return_model:
        output["model"] = pipeline

    return output
