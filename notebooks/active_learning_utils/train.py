from typing import Dict, Any, List, Tuple, Union, cast
import torch
from .pipeline import SentimentPipeline, MetricTracker
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
import numpy as np
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import Tensor


def turn_on_dropout(model: SentimentPipeline) -> SentimentPipeline:
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.training = True

    return model


@torch.no_grad()
def predict(
    model: SentimentPipeline,
    dataloader: DataLoader,
    num_stochastic_forward_passes: int = 1,
) -> np.ndarray:

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = model.to(device)
    model.eval()

    logits: List[Tensor] = []

    if num_stochastic_forward_passes > 1:
        model = turn_on_dropout(model)

        for _ in range(num_stochastic_forward_passes):

            outputs = []
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model.model(**batch)

                outputs.append(output)

            logits.append(torch.cat([p.logits for p in outputs]))
    else:
        outputs = []
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model.model(**batch)

            outputs.append(output)

        logits = [torch.cat([p.logits for p in outputs])]

    torch.cuda.empty_cache()

    # [n_passes, num_samples, num_classes]
    return torch.nn.functional.softmax(torch.stack(logits, dim=0), dim=-1).cpu().numpy()


def train_pl_model(
    cfg: Dict[str, Any],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    return_model: bool = False,
) -> Union[Dict[str, List[float]], Tuple[Dict[str, List[float]], SentimentPipeline]]:

    seed_everything(cfg["seed"])

    num_training_steps = cfg["epochs"] * len(train_dataloader)

    pipeline = SentimentPipeline(cfg, num_training_steps)

    metric_tracker = MetricTracker()

    trainer = Trainer(
        max_epochs=cfg["epochs"],
        gpus=1,
        callbacks=[metric_tracker],
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )

    trainer.fit(
        pipeline, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    if return_model:
        return metric_tracker.collection, pipeline

    return metric_tracker.collection
