import os
import sys

sys.path.append("src")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import time
from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import wandb
from src.datasets.cls_dataset import ClassifierDataset
from src.zoo import ResNet3dCSN
from src.zoo.utils import _initialize_weights

ROOT = "/mnt/nvme13TB/RAW_DATASETS/rsna-2023/"
ORGAN = "spleen"
SHAPE = (128, 128, 64)
FOLD_I = 2


class ClassifierResNet3dCSN2P1D(nn.Module):
    def __init__(
        self,
        encoder="r50ir",
        pool="max",
        norm_eval=False,
        num_classes=3,
        head_dropout=0.4,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(1024, num_classes),
        )
        self.avg_pool = (
            nn.AdaptiveAvgPool3d((1, 1, 1))
            if pool == "avg"
            else nn.AdaptiveMaxPool3d((1, 1, 1))
        )
        self.dropout = nn.Dropout(0.5)
        _initialize_weights(self)

        self.backbone = ResNet3dCSN(
            pretrained2d=False,
            pretrained=None,
            depth=int(encoder[1:-2]),
            with_pool2=False,
            bottleneck_mode=encoder[-2:],
            norm_eval=norm_eval,
            zero_init_residual=False,
        )

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1, 1)[:, :, :, :, :]
        x = self.backbone(x)[-1]
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.flatten(1)
        x = self.head(x)
        return x


class LitClassifier3d(pl.LightningModule):
    def __init__(
        self, config, num_classes, encoder="r50ir", pool="avg", pretrain="", **kwargs
    ):
        super().__init__()
        self.args = config
        self.learning_rate = config.lr
        self.save_hyperparameters()
        self.model = ClassifierResNet3dCSN2P1D(
            encoder=encoder, num_classes=num_classes, pool=pool
        )
        if pretrain:
            self.model.load_state_dict(torch.load(pretrain), strict=False)
            print(f"resumed from {pretrain}")

        self.loss_fn = nn.CrossEntropyLoss()
        self.metric = nn.CrossEntropyLoss(weight=torch.Tensor([1, 2, 4]))

    def forward(self, batch):
        return self.model(batch["image"])

    def training_step(self, batch, batch_idx):
        logits = self.model(batch["image"]).float()
        labels = batch["label"].float()

        loss = self.loss_fn(logits, labels)
        metric_loss = self.metric(logits, labels)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_metric_loss",
            metric_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.model(batch["image"]).float()
        labels = batch["label"].float()

        loss = self.loss_fn(logits, labels)
        metric = self.metric(logits, labels)

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        self.log(
            "val_metric",
            metric,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-2
        )
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                factor=0.5,
                patience=15,
                threshold=0.07,
            ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "train_loss",
            "name": "lr/reduce_on_plateau",
        }
        return [optimizer], [lr_scheduler]


def train(CFG):
    train_dataset = ClassifierDataset(
        mode="train",
        target=CFG.target,
        fold=CFG.fold,
        path_to_images=CFG.path_to_images,
        path_to_meta_csv=CFG.path_to_meta_csv,
        shape=CFG.shape,
    )

    val_dataset = ClassifierDataset(
        mode="eval",
        target=CFG.target,
        fold=CFG.fold,
        path_to_images=CFG.path_to_images,
        path_to_meta_csv=CFG.path_to_meta_csv,
        shape=CFG.shape,
    )

    batch_size = CFG.batch_size

    targets = train_dataset.meta[["healthy", "low", "high"]].values.argmax(1)

    class_counts = np.bincount(targets)

    class_weights = 1.0 / class_counts
    weights = class_weights[targets]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=CFG.num_workes,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=CFG.num_workes
    )
    pretrain = "weights/ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20200803-fc66ce8d.pth"

    encoder_name = "r152ir"
    pool = "max"
    num_classes = 3

    wandb_name = f"{ORGAN}_{encoder_name}_fold{CFG.fold}_shape:{CFG.shape}"

    wandb.login()
    wandb_logger = WandbLogger(
        project=CFG.wandb_project, name=wandb_name, group=CFG.group
    )

    wandb.config.update(
        {k: v for k, v in CFG.__dict__.items() if not k.startswith("__")}
    )

    model = LitClassifier3d(
        config=CFG,
        num_classes=num_classes,
        encoder=encoder_name,
        pool=pool,
        pretrain=pretrain,
    )
    # resume = f"pybooks/{ORGAN}/{ORGAN}_fold{FOLD_I}.pt"
    # model.model.load_state_dict(torch.load(resume), strict=False)
    # print(f"resume from {resume}")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    checkpoint_callback = ModelCheckpoint(
        dirpath=CFG.checkpoints_dir,
        monitor="val_loss",
        save_top_k=3,
        filename=f"{ORGAN}/"
        + wandb_name
        + "_"
        + str(CFG.fold)
        + "_{epoch}-{val_loss:.3f}",
        mode="min",
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        default_root_dir=CFG.default_root_dir,
        accelerator="gpu",
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=CFG.max_epochs,
        check_val_every_n_epoch=3,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


@dataclass
class CFG_CLS:
    # dataset
    path_to_images: str = f"{ROOT}/{ORGAN}_crops"
    path_to_meta_csv: str = f"{ROOT}/train_{ORGAN}.csv"
    fold: int = 0
    shape: tuple = SHAPE  # whd
    target: str = "multi"
    num_workes: int = 6
    group: str = "Timur"
    description: str = f"albu augs {ORGAN} adamw"

    # training
    num_classes: int = 3
    batch_size: int = 15
    wandb_project: str = f"RSNA_classification_{ORGAN}"
    default_root_dir: str = "cls"
    checkpoints_dir: str = "cls/checkpoints"
    lr: float = 3e-4
    weight_decay: float = 1e-2
    max_epochs: int = 300


# Note: you need to define `ROOT` somewhere above the dataclass for the f-strings to work.

if __name__ == "__main__":
    CFG = CFG_CLS(fold=FOLD_I)
    train(CFG)
    time.sleep(60)
