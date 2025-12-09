import pytorch_lightning as pl
import torch
import torch.nn as nn
import typing as tp
from omegaconf import DictConfig
from timm import create_model

from src.modules.losses import get_losses
from src.modules.metrics import get_metrics
from src.modules.io import load_object


class RegressionHead(nn.Module):
    """
        Class of Regression head with "out_features" outputs
    """

    def __init__(self, in_features: int, hidden_dim: int = 128, out_dim: int = 8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False)
        )

    def forward(self, x):
        return self.head(x)


class DewrapModel(nn.Module):
    """
        Class of Dewrap Model with Regression head
    """

    def __init__(
            self,
            model_name: str,
            pretrained: bool,
            out_dim: int = 8,
            hidden_dim: int = 128,
    ):
        super().__init__()

        self.backbone = create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=0,
            # global_pool='avg',
        )

        in_features = self.backbone.num_features
        self.regression_head = RegressionHead(in_features, hidden_dim, out_dim)

    def forward(self, x):
        features = self.backbone(x)
        output = self.regression_head(features)
        return output


class DewrapModule(pl.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

        # Create model
        self.model = DewrapModel(
            model_name=self.config.model_args.model_name,
            pretrained=self.config.model_args.pretrained,
            out_dim=self.config.model_args.num_outputs,
            hidden_dim=128,
        )
        print(self.model)
        # Set loss functions
        self.losses = get_losses(self.config.losses)

        # Set metrics
        metrics = get_metrics()
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        self.save_hyperparameters(dict(self.config))

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = load_object(self.config.optimizer)(
            self.model.parameters(), **self.config.optimizer_kwargs,
        )
        scheduler = load_object(self.config.scheduler)(optimizer, **self.config.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.config.checkpoint_callback.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx):
        images, images_names, im_size, gt = batch
        gt = torch.stack(gt)
        images = torch.stack(images)
        preds = self(images)
        loss = self.calculate_loss(preds, gt, 'train_')
        self.log('val_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, images_names, im_size, gt = batch
        gt = torch.stack(gt)
        images = torch.stack(images)
        preds = self(images)
        loss = self.calculate_loss(preds, gt, 'val_')
        metrics = self.val_metrics(preds, gt)
        self.log('val_mae', metrics['val_mae'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, images_names, im_size, gt = batch
        gt = torch.stack(gt)
        images = torch.stack(images)
        preds = self(images)
        self.test_metrics(preds, gt)
        # return preds, images_names, im_size

    def predict_step(self, batch, batch_idx):
        images, images_names, orig_size, infer_size = batch
        images = torch.stack(images)
        preds = self(images)

        return preds, images_names, orig_size, infer_size

    def on_validation_epoch_start(self) -> None:
        self.val_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), on_epoch=True, on_step=False)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute(), on_epoch=True, on_step=False)

    def calculate_loss(
            self,
            pred_masks_logits: torch.Tensor,
            gt_masks: torch.Tensor,
            prefix: str,
    ) -> torch.Tensor:
        total_loss = 0
        for _loss in self.losses:
            loss = _loss.loss(pred_masks_logits, gt_masks)
            total_loss += _loss.weight * loss
            self.log(f'{prefix}{_loss.name}_loss', loss.item())
        self.log(f'{prefix}total_loss', total_loss.item())
        return total_loss
