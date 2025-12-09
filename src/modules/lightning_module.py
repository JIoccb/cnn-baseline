import torch
import typing as tp
from omegaconf import DictConfig
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

from src.modules.io import load_object
from src.modules.losses import get_losses
from src.modules.metrics import get_metrics


class SegmModule(pl.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

        # Init model. If a fully-qualified ``model_class`` is provided, load it
        # dynamically; otherwise fall back to the SMP architecture name.
        if "model_class" in self.config:
            ModelClass = load_object(self.config.model_class)
            self.model = ModelClass(**self.config.model_args)
        else:
            Net = getattr(smp, self.config.model_arch)
            self.model = Net(
                **self.config.model_args
            )

        # Set metrics
        metrics = get_metrics(**self.config.metrics_kwargs)
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        # Set losses
        self.losses = get_losses(self.config.losses)

        # Save hparams
        self.save_hyperparameters(dict(self.config))

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, images_names, im_size, gt = batch
        gt = torch.stack(gt)
        images = torch.stack(images)
        preds = self(images)
        pred_logits = self._extract_logits(preds)
        loss = self.calculate_loss(pred_logits, gt, 'train_')
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, images_names, im_size, gt = batch
        gt = torch.stack(gt)
        images = torch.stack(images)
        preds = self(images)
        pred_logits = self._extract_logits(preds)
        loss = self.calculate_loss(pred_logits, gt, 'val_')
        metrics = self.val_metrics(pred_logits, gt)
        self.log('val_f1', metrics['val_f1'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', metrics['val_iou'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, images_names, im_size, gt = batch
        gt = torch.stack(gt)
        images = torch.stack(images)
        preds = self(images)
        pred_logits = self._extract_logits(preds)
        self.test_metrics(pred_logits, gt)

    def predict_step(self, batch, batch_idx):
        images, images_names, orig_size, infer_size = batch
        images = torch.stack(images)
        preds = torch.sigmoid(self._extract_logits(self(images)))

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

    @staticmethod
    def _extract_logits(preds: tp.Union[torch.Tensor, tp.Tuple, tp.List, tp.Dict]) -> torch.Tensor:
        """Handle models that return auxiliary outputs (e.g., coords).

        For tuple/list outputs the first element is assumed to be the mask
        logits. Dict outputs are expected to have ``'logits'`` key. Otherwise
        the prediction is returned unchanged.
        """
        if isinstance(preds, (list, tuple)):
            return preds[0]
        if isinstance(preds, dict) and 'logits' in preds:
            return preds['logits']
        return preds
