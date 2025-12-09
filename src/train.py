from pathlib import Path
import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.modules.datamodule import DewrapDM, prepare_and_split_datasets
from src.modules.lightning_module import DewrapModule


log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config")
def train(config: DictConfig) -> None:
    """Entry point for model training."""
    pl.seed_everything(config.seed)
    logging.basicConfig(level=logging.INFO)

    data_path = Path(config.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_path}")

    # Prepare splits only if they are missing
    expected_split = data_path / "df_train.csv"
    if not expected_split.exists():
        log.info("Dataset splits not found, preparing new splits...")
        prepare_and_split_datasets(data_path, train_fraction=config.dataset.train_size, seed=config.seed)

    datamodule = DewrapDM(config)

    checkpoint_dir = Path(config.checkpoints) / config.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"best_{config.project_name}_{config.experiment_name}",
        save_top_k=1,
        monitor=config.model.checkpoint_callback.monitor_metric,
        mode=config.model.checkpoint_callback.monitor_mode,
        verbose=config.model.checkpoint_callback.verbose,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=[config.device],
        max_epochs=config.n_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=config.log_nsteps,
        default_root_dir=config.work_dir,
    )

    model = DewrapModule(config.model)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
