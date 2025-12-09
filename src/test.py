import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
import pytorch_lightning as pl
import torch

from src.modules.datamodule import DewrapDM
from src.modules.lightning_module import DewrapModule


def postprocess(result: list, out_dir: Path) -> None:
    """
    Postprocess result
    """
    results = []
    for res in result:
        preds, im_names, orig_sizes, infer_sizes = res
        preds = preds.cpu().numpy()
        df1 = pd.DataFrame(
            np.hstack([
                np.array(im_names).reshape(-1, 1),
                orig_sizes,
                infer_sizes
            ]),
            columns=['filepath', 'height_orig', 'width_orig', 'height_infer', 'width_infer']
        )
        df2 = pd.DataFrame(preds,
            columns=np.concatenate([[f'x{i+1}', f'y{i+1}'] for i in range(4)]).tolist()
        )
        df = pd.concat([df1, df2], axis=1)
        results.append(df)

    results = pd.concat(results)
    results.to_csv(out_dir / 'results.csv', index=False)


@hydra.main(config_path='../configs', config_name='config')
def test(config: DictConfig) -> None:
    """Inference function."""
    best_checkpoint_path = Path(
        config.checkpoints) / config.experiment_name / f'best_{config.project_name}_{config.experiment_name}.ckpt'

    # out_dir = Path(config.out_dir) / config.project_name / config.experiment_name
    

    model = DewrapModule.load_from_checkpoint(best_checkpoint_path, config=config.model)
    trainer = pl.Trainer(
        devices=[config.device],
    )

    datamodule = DewrapDM(config)
    # datamodule.setup('test')
    # sample = next(iter(datamodule.test_dataloader()))
    # with open(Path(config.work_dir) / 'test_sample.pkl', 'wb') as f:
    #     pickle.dump(sample, f)


    if config.get('real_path') is None:
        print(f'Test from {Path(config.data_path).name}.......')
        datamodule.setup('test')

        sample = next(iter(datamodule.test_dataloader()))
        with open(Path(config.work_dir) / 'test_sample.pkl', 'wb') as f:
            pickle.dump(sample, f)

        trainer.test(model, datamodule=datamodule)
        result = trainer.predict(model, datamodule=datamodule)
        out_dir = Path(config.out_dir) / config.project_name / config.experiment_name / Path(config.data_path).name
    else:
        print(f'Test from {Path(config.real_path).name}.......')
        datamodule.setup('real-test')
        trainer.test(model, dataloaders=datamodule.test_dataloader())
        result = trainer.predict(model, dataloaders=datamodule.predict_dataloader())
        out_dir = Path(config.out_dir) / config.project_name / config.experiment_name / Path(config.real_path).name

    out_dir.mkdir(exist_ok=True, parents=True)
    # trainer = pl.Trainer(
    #     devices=[config.device],
    # )

    # trainer.test(model, datamodule=datamodule)
    # result = trainer.predict(model, datamodule=datamodule)
    # # preds, images_names, im_size

    # Postprocess and save
    postprocess(result, out_dir)

    # Save model checkpoint for service
    # infer_model_path = Path(
    #     config.checkpoints) / config.experiment_name / f'best_segm_trace_{config.experiment_name}.pt'
    # trace_model = torch.jit.trace(model, torch.rand(1, 1, 512, 512, device='cpu'))
    # torch.jit.save(trace_model, infer_model_path)


if __name__ == '__main__':
    test()
