import logging
import os
import json
from typing import Optional, Tuple, List
from pathlib import Path
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch

from src.modules.transforms import get_train_transforms, get_valid_transforms
from src.modules.dataset import DewrapDataset


def collate_fn(batch):
    return tuple(zip(*batch))


class DewrapDM(LightningDataModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.data_path = Path(config.data_path)
        self.real_path = Path(config.real_path) if config.get('real_path') is not None else None

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.train_transforms = get_train_transforms(width=self.config.dataset.image_width,
                                                     height=self.config.dataset.image_height,
                                                     )
        self.val_transforms = get_valid_transforms(width=self.config.dataset.image_width,
                                                   height=self.config.dataset.image_height,
                                                  )


    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datamodule
        """

        if stage == 'fit':
            train_names, train_coords = read_df(self.data_path, 'train')
            valid_names, valid_coords = read_df(self.data_path, 'valid')

            self.train_dataset = DewrapDataset(
                train_names,
                train_coords,
                transforms=self.train_transforms
            )
            self.valid_dataset = DewrapDataset(
                valid_names,
                valid_coords,
                transforms=self.val_transforms
            )

        elif stage == 'test':
            test_names, test_coords = read_df(self.data_path, 'test')
            self.test_dataset = DewrapDataset(
                test_names,
                test_coords,
                transforms=self.val_transforms
            )
            self.predict_dataset = DewrapDataset(
                test_names,
                transforms=self.val_transforms
            )
        
        elif stage == 'real-test':
            test_names, test_coords = read_df(self.real_path)
            self.test_dataset = DewrapDataset(
                test_names,
                test_coords,
                transforms=self.val_transforms
            )

            self.predict_dataset = DewrapDataset(
                test_names,
                transforms=self.val_transforms
            )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.n_workers,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.n_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.n_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=collate_fn
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.n_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=collate_fn
        )

def check_ds(df):
    "Check num of coords == 4"
    df['coord_len'] = df['coords'].apply(lambda x: len(json.loads(x)))
    df = df[df['coord_len'] == 4]
    return df.drop('coord_len', axis=1)

def prepare_and_split_datasets(data_path: Path, train_fraction: float = 0.8, seed: int = 0) -> None:
    df = pd.read_csv(data_path / 'annotations.csv', sep=',')
    df = df.drop_duplicates()
    logging.info(f'Deduplicated dataset: {len(df)}')
    # Drop samples with num of coords != 4
    df = check_ds(df)
    logging.info(f'Final dataset: {len(df)}')

    df['filepath'] = df['filepath'].apply(lambda x: data_path / Path(x).parent.name / Path(x).name)

    # Деление на трейн\валид\тест
    np.random.seed(seed)
    indicies = np.arange(df.shape[0])
    np.random.shuffle(indicies)

    train_idx = indicies[:int(df.shape[0] * train_fraction)]
    valid_idx = indicies[train_idx.size:]
    test_idx = valid_idx[valid_idx.size // 2:]
    valid_idx = valid_idx[-test_idx.size:]

    train_df = df.iloc[train_idx]
    valid_df = df.iloc[valid_idx]
    test_df = df.iloc[test_idx]

    logging.info(f'Train dataset: {len(train_df)}')
    logging.info(f'Valid dataset: {len(valid_df)}')
    logging.info(f'Test dataset: {len(test_df)}')

    # Normalize coords
    # train_df, valid_df, test_df = min_max_normalize(train_df, valid_df, test_df)

    train_df.to_csv(os.path.join(data_path, 'df_train.csv'), index=False)
    valid_df.to_csv(os.path.join(data_path, 'df_valid.csv'), index=False)
    test_df.to_csv(os.path.join(data_path, 'df_test.csv'), index=False)
    logging.info('Datasets successfully saved!')


def read_df(data_path: Path, mode: str = None) -> Tuple[List, List]:
    """
    Read df with annotations to list of image paths and coords list
    """
    
    if mode is not None:
        df = pd.read_csv(data_path / f'df_{mode}.csv')
        image_names = df['filepath'].to_list()
    else:
        df = pd.read_csv(data_path / 'annotations.csv')
        df = check_ds(df)
        df['filepath'] = df['filepath'].apply(lambda x: data_path / Path(x).parent.parent.name / Path(x).parent.name / Path(x).name)
        image_names = df['filepath'].apply(lambda x: data_path / Path(x).parent.parent.name / Path(x).parent.name / Path(x).name).to_list()


    image_names = df['filepath'].to_list()
    coords = df['coords'].values
    return image_names, coords

def min_max_normalize(train_df, valid_df, test_df):
    """
    Normalizes a list or NumPy array of numbers to the range [0, 1] using Min-Max scaling.
    """
    coords = np.array([json.loads(coord) for coord in train_df['coords'].values]).reshape(-1, 8)
    scaler = MinMaxScaler().fit(coords)
    coords = [coord.tolist() for coord in scaler.transform(coords)]
    train_df['coords'] = coords

    coords = np.array([json.loads(coord) for coord in valid_df['coords'].values]).reshape(-1, 8)
    coords = [coord.tolist() for coord in scaler.transform(coords)]
    valid_df['coords'] = coords

    coords = np.array([json.loads(coord) for coord in test_df['coords'].values]).reshape(-1, 8)
    coords = [coord.tolist() for coord in scaler.transform(coords)]
    test_df['coords'] = coords

    return train_df, valid_df, test_df

