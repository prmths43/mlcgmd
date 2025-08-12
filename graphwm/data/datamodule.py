import random
from typing import Optional, Sequence

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from graphwm.data.utils import dict_collate_fn


def worker_init_fn(worker_id: int):
    """
    DataLoader worker init function.

    Ensures different numpy and python random seeds for each worker
    to avoid duplicate augmentations.
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class PLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

        # instantiate datasets lazily in setup
        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None
        self.collate_fn = dict_collate_fn

    def prepare_data(self) -> None:
        # Optional: download datasets if needed
        pass

    def setup(self, stage: Optional[str] = None):
        if stage is None or stage == "fit":
            self.train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.val_datasets = [
                hydra.utils.instantiate(cfg) for cfg in self.datasets.val
            ]

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(cfg) for cfg in self.datasets.test
            ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size.train,
            shuffle=True,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                batch_size=self.batch_size.val,
                shuffle=True,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
                collate_fn=self.collate_fn,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                batch_size=self.batch_size.test,
                shuffle=True,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
                collate_fn=self.collate_fn,
            )
            for dataset in self.test_datasets
        ]

    def init_data(self):
        data_dict = self.collate_fn([self.train_dataset[0]])
        return {k: v.cuda() for k, v in data_dict.items()}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"datasets={self.datasets}, "
            f"num_workers={self.num_workers}, "
            f"batch_size={self.batch_size})"
        )
