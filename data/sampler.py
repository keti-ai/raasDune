"""
Adapted from PyTorch's DistributedSampler from here
https://github.com/pytorch/pytorch/blob/c2637a7b2656d95712078532c2bc2dd72c4143ff/torch/utils/data/distributed.py
"""

import logging
import math
from typing import Iterator, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler


_T_co = TypeVar("_T_co", covariant=True)
logger = logging.getLogger()


class InfiniteDistributedSampler(Sampler[_T_co]):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available.")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available.")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0  # Epoch counter
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        # Calculate number of samples per replica
        dataset_length = len(self.dataset)
        if self.drop_last and dataset_length % self.num_replicas != 0:
            # Split to nearest length that is evenly divisible
            self.num_samples = math.floor(dataset_length / self.num_replicas)
        else:
            self.num_samples = math.ceil(dataset_length / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[_T_co]:

        while True:
            if self.shuffle:
                # Deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))

            if not self.drop_last:
                # Add extra samples to make it evenly divisible
                padding_size = self.total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[
                        :padding_size
                    ]
            else:
                # Remove tail of data to make it evenly divisible
                indices = indices[: self.total_size]
            assert len(indices) == self.total_size

            # Subsample
            indices = indices[self.rank : self.total_size : self.num_replicas]
            assert len(indices) == self.num_samples

            # Yield indices for the current epoch
            for idx in indices:
                yield idx

            self.epoch += 1  # Move to the next epoch

            logging.info(
                "{} epoch set to {} (rank: {})".format(
                    self.__class__.__name__, self.epoch, self.rank
                )
            )

    def __len__(self) -> int:
        # Return the number of samples per epoch
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different random ordering
        for each epoch. Otherwise, the next iteration of this sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
