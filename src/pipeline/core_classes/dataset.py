from dataclasses import dataclass
from typing import Optional

from torch.utils import data

from pipeline.core_classes.common import Serialization, Typing, typecheck

__all__ = ['Dataset', 'IterableDataset']


class Dataset(data.Dataset, Typing, Serialization):
    """Dataset with output ports

    Please Note: Subclasses of IterableDataset should *not* implement input_types.
    """

    def _collate_fn(self, batch):
        """
        A default implementation of a collation function.
        Users should override this method to define custom data loaders.
        """
        return data.dataloader.default_collate(batch)

    @typecheck()
    def collate_fn(self, batch):
        """
        This is the method that user pass as functor to DataLoader.
        The method optionally performs neural type checking and add types to the outputs.

        Please note, subclasses of Dataset should not implement `input_types`.

        Usage:

        .. code-block:: python

            dataloader = torch.utils.data.DataLoader(
                    ....,
                    collate_fn=dataset.collate_fn,
                    ....
            )

        Returns:
            Collated batch, with or without types.
        """
        if self.input_types is not None:
            raise TypeError("Datasets should not implement `input_types` as they are not checked")

        # Simply forward the inner `_collate_fn`
        return self._collate_fn(batch)


class IterableDataset(data.IterableDataset, Typing, Serialization):
    """Iterable Dataset with output ports

    Please Note: Subclasses of IterableDataset should *not* implement input_types.
    """

    def _collate_fn(self, batch):
        """
        A default implementation of a collation function.
        Users should override this method to define custom data loaders.
        """
        return data.dataloader.default_collate(batch)

    @typecheck()
    def collate_fn(self, batch):
        """
        This is the method that user pass as functor to DataLoader.
        The method optionally performs neural type checking and add types to the outputs.

        # Usage:
        dataloader = torch.utils.data.DataLoader(
                ....,
                collate_fn=dataset.collate_fn,
                ....
        )

        Returns:
            Collated batch, with or without types.
        """
        if self.input_types is not None:
            raise TypeError("Datasets should not implement `input_types` as they are not checked")

        # Simply forward the inner `_collate_fn`
        return self._collate_fn(batch)


@dataclass
class DatasetConfig:
    """

    """

    # ...
    batch_size: int = 32
    drop_last: bool = False
    shuffle: bool = False
    num_workers: Optional[int] = 0
    pin_memory: bool = True
