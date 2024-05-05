from contextlib import contextmanager

import torch
from torch.nn import Module

from pipeline.core_classes.common import FileIO, Serialization, Typing

__all__ = ['NeuralModule']


class NeuralModule(Module, Typing, Serialization, FileIO):
    """
    Abstract class offering interface shared between all PyTorch Neural Modules.
    """

    @property
    def num_weights(self):
        """
        Utility property that returns the total number of parameters of NeuralModule.
        """
        return self._num_weights()

    @torch.jit.ignore
    def _num_weights(self):
        num: int = 0
        for p in self.parameters():
            if p.requires_grad:
                num += p.numel()
        return num

    def input_example(self, max_batch=None, max_dim=None):
        """
        Override this method if random inputs won't work
        Returns:
            A tuple sample of valid input data.
        """

        return None

    def freeze(self) -> None:
        r"""
        Freeze all params for inference.
        """
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def unfreeze(self) -> None:
        """
        Unfreeze all parameters for training.
        """
        for param in self.parameters():
            param.requires_grad = True

        self.train()

    @contextmanager
    def as_frozen(self):
        """
        Context manager which temporarily freezes a module, yields control and finally unfreezes the module.
        """
        training_mode = self.training
        grad_map = {}
        for pname, param in self.named_parameters():
            grad_map[pname] = param.requires_grad

        self.freeze()
        try:
            yield
        finally:
            self.unfreeze()

            for pname, param in self.named_parameters():
                param.requires_grad = grad_map[pname]

            if training_mode:
                self.train()
            else:
                self.eval()
