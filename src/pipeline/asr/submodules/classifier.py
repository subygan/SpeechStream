from typing import Dict, Optional

import torch
from torch import nn as nn

from pipeline.asr.transformer_utils import transformer_weights_init
from pipeline.core_classes.exportable import Exportable
from pipeline.core_classes.neural_module import NeuralModule
from pipeline.neural_types import ChannelType, NeuralType

__all__ = ['Classifier']


class Classifier(NeuralModule, Exportable):
    """
    A baseclass for modules to perform various classification tasks.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns definitions of module input ports.
        We implement it here since all NLP classifiers have the same inputs
        """
        return {"hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    def __init__(self, hidden_size: int, dropout: float = 0.0,) -> None:
        """
        Initializes the Classifier base module.
        Args:
            hidden_size: the size of the hidden dimension
            dropout: dropout to apply to the input hidden states
        """
        super().__init__()
        self._hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

    def post_init(self, use_transformer_init: bool):
        """
        Common post-processing to be called at the end of concrete Classifiers init methods
        Args:
          use_transformer_init : whether or not to apply transformer_weights_init
        """
        if use_transformer_init:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        sample = next(self.parameters())
        example = torch.randn(max_batch, max_dim, self._hidden_size).to(sample.device).to(sample.dtype)
        return tuple([example])

    def save_to(self, save_path: str):
        """
        Saves the module to the specified path.
        Args:
            save_path: Path to where to save the module.
        """
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """
        Restores the module from the specified path.
        Args:
            restore_path: Path to restore the module from.
        """
        pass
