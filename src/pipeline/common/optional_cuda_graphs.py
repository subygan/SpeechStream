import abc
from typing import Optional

import torch.nn as nn


class WithOptionalCudaGraphs(abc.ABC):
    """
    Abstract interface for modules with CUDA graphs.
    Allows to enable/disable CUDA graphs on the fly.
    """

    @classmethod
    def disable_cuda_graphs_recursive(cls, module: nn.Module, attribute_path: Optional[str] = None):
        """
        Disable CUDA graphs Enable CUDA graphs, finding submodule recursively.

        Args:
            module: instance of nn.Module
            attribute_path: field containing instance of WithOptionalCudaGraphs
                   E.g., "decoding.decoding" means that "<module>.decoding.decoding" are checked.
                   If None, "<module>" is checked.
        """
        attributes = attribute_path.split(".") if attribute_path else []

        for name, submodule in module.named_modules():
            object_to_check = submodule
            try:
                # recursively get attribute by iterating attribute_path
                for attribute in attributes:
                    object_to_check = getattr(object_to_check, attribute)
            except AttributeError:
                continue  # loop over modules, no attribute

            if isinstance(object_to_check, cls):
                object_to_check.disable_cuda_graphs()
                print(f"Disabled CUDA graphs for module {type(submodule)}" + ".".join([name] + attributes))

    @classmethod
    def enable_cuda_graphs_recursive(cls, module: nn.Module, attribute_path: Optional[str] = None):
        """
        Enable CUDA graphs, finding submodule recursively

        Args:
            module: instance of nn.Module
            attribute_path: field containing instance of WithOptionalCudaGraphs
                   E.g., "decoding.decoding" means that "<module>.decoding.decoding" are checked.
                   If None, "<module>" is checked.
        """
        attributes = attribute_path.split(".") if attribute_path else []

        for name, submodule in module.named_modules():
            object_to_check = submodule
            try:
                # recursively get attribute by iterating attribute_path
                for attribute in attributes:
                    object_to_check = getattr(object_to_check, attribute)
            except AttributeError:
                continue  # loop over modules, no attribute

            if isinstance(object_to_check, cls):
                object_to_check.maybe_enable_cuda_graphs()
                print(f"Enabled CUDA graphs for module {type(submodule)}" + ".".join([name] + attributes))

    @abc.abstractmethod
    def disable_cuda_graphs(self):
        """Disable (maybe temporary) CUDA graphs"""
        raise NotImplementedError

    @abc.abstractmethod
    def maybe_enable_cuda_graphs(self):
        """Enable CUDA graphs if all conditions met"""
        raise NotImplementedError

