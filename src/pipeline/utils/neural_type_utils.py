from collections import defaultdict
from typing import Dict, List, Optional

from pipeline.neural_types import AxisKind, NeuralType


def get_io_names(types: Optional[Dict[str, NeuralType]], disabled_names: List[str]) -> List[str]:
    if types is None:
        return []
    names = list(types.keys())
    for name in disabled_names:
        if name in names:
            names.remove(name)
    return names


def extract_dynamic_axes(name: str, ntype: NeuralType):
    """
        This method will extract BATCH and TIME dimension ids from each provided input/output name argument.

        For example, if module/model accepts argument named "input_signal" with type corresponding to [Batch, Time, Dim]
        shape, then the returned result should contain "input_signal" -> [0, 1] because Batch and Time are dynamic axes
        as they can change from call to call during inference.

        Args:
            name: Name of input or output parameter
            ntype: Corresponding Neural Type

        Returns:

        """

    def unpack_nested_neural_type(neural_type):
        if type(neural_type) in (list, tuple):
            return unpack_nested_neural_type(neural_type[0])
        return neural_type

    dynamic_axes = defaultdict(list)
    if type(ntype) in (list, tuple):
        ntype = unpack_nested_neural_type(ntype)

    if ntype.axes:
        for ind, axis in enumerate(ntype.axes):
            if axis.kind in [AxisKind.Batch, AxisKind.Time, AxisKind.Width, AxisKind.Height]:
                dynamic_axes[name].append(ind)
    return dynamic_axes


def get_dynamic_axes(types, names):
    dynamic_axes = defaultdict(list)
    if names is not None:
        for name in names:
            if name in types:
                dynamic_axes.update(extract_dynamic_axes(name, types[name]))
    return dynamic_axes
