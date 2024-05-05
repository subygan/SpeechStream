import torch
from torch import Tensor


def masked_instance_norm(
    input: Tensor, mask: Tensor, weight: Tensor, bias: Tensor, momentum: float, eps: float = 1e-5,
) -> Tensor:
    r"""Applies Masked Instance Normalization for each channel in each data sample in a batch.

    See :class:`~MaskedInstanceNorm1d` for details.
    """
    lengths = mask.sum((-1,))
    mean = (input * mask).sum((-1,)) / lengths  # (N, C)
    var = (((input - mean[(..., None)]) * mask) ** 2).sum((-1,)) / lengths  # (N, C)
    out = (input - mean[(..., None)]) / torch.sqrt(var[(..., None)] + eps)  # (N, C, ...)
    out = out * weight[None, :][(..., None)] + bias[None, :][(..., None)]

    return out


class MaskedInstanceNorm1d(torch.nn.InstanceNorm1d):
    r"""Applies Instance Normalization over a masked 3D input
    (a mini-batch of 1D inputs with additional channel dimension)..

    See documentation of :class:`~torch.nn.InstanceNorm1d` for details.

    Shape:
        - Input: :math:`(N, C, L)`
        - Mask: :math:`(N, 1, L)`
        - Output: :math:`(N, C, L)` (same shape as input)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        super(MaskedInstanceNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input: Tensor, mask: Tensor) -> Tensor:
        return masked_instance_norm(input, mask, self.weight, self.bias, self.momentum, self.eps,)