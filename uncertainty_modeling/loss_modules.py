from typing import Optional, Callable

import torch
from torch import nn, Tensor


class SoftDiceLoss(nn.Module):
    def __init__(
        self,
        apply_nonlin: Optional[Callable[[Tensor], Tensor]] = None,
        do_bg: bool = True,
        smooth: float = 1e-5,
        smooth_in_nom: bool = True,
    ):
        """
        Compute soft dice loss

        Args:
            apply_nonlin: if logits will be provided in the forward pass,
                this defines nonlinear which needs to be applied to convert
                them to probabilities
            do_bg: include background class in dice loss
            smooth: smoothing for denominator
            smooth_in_nom: smoothing for nominator
        """
        super(SoftDiceLoss, self).__init__()
        self.do_bg = do_bg
        self.apply_nonlin = apply_nonlin
        if smooth_in_nom:
            self.smooth_in_nom = smooth
        else:
            self.smooth_in_nom = 0
        self.smooth = smooth

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute loss

        Args:
            x: predictions of network
            y: ground truth

        Returns:
            Tensor: soft dice loss
        """
        y = y.long()
        shp_x = x.shape
        shp_y = y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        if len(shp_x) != len(shp_y):
            y = y.view((shp_y[0], 1, *shp_y[1:]))

        # now x and y should have shape (B, C, X, Y(, Z))) and (B, 1, X, Y(, Z))), respectively
        y_onehot = torch.zeros_like(x)
        y_onehot.scatter_(1, y, 1)

        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]

        l = soft_dice(x, y_onehot, self.smooth, self.smooth_in_nom)
        return l


def soft_dice(
    net_output: Tensor, gt: Tensor, smooth: float = 1.0, smooth_in_nom: float = 1.0
) -> Tensor:
    """
    Soft dice functional interface

    Args:
        net_output: predicted probabilities
        gt: ground truth (need to be one hot encoded)
        smooth: smoothing in denominator
        smooth_in_nom: smoothing in nominator

    Returns:
        Tensor: soft dice loss
    """
    axes = tuple(range(2, len(net_output.size())))
    intersect = (net_output * gt).sum(axes, keepdim=False)
    denom = (net_output + gt).sum(axes, keepdim=False)
    result = (-((2 * intersect + smooth_in_nom) / (denom + smooth))).mean()
    return result
