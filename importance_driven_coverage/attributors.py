"""
This module contains example attributors, a user may define their
own attributors and use them with the IDC class.
An attributor is a function that has arguments:
    - model: A neural network model
    - data: A dataloader
    - layer: A layer of the model
and returns
    - attributions: A tensor of attributions for the given layer
This is also defined by the AttributorType type alias, a variable of
this type is provided to IDC class to calculate the coverage.
"""
from typing import Any, Callable, Optional

import captum
import torch
from torch import nn
from torch.utils.data import DataLoader

AttributorType = Callable[[nn.Module, DataLoader, nn.Module], torch.Tensor]


def CaptumLRPAttributor(
    cons_args: Optional[tuple[Any, ...]] = (),
    cons_kwargs: Optional[dict[Any, Any]] = {},
    attribute_args: Optional[tuple[Any, ...]] = (),
    attribute_kwargs: Optional[dict[Any, Any]] = {},
    device: Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
) -> AttributorType:
    """
    Returns a function that computes Layer-wise Relevance Propagation (LRP) attributions
    for a given model, dataloader and layer using Captum library.

    Args:
        cons_args (Optional[Tuple[Any, ...]], optional): Arguments to be passed to the constructor of the attribution method. Defaults to None.
        cons_kwargs (Optional[Dict[Any, Any]], optional): Keyword arguments to be passed to the constructor of the attribution method. Defaults to None.
        attribute_args (Optional[Tuple[Any, ...]], optional): Arguments to be passed to the attribute method of the attribution method. Defaults to None.
        attribute_kwargs (Optional[Dict[Any, Any]], optional): Keyword arguments to be passed to the attribute method of the attribution method. Defaults to None.
        device (Optional[str], optional): Device to be used for the computation. Defaults to "cuda" if available else "cpu".

    Returns:
        AttributorType: A function that takes a model, data and layer as input and returns the LRP attributions.
    """

    def func(model: nn.Module, data: DataLoader, layer: nn.Module) -> torch.Tensor:
        attr = captum.attr.LayerLRP(model, layer, *cons_args, **cons_kwargs)

        ret = None

        for X, y in data:
            X, y = X.to(device), y.to(device)

            attribution = attr.attribute(
                X, *attribute_args, **attribute_kwargs, target=y
            )
            attribution = torch.sum(attribution, dim=0)
            ret = torch.zeros_like(attribution) if ret is None else ret
            ret += attribution

        return ret if ret is not None else torch.Tensor()

    return func
