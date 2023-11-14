from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.utils.data import DataLoader

import captum


class Attributor(ABC):
    @abstractmethod
    def get_attributions(
        self, model: nn.Module, data: DataLoader, layer: nn.Module, *args, **kwargs
    ) -> torch.Tensor:
        pass


class CaptumLRP(Attributor):
    def get_attributions(
        self, model: nn.Module, data: DataLoader, layer: nn.Module, *args, **kwargs
    ) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        attr = captum.attr.LayerLRP(model, layer, *args, **kwargs)

        ret = None

        for X, y in data:
            X, y = X.to(device), y.to(device)

            attribution = attr.attribute(X, target=y)
            attribution = torch.sum(attribution, dim=0)
            ret = torch.zeros_like(attribution) if ret is None else ret
            ret += attribution

        return ret if ret is not None else torch.Tensor()
