from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .attributors import AttributorType
from .clusterers import ClustererType
from .saveloader import file_saveloader


class ImportanceDrivenCoverage:
    def __init__(
        self, model: nn.Module, attributor: AttributorType, clusterer: ClustererType
    ) -> None:
        self.model = model
        self.attributor = attributor
        self.clusterer = clusterer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def calculate(
        self,
        train_data: DataLoader,
        test_data: DataLoader,
        layer: nn.Module,
        n: int,
        attributions_path: Optional[str] = None,
        centroids_path: Optional[str] = None,
    ) -> Tuple[float, set]:
        attributor = (
            self.attributor
            if attributions_path is None
            else file_saveloader(attributions_path, (layer))(self.attributor)
        )

        attributions = attributor(self.model, train_data, layer)

        centroids_func = (
            self.get_centroids
            if centroids_path is None
            else file_saveloader(centroids_path, (layer, n))(self.get_centroids)
        )

        centroids = centroids_func(attributions, train_data, layer, n)

        return (0.0, set())

    def get_centroids(
        self, attributions: torch.Tensor, data: DataLoader, layer: nn.Module, n: int
    ) -> list[list[float]]:
        top_n = attributions.ravel().topk(n).indices
        acts = self.get_activations(data, layer, top_n)

        centroids = self.clusterer(acts)
        return centroids

    def get_activations(
        self, data: DataLoader, layer: nn.Module, indices: torch.Tensor
    ) -> torch.Tensor:
        acts = []

        def hook(module, input, output):
            acts.append(output.detach())

        try:
            handle = layer.register_forward_hook(hook)
            for X, y in data:
                X, y = X.to(self.device), y.to(self.device)
                self.model(X)
        finally:
            handle.remove()

        all_acts = torch.cat(acts, dim=0)
        return all_acts.reshape(all_acts.shape[0], -1)[:, indices]
