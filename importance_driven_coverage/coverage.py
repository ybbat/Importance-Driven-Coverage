import itertools
from typing import Callable, Optional, Tuple

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
        transform: Optional[Callable] = None,
        attributions_path: Optional[str] = None,
        centroids_path: Optional[str] = None,
    ) -> Tuple[float, set]:
        attributor = (
            self.attributor
            if attributions_path is None
            else file_saveloader(attributions_path, (layer))(self.attributor)
        )

        attributions = attributor(self.model, train_data, layer)

        indices = attributions.ravel().topk(n).indices

        centroids_func = (
            self.get_centroids
            if centroids_path is None
            else file_saveloader(centroids_path, (layer, n))(self.get_centroids)
        )

        centroids = centroids_func(attributions, train_data, layer, indices)

        activations = self.activations(test_data, self.model, indices, transform).to(
            "cpu"
        )

        covered_combs = set()

        Y = range(len(test_data.dataset))  # type: ignore

        for y in Y:
            covered = []
            for i in range(n):
                a = activations[y, i]

                closest = min(centroids[i], key=lambda x: abs(x - a.squeeze()))
                covered.append(closest)

            covered_combs.add(tuple(covered))

        all_combs = set(itertools.product(*centroids))
        return float(len(covered_combs)) / len(all_combs), covered_combs

    def get_centroids(
        self,
        data: DataLoader,
        layer: nn.Module,
        indices: torch.Tensor,
    ) -> list[list[float]]:
        acts = self.activations(data, layer, indices)

        centroids = self.clusterer(acts)
        return centroids

    def activations(
        self,
        data: DataLoader,
        layer: nn.Module,
        indices: torch.Tensor,
        transform: Optional[Callable] = None,
    ) -> torch.Tensor:
        acts = []

        def hook(module, input, output):
            acts.append(output.detach())

        try:
            handle = layer.register_forward_hook(hook)
            for X, y in data:
                X, y = X.to(self.device), y.to(self.device)

                if transform is not None:
                    X = transform(X)

                self.model(X)
        finally:
            handle.remove()

        all_acts = torch.cat(acts, dim=0)
        return all_acts.reshape(all_acts.shape[0], -1)[:, indices]
