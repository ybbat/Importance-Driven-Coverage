import itertools
from typing import Callable, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .attributors import AttributorType
from .clusterers import ClustererType
from .saveloader import file_saveloader


class ImportanceDrivenCoverage:
    """
    A class for computing importance-driven coverage of a neural network.

    Args:
        model (nn.Module): The neural network model to compute coverage for.
        attributor (AttributorType): The attributor function to use for computing neuron attributions.
        clusterer (ClustererType): The clustering function to use for clustering neuron activations.

    Attributes:
        model (nn.Module): The neural network model to compute coverage for.
        attributor (AttributorType): The attributor function to use for computing neuron attributions.
        clusterer (ClustererType): The clustering function to use for clustering neuron activations.
        device (torch.device): The device to use for computations

    Methods:
        calculate(train_data, test_data, layer, n, transform=None, attributions_path=None, centroids_path=None):
            Computes the importance-driven coverage of the neural network.

        get_centroids(data, layer, indices):
            Computes the centroids of the neuron activations.

        activations(data, layer, indices, transform=None):
            Computes the neuron activations for the given data.

    """

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
        """
        Calculates the importance-driven coverage of a given layer in a neural network.

        Args:
            train_data (DataLoader): The data used to train the neural network.
            test_data (DataLoader): The data to calculate coverage for.
            layer (nn.Module): The layer for which to calculate the coverage.
            n (int): The number of most important neurons to consider.
            transform (Optional[Callable], optional): A function to transform the test data. Defaults to None.
            attributions_path (Optional[str], optional): The path to save/load attributions. Defaults to None.
            centroids_path (Optional[str], optional): The path to save/load centroids. Defaults to None.

        Returns:
            Tuple[float, set]: A tuple containing the coverage score and the set of covered combinations.
        """
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
