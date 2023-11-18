"""
This module contains example clusterers, a user may define their
own clusterers and use them with the IDC class.
An clusterers is a function that has arguments:
    - activations: Subset of activations of a layer for top n neurons
                   in shape (cases, n)
and returns
    - centroids: List of centroids for each neuron
This is also defined by the ClustererType type alias, a variable of
this type is provided to IDC class to calculate the coverage.
"""
from typing import Callable

import numpy as np
import torch
from sklearn import cluster

ClustererType = Callable[[torch.Tensor], list[list[float]]]


def euclidean_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b) ** 2) ** 0.5


def simple_silhouette_score(
    x: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    dist_func: Callable = euclidean_dist,
) -> float:
    """
    Compute the simple silhouette score for a clustering given the data, labels, and centroids.
    Regular silhouette score can be used but it is much more computationally expensive.
    https://en.wikipedia.org/wiki/Silhouette_(clustering)#Simplified_Silhouette_and_Medoid_Silhouette

    Args:
        x (np.ndarray): The data to cluster.
        labels (np.ndarray): The cluster labels for each data point.
        centroids (np.ndarray): The centroids of each cluster.
        dist_func (Callable, optional): The distance function to use. Defaults to dist.

    Returns:
        float: The silhouette score for the clustering.
    """
    center_x = centroids[labels]  # centroids of clusters that contains each object
    A = dist_func(
        x.squeeze(), center_x
    )  # distances between object and containing cluster

    mask = center_x[:, np.newaxis] != centroids
    masked = np.broadcast_to(centroids, mask.shape)[
        mask
    ]  # centres of clusters that doesnt contain the object at index
    center_not_x = np.reshape(
        masked, (len(x), len(centroids) - 1)
    )  # shape of x, then -1 the amount of clusters, since array contains all centroids that x is not contained in
    dists = dist_func(x, center_not_x)  # distances between x and other centres
    B = np.min(dists, axis=1)  # distances between x and closest other centre

    S = (B - A) / np.maximum(A, B)
    return np.mean(S)


def KMeansClustererSimpleSilhouette(maxk=8) -> ClustererType:
    """
    Returns a function that clusters activations using KMeans clustering with the simple silhouette score as the metric.
    The function takes in a tensor of activations and returns a list of lists of cluster centroids, where each inner list corresponds to a neuron.

    Args:
        maxk (int): The maximum number of clusters to consider.

    Returns:
        function: A function that takes in a tensor of activations and returns a list of lists of cluster centroids.
    """

    def func(activations: torch.Tensor) -> list[list[float]]:
        ret = []
        for neuron in range(activations.shape[1]):
            acts = activations[:, neuron].numpy(force=True).reshape(-1, 1)

            best_score = -1.0
            best_centroids = np.array([])
            for k in range(2, maxk + 1):
                kmeans = cluster.KMeans(n_clusters=k, n_init="auto")
                clustered = kmeans.fit_predict(acts)
                centroids = kmeans.cluster_centers_.squeeze()
                if (
                    score := simple_silhouette_score(acts, clustered, centroids)
                ) > best_score:
                    best_score = score
                    best_centroids = centroids
            ret.append(best_centroids.tolist())
        return ret

    return func
