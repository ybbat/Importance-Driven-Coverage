# Importance Driven Coverage

## Overview

This project is an implementation of the Importance Driven Coverage metric described in Importance-Driven Deep Learning System Testing [[1]](#1).

IDC uses a training set to find the most important neurons in a layer, it then clusters the activation values for these neurons. The score is then calculated by exploring what combination of these clusters are activated by the testing set.

## Components

The importance driven coverage calculation requires two components, an attributor and a clusterer. This project provides some in attributors.py and clusterers.py, but users may provide their own to fit their needs.

An attributor is a function with arguments:
* model
* dataloader
* layer

And returns a tensor containing attributions for the neurons within that layer.

A clusterer is a function with arguments:
* activations (the activation values for training cases for each specifc neuron in shape (cases, n))

And returns a list of the centroids for each important neuron.


## Usage

Basic usage consists of initialising the `ImportanceDrivenCoverage` using the model under test, an attribution method, and a clustering method.
The calculate method can then be used, providing a training dataloader, test dataloader, the layer to test, and the number of important neurons to select.

```python
attributor = attributors.CaptumLRPAttributor()
clusterer = clusterers.KMeansClustererSimpleSilhouette()

idc = coverage.ImportanceDrivenCoverage(model, attributor, clusterer)

score, combs = idc.calculate(train_loader, test_loader, layer, n)
```

Demonstration of usage using the LeNet5 model is shown in notebooks/usage.ipynb

## References
<a id="1">[1]</a> Importance-Driven Deep Learning System Testing; Gerasimou, Eniser, Sen, And Cakan [(2020)](https://arxiv.org/abs/2002.03433)