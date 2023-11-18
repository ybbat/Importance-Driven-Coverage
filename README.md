# Importance Driven Coverage

## Overview

This project is an implementation of the Importance Driven Coverage metric described in Importance-Driven Deep Learning System Testing [[1]](#1)

## Usage

Basic usage consists of initialising the `ImportanceDrivenCoverage` using the model under test, an attribution method, and a clustering method.
The calculate method can then be used, providing a training dataloader, test dataloader, the layer to test, and the number of important neurons to select.

```python
attributor = attributors.CaptumLRPAttributor()
clusterer = clusterers.KMeansClustererSimpleSilhouette()

idc = coverage.ImportanceDrivenCoverage(model, attributor, clusterer)

score, combs = idc.calculate(train_loader, test_loader, layer, n)
```

Demonstration of usage is shown in notebooks/usage.ipynb

## References
<a id="1">[1]</a> Importance-Driven Deep Learning System Testing; Gerasimou, Eniser, Sen, And Cakan [(2020)](https://arxiv.org/abs/2002.03433)