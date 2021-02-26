# mlviz
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ChakriCherukuri/mlviz/master?urlpath=Index.ipynb)

Notebooks containing theoretical and applied machine learning algorithms/models. All the examples are built using widget libraries ([ipywidgets](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html), [bqplot](https://bqplot.readthedocs.io/en/latest/) and [voila](https://github.com/voila-dashboards/voila)) running in the Jupyter notebook. The notebooks can be run by setting up the conda environment (see below) and starting the notebook server. The notebooks can be also be run as codeless dashboards using voila (see below).

## Environment Setup (for running the notebooks)
If you have [miniconda](https://docs.conda.io/en/latest/miniconda.html) distribution, then do the following steps to start the jupyter notebook:

* create conda env called mlviz (one time setup)
```console
$ conda env create -f environment.yml
```
* activate mlviz conda env
```console
$ conda activate mlviz
```
* start jupyter notebook server
```console
$ jupyter notebook
```
* Access the index to all the dashboards using the following link

`http://localhost:8888/notebooks/Index.ipynb`

## Voila dashboards

Any notebook can be rendered as a voila dashboard by clicking the voila menu button in the notebook. The index page of dashboards can be accessed using the following link

`http://localhost:8888/voila/render/Index.ipynb`

Live voila dashboards (running on mybinder) can be accessed [here](https://mybinder.org/v2/gh/ChakriCherukuri/mlviz/master?urlpath=voila%2Frender%2FIndex.ipynb). 

### Table Of Contents
* Data distributions
    * Datasaurus Dozen
    * Univariate Gaussian Distribution
* Unsupervised learning
    * Low dimensional representations
        * IRIS
        * MNIST
    * Clustering
        * K-Means
* Supervised learning
    * Linear Regression
    * Perceptron
    * Kernel Regression
    * Quantile Regression
    * Gradient Descent

* Bayesian Optimization
    * Gaussian Process Regression
    * Acquisition Functions and Bayesian Optimization
