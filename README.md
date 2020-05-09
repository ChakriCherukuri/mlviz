# mlviz
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ChakriCherukuri/mlviz/master?filepath=Index.ipynb)

Visualizations of machine learning algorithms/models using interactive widgets ([ipywidgets](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html) and [bqplot](https://bqplot.readthedocs.io/en/latest/)) in the Jupyter notebook

## Setup
### Docker
The provided Dockerfile can be used to build a docker image (assuming docker is already installed) and launch Jupyter notebooks.

Instructions to set up the environment and run the jupyter notebook:

* Build the docker image

`docker build -t jupyter_widgets:v1 .` (don't forget the dot at the end!)

* Start the notebook server

`docker run -p 8888:8888 -v "$PWD":/home/jovyan jupyter_widgets:v1`

* Access the index notebook using the following link

`http://localhost:8888/notebooks/Index.ipynb`

### Conda Environment
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
* Access the index notebook using the following link

`http://localhost:8888/notebooks/Index.ipynb`
