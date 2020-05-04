# mlviz
[![Binder](https://mybinder.org/badge_logo.svg)](https://hub.gke.mybinder.org/user/chakricherukuri-mlviz-x642xzch/notebooks/Index.ipynb)

Visualizations of machine learning algorithms/models using interactive widgets ([ipywidgets](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html) and [bqplot](https://bqplot.readthedocs.io/en/latest/)) in the Jupyter notebook

## Setup
The provided Dockerfile can be used to build a docker image (assuming docker is already installed) and launch Jupyter notebooks.

Instructions to set up the environment and run the jupyter notebook:

* Build the docker image

`docker build -t jupyter_widgets:v1 .` (don't forget the dot at the end!)

* Start the notebook server

`docker run -p 8888:8888 -v "$PWD":/home/jovyan jupyter_widgets:v1`

* Access notebooks using the following link

`http://localhost:8888`
