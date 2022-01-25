Notebooks containing theoretical and applied machine learning algorithms/models. All the examples are built using the jupyter widget libraries ([ipywidgets](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html) and [bqplot](https://bqplot.readthedocs.io/en/latest/)). [voila](https://github.com/voila-dashboards/voila) is used to render the notebooks as code less dashboards.

View the live `mlviz` gallery live at https://chakricherukuri.github.io/mlviz/

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

`http://localhost:8888/voila/render/notebooks/Index.ipynb`
