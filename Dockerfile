ARG BASE_CONTAINER=jupyter/base-notebook
FROM $BASE_CONTAINER

#Set the working directory
WORKDIR /home/jovyan/

LABEL version=".1"
LABEL description="Jupyter notebook with support for scientific libraries and interactive widgets"
LABEL maintainer="Chakri Cherukuri <chakri.v.cherukuri@gmail.com>"

RUN conda install -c conda-forge --quiet --yes \
    'numpy=1.16.4' \
    'scipy' \
    'pandas' \
    'scikit-learn' \
    'ipywidgets' \
    'bqplot' \
    'voila' && \
    jupyter serverextension enable voila --sys-prefix && \
    pip install tensorflow==2.0.0-rc1 && \
    rm -rf work

# Expose the notebook port
EXPOSE 8888

# Start the notebook server
CMD jupyter notebook --no-browser --port 8888 --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.disable_check_xsrf=True
