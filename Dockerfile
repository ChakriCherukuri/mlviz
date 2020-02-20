FROM jupyter/base-notebook:63d0df23b673

USER root

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

# set the working directory
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

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

# Expose the notebook port
EXPOSE 8888

# Start the notebook server
CMD jupyter notebook --no-browser --port 8888 --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.disable_check_xsrf=True
