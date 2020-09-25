# Improving Semi-Supervised Learning for Remaining Useful Lifetime Estimation Through Self-Supervision

This is the companion repository for the paper "*Improving Semi-Supervised Learning for Remaining Useful Lifetime Estimation Through Self-Supervision*".
It is currently available as a manuscript on [arXiv](https://arxiv.org/abs/2108.08721) and under review at the RESS journal.

## Installation

### Environment
This project uses Python 3.7 and pipenv to manage its environment. Install pipenv as:

```
pip install -U pipenv
```

You can set up this project in a virtual environment afterwards with:

```shell
cd <PROJECT_ROOT>
pipenv install --skip-lock
```

Please make sure that the appropriate Nvidia driver, CUDA 10.2 and cuDNN are installed on your system.

### Data

Download the NASA C-MAPSS dataset [here](https://ti.arc.nasa.gov/c/6/) and extract the content to ``<PROJECT_ROOT>/data/CMAPSS``.

### Verify Installation

Test the installation by running the unit tests:

```shell
pipenv shell
export PYTHONPATH=$PYTHONPATH:./src
python -m unittest -v
```

## Usage

### Concerning Ray

This project uses ``ray`` to parallelize the training.
It expects a GPU and at least 3 CPU cores to run one training instance and will try to put two runs on one GPU if more CPU cores are available.
If you want to customize these requirements, please edit the ``ray.remote`` function decorators in the file ``run_semi_supervised.py``.

### Basic Usage

To start reproducing the results, enter the source directory of the project and open the pipenv shell if you haven't already:

```shell
pipenv shell
cd <PROJECT_ROOT>/src
```

To reproduce all baseline experiments call:

```shell
python reproduce_baseline.py 2021
```

To reproduce each of the SSL methods call:

```shell
python reproduce_semi_supervised.py 2021 --mode metric
python reproduce_semi_supervised.py autoencoder --mode autoencoder
python reproduce_semi_supervised.py rbm --mode rbm
```

### Advanced Usage

If you want to modify any of the experiments, you can start with the hyperparameter configurations.
The ``config`` folder contains JSON configs for each experiment.

If you want to run an autoencoder or self-supervised pre-training experiment, you can use the ``run_pretraining.py`` script.
Pre-training experiments with RBMs can be run with the ``run_rbm.py`` script.
Please consult the scripts help page for their usage with ``python <SCRIPT> --help``.