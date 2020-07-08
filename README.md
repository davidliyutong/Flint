# Project Flint

## Introduction
This is a machine learning framework for single precision models.

## Requirements
**Python packages**

- cupy-cuda (to run the model on GPU, need to specifiy a version)
- json (to save the model parameters)
- numpy

**Notice:** `CuPy` won't work unless properly configured CUDA toolkits. This project contains a `requirements.pip` that uses cupy-cuda101, which works with CUDA=10.1. However, you might need to adjust it.

## Tested  Environment
- Ubuntu 18.04 LTS
- Python 3.8
- CUDA=10.1
- cuDNN

and

- Windows 10 1909
- Python 3.8
- CUDA=10.2
- cuDNN

