# Project Flint

## Caracteristics
- All models are optimized for `float32` inputs

## Requirements
#### Tested system
##### **Hardware:**
- **Processor:** i7 9700F
- **RAM:** 16G
- **Graphics Card:** RTX 2060Super

##### **Environment**
- Ubuntu 18.04 LTS
- Python 3.8
- CUDA=10.1
- cuDNN

or

- Windows 10 1909
- Python 3.8
- CUDA=10.2
- cuDNN

#### **Python packages**
- cupy-cuda (to run the model on GPU, need to specifiy a version)
- json (to save the model parameters)
- numpy

**Notice** Cupy won't work unless properly configured CUDA toolkits. This project contains a `requirements.pip` that uses cupy-cuda101, which works with CUDA=10.1. However, you might need to adjust it.
