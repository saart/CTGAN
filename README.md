<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![PyPI Shield](https://img.shields.io/pypi/v/ctgan.svg)](https://pypi.python.org/pypi/ctgan)
[![Travis CI Shield](https://travis-ci.org/DAI-Lab/CTGAN.svg?branch=master)](https://travis-ci.org/DAI-Lab/CTGAN)
[![Downloads](https://pepy.tech/badge/ctgan)](https://pepy.tech/project/ctgan)
[![Coverage Status](https://codecov.io/gh/DAI-Lab/CTGAN/branch/master/graph/badge.svg)](https://codecov.io/gh/DAI-Lab/CTGAN)

# CTGAN

Implementation of our NeurIPS paper **Modeling Tabular data using Conditional GAN**.

CTGAN is a GAN-based data synthesizer that can generate synthetic tabular data with high fidelity.

- Free software: [MIT license](https://github.com/DAI-Lab/CTGAN/tree/master/LICENSE)
- Documentation: https://DAI-Lab.github.io/CTGAN
- Homepage: https://github.com/DAI-Lab/CTGAN

# Overview

Based on previous work ([TGAN](https://github.com/DAI-Lab/TGAN)) on synthetic data generation,
we develop a new model called CTGAN. Several major differences make CTGAN outperform TGAN.

- **Preprocessing**: CTGAN uses more sophisticated Variational Gaussian Mixture Model to detect
  modes of continuous columns.
- **Network structure**: TGAN uses LSTM to generate synthetic data column by column. CTGAN uses
  Fully-connected networks which is more efficient.
- **Features to prevent mode collapse**: We design a conditional generator and resample the
  training data to prevent model collapse on discrete columns. We use WGANGP and PacGAN to
  stabilize the training of GAN.


# Install

## Requirements

**CTGAN** has been developed and tested on [Python 3.5, 3.6 and 3.7](https://www.python.org/downloads/)

## Install from PyPI

The recommended way to installing **CTGAN** is using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install ctgan
```

This will pull and install the latest stable release from [PyPI](https://pypi.org/).

If you want to install from source or contribute to the project please read the
[Contributing Guide](https://DAI-Lab.github.io/CTGAN/contributing.html#get-started).

# Data Format

**CTGAN** expects the input data to be a table given as either a `numpy.ndarray` or a
`pandas.DataFrame` object with two types of columns:

* **Continuous Columns**: Columns that contain numerical values and which can take any value.
* **Discrete columns**: Columns that only contain a finite number of possible values, wether
these are string values or not.

This is an example of a table with 4 columns:

* A continuous column with float values
* A continuous column with integer values
* A discrete column with string values
* A discrete column with integer values

|   | A    | B   | C   | D |
|---|------|-----|-----|---|
| 0 | 0.1  | 100 | 'a' | 1 |
| 1 | -1.3 | 28  | 'b' | 2 |
| 2 | 0.3  | 14  | 'a' | 2 |
| 3 | 1.4  | 87  | 'a' | 3 |
| 4 | -0.1 | 69  | 'b' | 2 |


**NOTE**: CTGAN does not distinguish between float and integer columns, which means that it will
sample float values in all cases. If integer values are required, the outputted float values
must be rounded to integers in a later step, outside of CTGAN.

# Python Quickstart

In this short tutorial we will guide you through a series of steps that will help you
getting started with **CTGAN**.

## 1. Model the data

### Step 1: Prepare your data

Before being able to use CTGAN you will need to prepare your data as specified above.

For this example, we will be loading some data using the `ctgan.load_demo` function.

```python
from ctgan import load_demo

df = load_demo()
```

This will download a copy of the [Adult Census Dataset] as a dataframe:

|   age | workclass        |  ...  |   hours-per-week | native-country   | income   |
|-------|------------------|-------|------------------|------------------|----------|
|    39 | State-gov        |  ...  |               40 | United-States    | <=50K    |
|    50 | Self-emp-not-inc |  ...  |               13 | United-States    | <=50K    |
|    38 | Private          |  ...  |               40 | United-States    | <=50K    |
|    53 | Private          |  ...  |               40 | United-States    | <=50K    |
|    28 | Private          |  ...  |               40 | Cuba             | <=50K    |
|   ... | ...              |  ...  |              ... | ...              | ...      |

Aside from the table itself, you will need to create a list with the names of the discrete
variables.

For this example:

```python
discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]
```

### Step 2: Fit CTGAN to your data

Once you have the data ready, you need to import and create an instance of the `CTGANSynthesizer`
class and fit it passing your data and the list of discrete columns.

```python
from ctgan import CTGANSynthesizer

ctgan = CTGANSynthesizer()
ctgan.fit(data, discrete_columns)
```

This process is likely to take a long time to run.
If you want to make the process shorter, or longer, you can control the number of training epochs
that the model will be performing by adding it to the `fit` call:

```python
ctgan.fit(data, discrete_columns, epochs=1)
```

## 2. Generate synthetic data

Once the process has finished, all you need to do is call the `sample` method of your
`CTGANSynthesizer` instance indicating the number of rows that you want to generate.

```python
samples = ctgan.sample(1000)
```

# Join our community

1. If you would like to try more dataset examples, please have a look at the [examples folder](
https://github.com/DAI-Lab/CTGAN/tree/master/examples) of the repository. Please contact us
if you have a usage example that you would want to share with the community.
2. If you want to contribute to the project code, please head to the [Contributing Guide](
https://DAI-Lab.github.io/CTGAN/contributing.html#get-started) for more details about how to do it.
3. If you have any doubts, feature requests or detect an error, please [open an issue on github](
https://github.com/DAI-Lab/CTGAN/issues)
4. Also do not forget to check the [API Reference](https://DAI-Lab.github.io/CTGAN/api)!


# Citing TGAN

If you use CTGAN, please cite the following work:

- *Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni.* **Modeling Tabular data using Conditional GAN**. NeurIPS, 2019.

```LaTeX
@inproceedings{xu2019modeling,
  title={Modeling Tabular data using Conditional GAN},
  author={Xu, Lei and Skoularidou, Maria and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```
