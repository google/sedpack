# Sedpack - Scalable and efficient data packing

Mainly refactored from the [SCAAML](https://github.com/google/scaaml) project.

## Available components

- TODO

## Install

### Dependencies

To use this library you need to have a working version of [TensorFlow 2.x](https://www.tensorflow.org/install).

### Dataset install

#### Development install

1. Clone the repository: `git clone https://github.com/google/sedpack`
2. Install dependencies: `python3 -m pip install --require-hashes -r requirements.txt`
3. Install the package in development mode: `python3 -m pip install --editable .` (short `pip install -e .` or legacy `python setup.py develop`)

### Update dependencies

Make sure to have: `sudo apt install python3 python3-pip python3-venv` and
activated the virtual environment.

Install requirements: `pip install --require-hashes -r base-tooling-requirements.txt`

Update: `pip-compile requirements.in --generate-hashes --upgrade` and commit requirements.txt.

#### Package install

`pip install sedpack`

### Tutorial

Tutorials available in the docs/tutorials/ directory.  For a "hello world" see
[docs/tutorials/quick_start/mnist_save.py](https://github.com/google/sedpack/blob/main/docs/tutorials/quick_start/mnist_save.py) and
[docs/tutorials/quick_start/mnist_save.py](https://github.com/google/sedpack/blob/main/docs/tutorials/quick_start/mnist_read.py).

## Disclaimer

This is not an official Google product.
