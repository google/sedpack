# Sedpack - Scalable and efficient data packing

[![Coverage Status](https://coveralls.io/repos/github/google/sedpack/badge.svg?branch=main)](https://coveralls.io/github/google/sedpack?branch=main)

Mainly refactored from the [SCAAML](https://github.com/google/scaaml) project.

## Available components

-   TODO

## Install

### Dependencies

To use this library you need to have a working version of [TensorFlow
2.x](https://www.tensorflow.org/install).

Development dependencies:

-   cmake for Rust flate2 with [zlib-ng](https://docs.rs/flate2/latest/flate2/)

-   python-dev and gcc for [xxhash](https://pypi.org/project/xxhash/)

### Dataset install

#### Development install

1.  Clone the repository: `git clone https://github.com/google/sedpack`
2.  Install dependencies: `python3 -m pip install --require-hashes -r requirements.txt`
3.  Install the package in development mode: `python3 -m pip install --editable
    .` (short `pip install -e .` or legacy `python setup.py develop`)

#### Rust install

-   Activate your Python virtual environment
-   [Install Rust](https://www.rust-lang.org/tools/install)
-   Run `maturin develop --release`
-   Run `python -m pytest` from the project root directory -- no tests should
    be skipped

### Update dependencies

Make sure to have: `sudo apt install python3 python3-pip python3-venv` and
activated the virtual environment.

Install requirements: `pip install --require-hashes -r base-tooling-requirements.txt`

Update: `pip-compile pyproject.toml --generate-hashes --upgrade` and commit requirements.txt.

#### Package install

`pip install sedpack`

### Tutorial

Tutorials available in the docs/tutorials/ directory.  For a "hello world" see
[docs/tutorials/quick_start/mnist_save.py](https://github.com/google/sedpack/blob/main/docs/tutorials/quick_start/mnist_save.py)
and
[docs/tutorials/quick_start/mnist_save.py](https://github.com/google/sedpack/blob/main/docs/tutorials/quick_start/mnist_read.py).

## Disclaimer

This is not an official Google product.
