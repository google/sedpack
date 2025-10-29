# Sedpack - Scalable and efficient data packing

[![Coverage Status](https://coveralls.io/repos/github/google/sedpack/badge.svg?branch=main)](https://coveralls.io/github/google/sedpack?branch=main)

[Documentation](https://google.github.io/sedpack/)

Mainly refactored from the [SCAAML](https://github.com/google/scaaml) project.

## Available components

See the documentation website:
[https://google.github.io/sedpack/](https://google.github.io/sedpack/).

## Install

### Dependencies

To use this library you need to have a working version of [TensorFlow
2.x](https://www.tensorflow.org/install).

Development dependencies:

-   python-dev and gcc for [xxhash](https://pypi.org/project/xxhash/)

### Dataset install

#### Development install

1.  Clone the repository: `git clone https://github.com/google/sedpack`
2.  Install dependencies: `python3 -m pip install --require-hashes -r requirements/Linux_py3.13_requirements.txt`
3.  Install the package in development mode: `python3 -m pip install --editable
    .` (short `pip install -e .` or legacy `python setup.py develop`)

#### Rust install

-   Activate your Python virtual environment
-   [Install Rust](https://www.rust-lang.org/tools/install)
-   Run `maturin develop --release`
-   Run `python -m pytest` from the project root directory -- no tests should
    be skipped

### Update dependencies

Run `tools/pip_compile.sh` (which will install [uv](https://docs.astral.sh/uv/guides/install-python/)).

#### Package install

`pip install sedpack`

### Tutorial

A tutorial and documentation is available at
[https://google.github.io/sedpack/](https://google.github.io/sedpack/).

Code for the tutorials is available in the `docs/tutorials` directory. For a
"hello world" see
[https://google.github.io/sedpack/tutorials/mnist/](https://google.github.io/sedpack/tutorials/mnist/).

## Disclaimer

This is not an official Google product.
