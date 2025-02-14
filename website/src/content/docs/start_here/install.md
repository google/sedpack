---
title: Sedpack Installation
description: Sedpack Installation
---

Installation of the Sedpack Python package.

## Python Package Index

All one should need to install the package from
[PyPI](https://pypi.org/project/sedpack/) is:

```bash
pip install sedpack
```

Note that this is the latest stable version of the package.
If you want the bleeding edge features install from source instead.

## Installing from Source

One can always opt for installation from the source.

Development dependencies:

-   cmake for Rust flate2 with [zlib-ng](https://docs.rs/flate2/latest/flate2/)
-   python-dev and gcc for [xxhash](https://pypi.org/project/xxhash/)
-   [Install Rust](https://www.rust-lang.org/tools/install)

```bash
git clone github.com/google/sedpack/  # Clone the repository
python3 -m venv my_env  # Create Python virtual environment
source my_env/bin/activate  # Activate your virtual environment
cd sedpack/  # Change directory to the cloned git repository
python3 -m pip install --require-hashes -r requirements.txt  # Install dependencies
python3 -m pip install --editable '.[dev]'  # Install sedpack with development dependencies
maturin develop --release  # Rebuild Rust library after a change
python -m pytest  # Run unit-tests, none should be skipped
```
