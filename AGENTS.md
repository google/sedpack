# AGENTS.md

Sedpack is a scalable and efficient sharded dataset storage and streaming
library: a Python package (`src/sedpack/`), a Rust native extension (`rust/`,
built with PyO3 and `maturin` as `sedpack._sedpack_rs`), a user-facing LLM
agent skill (`src/sedpack/.agents/skills/sedpack/SKILL.md`), tutorials
(`docs/tutorials/`), a `pytest` suite (`tests/`), and an Astro Starlight
documentation website (`website/`).

If you are using `sedpack` in another project rather than developing `sedpack`
itself, read `src/sedpack/.agents/skills/sedpack/SKILL.md` for the high-level
`sedpack.io` API guide.

## Before Opening a Pull Request

-   Read the contribution guidelines in `docs/contributing.md`.
-   Keep each pull request focused on a single problem with a concise
    description of what changed, why, and how it was tested. Avoid unrelated
    refactoring, renames, formatting churn, version bumps, or `requirements.txt`
    lock file churn unless explicitly requested.
-   Every tracked `.py` and `.sh` file (except generated FlatBuffers files under
    `src/sedpack/io/flatbuffer/shardfile/`) must include the Apache 2.0
    copyright header. Verify with `./tools/check_copyright.sh`.

## Working in the Repository

-   **Build and editable install**: Activate the virtual environment, install
    dependencies (`pip install --editable ".[dev,tf]"`), and rebuild the Rust
    extension after any change in `rust/` via `maturin develop --release`.
-   **Unit tests**: Run `python -m pytest` from the repository root. When
    TensorFlow is installed (`".[dev,tf]"`), no tests should be skipped.
-   **Python formatting (`yapf`)**: Configured in `.style.yapf`. Check
    formatting with `yapf --recursive --diff .` and format in-place with
    `yapf --recursive --in-place .`.
-   **Python type checking (`mypy`)**: Configured in `mypy.ini` (strict mode for
    `sedpack.*`). Run `PYTHONPATH=./src mypy --no-color-output src docs`.
-   **Python linting (`pylint`)**: Configured in `.pylintrc`. Run
    `./tools/run_pylint.sh` (which runs `pylint *.py src docs`).
-   **Rust checks**: From `rust/`, run `cargo +nightly fmt -- --check`,
    `cargo +nightly clippy --all-targets`, and `cargo test`.
-   **Markdown linting (`markdownlint`)**: All `*.md` and `**/*.md` files are
    checked against `.markdownlint.json`:
    -   Wrap prose, list items, and block quotes at 80 characters (120 for
        headings; tables and fenced code blocks are exempt).
    -   Use a hyphen followed by 3 spaces for unordered list items and a number
        plus period followed by 2 spaces for ordered list items, with 4-space
        indentation for nested lists and continuation lines.
    -   Always specify a language tag on fenced code blocks (`python`, `bash`,
        `text`, `json`, `toml`).
-   **Spell checking (`cspell`)**: Checked in CI against `cspell.json` and
    `project-words.txt`. If a change introduces legitimate domain terms or
    identifiers not recognized by `cspell`, add them in sorted order to
    `project-words.txt`.
-   **Generated files and skill synchronization**:
    -   Never hand-edit generated FlatBuffers bindings
        (`src/sedpack/io/flatbuffer/shardfile/*.py` or
        `rust/src/shard_generated.rs`); regenerate them from
        `src/sedpack/io/flatbuffer/shard.fbs`.
    -   When modifying the public `sedpack.io` API, update
        `src/sedpack/.agents/skills/sedpack/SKILL.md` so the bundled agent skill
        stays accurate.
