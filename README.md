# Sedpack - Scalable and efficient data packing

[![Coverage Status](https://coveralls.io/repos/github/google/sedpack/badge.svg?branch=main)](https://coveralls.io/github/google/sedpack?branch=main)

[Documentation](https://google.github.io/sedpack/)

Mainly refactored from the [SCAAML](https://github.com/google/scaaml) project.

## Available components

See the documentation website:
[https://google.github.io/sedpack/](https://google.github.io/sedpack/).

## Install

### Package install

```bash
pip install sedpack
```

[TensorFlow 2.x](https://www.tensorflow.org/install) is an optional dependency
required only when using the `"tfrec"` shard format or `Dataset.as_tfdataset`:

```bash
pip install "sedpack[tf]"
```

### Development install

Development prerequisites:

-   `python3-dev` and `gcc` for [xxhash](https://pypi.org/project/xxhash/)
-   [Install Rust](https://www.rust-lang.org/tools/install) (required by
    `maturin` to build `sedpack._sedpack_rs`)

Steps:

1.  Clone the repository and create a virtual environment:

    ```bash
    git clone https://github.com/google/sedpack
    cd sedpack
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  Install pinned dependencies (optional):

    ```bash
    python3 -m pip install --require-hashes -r requirements.txt
    ```

3.  Install the package in editable mode:

    ```bash
    python3 -m pip install --editable ".[dev,tf]"
    ```

    Or for a minimal installation without TensorFlow and development tools:

    ```bash
    python3 -m pip install --editable .
    ```

4.  Rebuild the Rust extension after changes in `rust/` and run unit tests:

    ```bash
    maturin develop --release
    python -m pytest
    ```

    No tests should be skipped when TensorFlow is installed via
    `pip install -e ".[dev,tf]"`.

### Update dependencies

With an active virtual environment, regenerate `requirements.txt` and commit the
updated lock file:

```bash
uv pip compile pyproject.toml --generate-hashes -o requirements.txt
```

### Tutorial

Tutorials and documentation are available at
[https://google.github.io/sedpack/](https://google.github.io/sedpack/).

Code for the tutorials is available in the `docs/tutorials` directory. For a
"hello world" example, see the
[MNIST tutorial](https://google.github.io/sedpack/tutorials/mnist/).

## AI Agents and LLM Skills

Sedpack includes built-in documentation for AI coding assistants and LLM agent
frameworks:

-   **Using `sedpack` in your project (`SKILL.md`)**: The high-level
    `sedpack.io` API skill is bundled inside the package at
    `src/sedpack/.agents/skills/sedpack/SKILL.md` (and declared in
    `skills.sh.json` at the repository root).
    -   **Option 1 — Link from your Python environment via
        [Library Skills](https://library-skills.io/) (recommended for `.venv`
        projects)**: Scans your project's installed Python packages and creates
        a relative directory symlink in `.agents/skills/sedpack` so the skill
        always matches your installed `sedpack` version and updates
        automatically on `pip install -U sedpack`:

        ```bash
        # Interactive discovery and installation:
        uvx library-skills

        # Non-interactive install into .agents/skills/sedpack:
        uvx library-skills install --skill sedpack --yes

        # Also link into .claude/skills/sedpack, or copy files instead of symlinking:
        uvx library-skills install --skill sedpack --yes --claude
        uvx library-skills install --skill sedpack --yes --copy

        # Inspect status, check symlinks in CI, or repair after recreating .venv:
        uvx library-skills list
        uvx library-skills --check
        uvx library-skills --yes
        ```

        `library-skills` automatically detects `.venv` (`venv`, `virtualenv`,
        `uv`, `pdm`, and in-project `poetry`/`pipenv`) and Conda
        (`CONDA_PREFIX`). For external virtual environments (such as Poetry's
        default cache, Pipenv, or Hatch), pass `UV_PROJECT_ENVIRONMENT`:

        ```bash
        UV_PROJECT_ENVIRONMENT="$(poetry env info --path)" uvx library-skills install --skill sedpack --yes
        ```

    -   **Option 2 — Install from GitHub via
        [Agent Skills CLI (`npx skills`)](https://skills.sh/)**: Installs the
        skill directly from `google/sedpack` (or a local checkout) into
        project-local or user-global (`-g`) directories across 40+ coding
        agents:

        ```bash
        # Interactive project-local install from GitHub:
        npx skills add google/sedpack

        # Non-interactive install (optionally targeting a specific agent):
        npx skills add google/sedpack --skill sedpack -y
        npx skills add google/sedpack --skill sedpack --agent gemini-cli -y
        npx skills add google/sedpack --skill sedpack --agent claude-code -y

        # Global install into ~/.agents/skills/sedpack:
        npx skills add google/sedpack --skill sedpack -g -y

        # List available skills in the repo, or check/update installed skills:
        npx skills add google/sedpack --list
        npx skills check
        npx skills update
        ```

    -   **Option 3 — Link with Gemini CLI**: Link the bundled skill directory
        directly from your active Python environment (into `~/.gemini/skills` or
        project-local `.gemini/skills` with `--scope workspace`):

        ```bash
        gemini skills link "$(python -c 'import importlib.resources as r; print(r.files("sedpack") / ".agents/skills/sedpack")')"
        ```

    -   **Updating skills when `sedpack` updates (automatic vs. manual, local
        `.venv` vs. global `~/`)**:
        -   **Project-local `.venv` symlink (`uvx library-skills`) —
            Automatic**: Because `.agents/skills/sedpack` is a directory symlink
            into `site-packages/sedpack/.agents/skills/sedpack`, upgrading
            `sedpack` in-place (`pip install -U sedpack`,
            `uv sync --upgrade-package sedpack`, or `poetry update sedpack`)
            updates `SKILL.md` immediately with no extra command, and keeps each
            project isolated to its own `.venv` version.
        -   **Recreating `.venv` or upgrading Python minor versions (`3.12` ->
            `3.13`)**: If `.venv` is recreated with the *same* Python minor
            version, the relative symlink resolves again as soon as `sedpack` is
            reinstalled. If you upgrade Python minor versions (which changes
            `lib/python3.X/site-packages`), repair the symlink with:

            ```bash
            uvx library-skills --yes
            ```

        -   **Project-local file copy (`uvx library-skills --copy`) — Manual**:
            Copied files are static snapshots and do not update on
            `pip install -U sedpack`. Re-sync after upgrading `sedpack`:

            ```bash
            rm -rf .agents/skills/sedpack
            uvx library-skills install --skill sedpack --yes --copy
            ```

        -   **GitHub-installed skills (`npx skills`, local or global `-g`) —
            Manual**: `npx skills` tracks the `google/sedpack` GitHub repository
            rather than your `.venv` (and `-g` shares a single global version
            across all projects). Check for and pull updates from GitHub with:

            ```bash
            npx skills check
            npx skills update
            ```

        -   **Gemini CLI link (`gemini skills link`)**: Updates automatically on
            in-place `pip install -U sedpack` in the linked environment (run
            `/skills reload` in an active Gemini CLI session), or on `git pull`
            for an editable install (`pip install -e .`). Re-run
            `gemini skills link` if the linked `.venv` is moved or upgraded to a
            new Python minor version.

-   **Contributing to `sedpack` (`AGENTS.md`)**: Repository architecture, build
    commands, test suites, and linter rules for coding agents working on
    `sedpack` itself are documented in `AGENTS.md` at the repository root.

## Disclaimer

This is not an official Google product.
