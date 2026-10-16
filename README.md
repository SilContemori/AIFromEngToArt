# StyleLoop

A university project for **AI From Engineering to Arts**.

The planned application will classify an input image's artistic style with a
pretrained CNN, generate an image from that style and a user-provided subject
with a pretrained diffusion model, and classify the result to evaluate style
consistency. A simple Gradio interface is planned.

## Phase 1: project setup

Only the project skeleton, directory configuration, and smoke tests are
implemented. Dataset processing, training, prediction, generation, consistency
evaluation, and the interface are not implemented. Importing the modules does
not download models, load data, or run computation.

## Files and responsibilities

| Path | Responsibility |
| --- | --- |
| `data/` | Local datasets (not committed). |
| `models/` | Saved model checkpoints (not committed). |
| `outputs/` | Generated images and evaluation results (not committed). |
| `config.py` | All configurable parameters; currently project and directory paths. |
| `dataset.py` | Future data loading, preprocessing, and dataset splitting. |
| `train.py` | Future CNN transfer learning and checkpoint saving. |
| `classifier.py` | Future CNN loading and style prediction. |
| `generator.py` | Future diffusion generation conditioned on style and subject. |
| `pipeline.py` | Future classify–generate–classify workflow and consistency evaluation. |
| `app.py` | Future Gradio interface. |
| `pyproject.toml` | Project metadata and dependencies; none needed for Phase 1. |
| `.python-version` | Python version used by uv. |
| `uv.lock` | Dependency lockfile maintained by uv. |
| `tests/test_smoke.py` | Import and configuration smoke tests. |
| `AGENTS.md` | Development rules and phase boundaries. |
| `.gitignore` | Excludes local environments, caches, datasets, and model/output files. |

The existing `main.py` is a PyCharm sample, not a StyleLoop entry point.

## Setup and verification

Use uv to manage the Python environment and dependencies. The project selects
Python 3.12 through `.python-version`. With uv installed, run from the repository root:

```sh
uv sync
uv run python -m unittest discover -s tests -v
uv run python -c "import config, dataset, train, classifier, generator, pipeline, app; print(config.PROJECT_ROOT)"
```

`uv sync` prepares `.venv` and updates `uv.lock`; activation is not required
when using `uv run`. Add dependencies with `uv add` only in approved future phases.

The tests should pass and the final command should print the repository's
absolute path. Paths are resolved relative to `config.py`, independently of the
current working directory. Directory `.gitkeep` files preserve empty folders in Git.

There is no application to launch in this phase. Running `app.py` does nothing.
No ML training, model inference, or GPU execution is validated by these tests.

## Smoke test functions

- `test_modules_import`: verifies that all seven project modules import.
- `test_project_root`: verifies that the root is the absolute repository path.
- `test_directory_paths`: verifies each configured directory's name, parent,
  and existence.

No production functions or classes are introduced in Phase 1. Later phases
will add dependencies and implementation only when requested.
