# Repository Guidelines

## Project Structure & Module Organization
Core runtime lives in `slime/` (training loops, rollout orchestration, misc utilities) while `slime_plugins/` carries optional reward-model, router, and scheduler integrations. Launch helpers reside in `scripts/`, with model presets under `scripts/models/*.sh` and training entrypoints such as `scripts/run-glm4-9B.sh`. Reusable recipes stay in `examples/`; long-form docs live in `docs/en` and `docs/zh`. GPU-facing regression harnesses and pytest modules both sit in `tests/`, conversion helpers in `tools/`, and Docker assets under `docker/` plus `docker-compose.yml`.

## Build, Test, and Development Commands
- `pip install -e .` after cloning (inside the slime Docker image) for an editable setup that tracks the repo.
- `pip install -r requirements.txt` when CLI utilities or docs scripts add new third-party dependencies.
- `pre-commit run --all-files --show-diff-on-failure` to apply Black/isort/Ruff before committing; install once with `pre-commit install`.
- `python -m pytest tests -m "not skipduringci"` for CPU-level coverage; add `-k rollout` or similar to narrow focus.
- `bash scripts/run-glm4-9B.sh` (or another launcher) after sourcing the matching `scripts/models/<model>.sh` to exercise the full rollout→train loop end-to-end.

## Coding Style & Naming Conventions
Python targets 3.10 with 4-space indentation, Black-compatible formatting, line length 119, and import order enforced by isort (see `pyproject.toml`). Keep modules and functions `snake_case`, classes `CamelCase`, and shared CLI argument arrays in `UPPER_CASE` (e.g., `ROLLOUT_ARGS`). Prefer dataclasses/TypedDicts for configuration objects, guard GPU-specific codepaths, and add concise docstrings for new user-facing commands.

## Testing Guidelines
Name unit tests `tests/test_<feature>.py` and mark them with `@pytest.mark.unit` or `integration` per the existing marker set. Heavy GPU or distributed checks should follow the style of `tests/test_qwen3-0.6B_fsdp_distributed.sh`, gated by environment variables (`SLIME_SCRIPT_*`) so they can be skipped in CI. Cover new utilities via `pytest tests/test_quick_start_glm4_9B.py -k <case>` and document any WANDB/HF credentials needed. When adding shell harnesses, reuse helpers from `tests/command_utils.py` to manage Ray jobs and checkpoint conversion.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects with optional scopes and issue references (e.g., `[docker] update sglang patch (#704)`). Keep each commit focused on one concern and mention affected scripts or configs in the body. Pull requests should include: a concise problem statement, environment details (GPU type, CUDA, dataset location), reproduction commands, and before/after metrics or logs for training changes. Link issues via `Fixes #ID`, call out docs/tests added, and attach screenshots when touching dashboards or monitoring outputs.

## Security & Configuration Tips
Never commit secrets such as `WANDB_API_KEY`, Hugging Face tokens, or customer datasets; rely on environment variables and document the expectation in README or sample `.env.template`. When sharing configs, reference the sanitized presets in `scripts/models/` instead of pasting internal paths. Validate Docker changes locally with `docker compose -f docker-compose.yml up --build` and keep host paths under `/root/slime` to stay compatible with CI images.
