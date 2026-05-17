# Contributing Guide

Thank you for contributing to this project.

## Development Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Before Opening a Pull Request

- Keep changes focused and scoped to one concern.
- Update `README.md` or `docs/` when behavior changes.
- Avoid committing large model weights unless there is a clear reason.
- Verify the app still starts:

```powershell
python main.py --source 0 --device cpu
```

- If you have CUDA available, also verify:

```powershell
python main.py --source 0 --device cuda
```

## Coding Guidelines

- Prefer small, readable modules.
- Keep configuration in `config.py` unless a runtime flag is more appropriate.
- Add concise comments only where logic is not obvious.
- Preserve CPU fallback behavior.

## Commit and Pull Request Notes

- Use clear commit messages.
- Describe what changed, why it changed, and how it was tested.
- Include screenshots or short recordings for UI or visualization changes when useful.

