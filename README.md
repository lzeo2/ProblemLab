# ProblemLab

ProblemLab is a small project for extracting numbered questions from PDF files and experimenting with classifying maths problems. It contains Python tools and an unfinished C++ prototype.

## Requirements

- Windows
- Python 3.13 (the project pins this in `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) for Python environments and packages
- For the C++ prototype: a C++17 compiler such as `g++`

The repository currently has `g++`, CMake, and Ninja available in the development environment. Python is provided by the project virtual environment, so use `uv` commands rather than relying on a system-wide `python` command.

## Set up Python

From the project directory, run:

```powershell
uv sync --locked
```

This creates or updates `.venv` from `pyproject.toml` and `uv.lock`. Run project scripts through `uv` so they use that environment:

```powershell
uv run pdfextractor.py
```

The extractor currently opens `AMC_2024_Intermediate.pdf` from the project directory and writes extracted questions to `AMC_2024_Intermediate.pdf.json`.

## Classifier

`inference.py` loads the `answerdotai/ModernBERT-base` tokenizer and configuration from Hugging Face and loads weights from `model.safetensors` in the project directory. The first run may need internet access to fetch the tokenizer and model configuration.

```powershell
uv run inference.py
```

The script currently includes its own sample problems and runs classification when executed.

## C++ prototype

`main.cpp` is a small C++ wrapper that runs the PDF extractor and reads the generated JSON. The `nlohmann/json` single header is included at `third_party/nlohmann/json.hpp` (version 3.12.0). In Zed, run the **Build and run ProblemLab** task from Task: Spawn; it compiles the program with `third_party` on the include path, then runs it and shows its output. The `.clangd` file adds the same path for editor diagnostics.

## Project files

- `pyproject.toml`, `uv.lock`: Python project metadata and locked dependencies
- `requirements.txt`: pip-compatible dependency list
- `pdfextractor.py`: extracts numbered questions from the sample PDF
- `inference.py`: loads the classifier and runs sample predictions
- `model.safetensors`: classifier weights
- `main.cpp`: unfinished C++ prototype
