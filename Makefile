PYTHON_VERSION := 3.13
UV := .tools/bin/uv
PY := .venv/bin/python
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.tools/python
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib

.PHONY: help setup test notebook example notebooks lock

help:
	@echo "make setup      Install Python 3.13 and locked dependencies in .venv"
	@echo "make test       Run the test suite (headless plotting)"
	@echo "make notebook   Open JupyterLab with the project environment"
	@echo "make example    Run a small Calogero VMC example"
	@echo "make notebooks  Install optional quantum/geospatial notebook packages"
	@echo "make lock       Refresh requirements.txt from requirements.in"

$(UV):
	python3 -m venv .tools
	.tools/bin/python -m pip install --disable-pip-version-check --no-cache-dir uv==0.13.0

.venv/bin/python: $(UV)
	$(UV) venv --python $(PYTHON_VERSION) .venv

setup: .venv/bin/python requirements.txt
	$(UV) pip sync --python $(PY) requirements.txt

lock: $(UV)
	$(UV) pip compile --python-version $(PYTHON_VERSION) requirements.in -o requirements.txt

test: setup
	MPLBACKEND=Agg $(PY) -m pytest

notebook: setup
	$(PY) -m jupyterlab

example: setup
	MPLBACKEND=Agg $(PY) examples/calogero_vmc.py --n 3 --walkers 32 --steps 10 --sweeps 1

notebooks: setup
	$(UV) pip install --python $(PY) -r requirements-notebooks.txt
