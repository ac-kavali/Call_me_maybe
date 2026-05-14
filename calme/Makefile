.PHONY: install run debug clean lint lint-strict setup

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache src/__pycache__

lint:
	flake8 ./src
	mypy ./src --warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	flake8 ./src
	mypy ./src --strict

setup:
	echo 'export UV_CACHE_DIR=~/goinfre/uv/cache' >> ~/.zshrc
	echo 'export UV_TOOL_DIR=~/goinfre/uv/tools' >> ~/.zshrc
	echo 'export HF_HOME=~/goinfre/hf_cache' >> ~/.zshrc
	echo 'export HUGGINGFACE_HUB_CACHE=~/goinfre/hf_cache' >> ~/.zshrc
	echo 'export PIP_CACHE_DIR=~/goinfre/pip/cache' >> ~/.zshrc
	source ~/.zshrc
	mkdir -p ~/goinfre/hf_cache
	uv venv ~/goinfre/.venv
	ln -s ~/goinfre/.venv .venv   # symlink into your project
	uv sync