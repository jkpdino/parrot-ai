# ParrotLM

A GPT-style language model implementation.

## Setup with Pipenv

This project uses Pipenv for dependency management. Follow these steps to set up the project:

1. Install Pipenv if you haven't already:

```bash
pip install pipenv
```

2. Install dependencies:

```bash
pipenv install
```

3. Activate the virtual environment:

```bash
pipenv shell
```

## Usage

You can use the predefined scripts:

- Train a model:

```bash
pipenv run train --config your_config
```

- Chat with a trained model:

```bash
pipenv run chat --model_name your_model --checkpoint path/to/checkpoint
```

- Calculate model size:

```bash
pipenv run calculate_model_size --config your_config
```

Alternatively, after activating the virtual environment with `pipenv shell`, you can run the Python scripts directly:

```bash
python src/train.py --config your_config
```

## Development

To add new dependencies:

```bash
pipenv install package_name
```

For development dependencies:

```bash
pipenv install --dev package_name
```
