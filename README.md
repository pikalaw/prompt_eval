# Prompt eval

## Setup

```bash
poetry init
poetry shell

poetry add pydantic devtools
poetry add mypy black pytest --group dev

poetry add datasets huggingface_hub
poetry add google-generativeai
```

If `poetry` hangs, add this to the `.zshrc`:

```bash
# To prevent poetry from hanging forever.
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

Add to pyproject.toml:

```toml
[tool.mypy]
# Many Google's libraries do not have stub packages.
ignore_missing_imports = true
```

## Citation

The dataset came from here:

```
@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```
