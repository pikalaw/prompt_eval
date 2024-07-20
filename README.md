# Prompt eval

## Setup

```bash
poetry init
poetry shell

poetry add pydantic devtools
poetry add mypy black pytest --group dev

poetry add datasets huggingface_hub
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
