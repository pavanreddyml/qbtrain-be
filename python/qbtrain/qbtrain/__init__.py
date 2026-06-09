"""qbtrain — tools for training and serving models.

The single source of truth for the version is ``pyproject.toml``. At runtime we
read the *installed* package metadata so the library, the server, and the Docker
image all report the exact same number.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("qbtrain")
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
