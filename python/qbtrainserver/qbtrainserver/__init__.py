"""qbtrainserver — Django server for the QBTrain platform.

The server ships as the same release as the ``qbtrain`` library, so its version
is read from the installed library: library == server == image tag.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("qbtrain")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
