"""UrbanGraph Intelligence: research and engineering foundations."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("urbangraph")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
