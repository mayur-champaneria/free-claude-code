"""LQQQ provider exports."""

from providers.defaults import LQQQ_DEFAULT_BASE

from .client import LqqqProvider

__all__ = [
    "LQQQ_DEFAULT_BASE",
    "LqqqProvider",
]
