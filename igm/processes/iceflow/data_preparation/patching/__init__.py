"""
Patching module for splitting tensors into smaller patches.

This module provides different patching strategies for processing large tensors
by splitting them into smaller, manageable patches.
"""

from .base import Patching
from .overlap import OverlapPatching
from .grid import GridPatching

__all__ = ["Patching", "OverlapPatching", "GridPatching"]
