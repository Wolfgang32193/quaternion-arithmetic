"""Quaternion arithmetic for four-component real quaternions.

This package provides pure-Python functions for Hamilton product,
conjugate, norm, and rotation extraction. Quaternions are represented
as tuples ``(w, x, y, z)`` of real numbers.
"""

from .core import conjugate, hamilton_product, norm, rotation

__all__ = ["hamilton_product", "conjugate", "norm", "rotation"]
