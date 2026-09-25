# Quaternion Arithmetic

Pure-Python functions for Hamilton product, conjugate, norm, and rotation extraction on four-component real quaternions represented as `(w, x, y, z)` tuples.

```python
from quaternion_arithmetic import hamilton_product, conjugate, norm, rotation

q1 = (1.0, 2.0, 3.0, 4.0)
q2 = (5.0, 6.0, 7.0, 8.0)

product = hamilton_product(q1, q2)
conj = conjugate(q1)
n = norm(q1)

unit = (0.7071067811865476, 0.7071067811865476, 0.0, 0.0)
angle, axis = rotation(unit)
```

## Why this library exists

Quaternions are the standard way to represent 3D rotations without gimbal lock, but the formulas are easy to get wrong by a sign here or there. This library provides a minimal, dependency-free implementation of the core operations with explicit handling of non-commutativity and unit-norm requirements.

The main trade-off is that quaternions are plain tuples, not a class. This keeps the API simple and avoids object overhead, but it means the functions cannot enforce normalization or type at construction time. Instead, `rotation()` validates its input explicitly.

## Awkward edge cases

- The Hamilton product is non-commutative: `hamilton_product(q1, q2) != hamilton_product(q2, q1)` in general.
- `rotation()` raises `ValueError` for quaternions whose norm differs from 1.0 by more than 1e-12. This tolerance absorbs floating-point rounding when constructing unit quaternions from trigonometric values.
- Both `(1,0,0,0)` and `(-1,0,0,0)` represent the identity rotation; `rotation()` returns angle `0.0` and axis `(1.0, 0.0, 0.0)` for both.
- Inputs are coerced to floats; integer components are accepted and the results are always floats.
