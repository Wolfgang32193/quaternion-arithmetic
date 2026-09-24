"""Core quaternion operations on ``(w, x, y, z)`` tuples.

All functions are pure and operate on 4-element sequences of real numbers.
The real component is stored first: ``(w, x, y, z)`` where
``w + x*i + y*j + z*k`` is the quaternion.
"""

from math import sqrt


def _coerce_quaternion(value, name="quaternion"):
    """Return a 4-tuple of floats or raise TypeError/ValueError.

    Accepts any iterable of exactly four real numbers. Strictly enforces
    length so callers do not silently pass 3-vectors where quaternions are
    expected.
    """
    try:
        components = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of 4 real numbers") from exc
    if len(components) != 4:
        raise ValueError(f"{name} must contain exactly 4 components")
    try:
        return tuple(float(c) for c in components)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain only real numbers") from exc


def hamilton_product(q1, q2):
    """Return the Hamilton product of two quaternions.

    The product is non-commutative: ``q1 * q2`` is generally different
    from ``q2 * q1``. We use the standard convention
    ``i*j = k``, ``j*k = i``, ``k*i = j``.
    """
    a1, b1, c1, d1 = _coerce_quaternion(q1, "q1")
    a2, b2, c2, d2 = _coerce_quaternion(q2, "q2")

    return (
        a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
        a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
        a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
        a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
    )


def conjugate(q):
    """Return the quaternion conjugate ``(w, -x, -y, -z)``."""
    w, x, y, z = _coerce_quaternion(q, "q")
    return (w, -x, -y, -z)


def norm(q):
    """Return the Euclidean norm ``sqrt(w^2 + x^2 + y^2 + z^2)``.

    Uses ``math.sqrt``, which returns a float. For exact integer inputs
    whose square root is irrational, the result is therefore an
    approximation.
    """
    w, x, y, z = _coerce_quaternion(q, "q")
    return sqrt(w * w + x * x + y * y + z * z)


def rotation(q):
    """Extract the rotation angle and axis from a unit quaternion.

    The quaternion is interpreted as a rotation in 3D space. Returns a
    tuple ``(angle, axis)`` where ``angle`` is in radians and ``axis``
    is a 3-tuple ``(x, y, z)``. The angle is in ``[0, pi]`` and the axis
    is normalized.

    For a non-unit quaternion the result is undefined by the underlying
    mathematics, so we require the caller to provide a unit quaternion.
    The norm is checked and a ``ValueError`` is raised if it deviates
    from 1.0 by more than 1e-12. This tolerance avoids spurious failures
    from floating-point rounding while still rejecting genuinely
    non-normalized inputs.

    Edge cases:
        - Identity quaternion ``(1, 0, 0, 0)`` returns ``(0.0,
          (1.0, 0.0, 0.0))``.
        - The ``-identity`` quaternion ``(-1, 0, 0, 0)`` is equivalent
          to the identity rotation, so it also returns angle 0.
    """
    q = _coerce_quaternion(q, "q")
    n = norm(q)
    if abs(n - 1.0) > 1e-12:
        raise ValueError("rotation() requires a unit quaternion")

    w, x, y, z = q
    if w >= 1.0 - 1e-15:
        return (0.0, (1.0, 0.0, 0.0))
    if w <= -1.0 + 1e-15:
        return (0.0, (1.0, 0.0, 0.0))

    angle = 2.0 * __import__("math").acos(w)
    s = sqrt(1.0 - w * w)
    axis = (x / s, y / s, z / s)
    return (angle, axis)
