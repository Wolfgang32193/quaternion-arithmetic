import math
import unittest

from quaternion_arithmetic.core import conjugate, hamilton_product, norm, rotation


class HamiltonProductTests(unittest.TestCase):
    def test_identity(self):
        q = (1.0, 0.0, 0.0, 0.0)
        self.assertEqual(hamilton_product(q, (2.0, 3.0, -1.0, 0.5)), (2.0, 3.0, -1.0, 0.5))
        self.assertEqual(hamilton_product((2.0, 3.0, -1.0, 0.5), q), (2.0, 3.0, -1.0, 0.5))

    def test_basis_multiplication(self):
        i = (0.0, 1.0, 0.0, 0.0)
        j = (0.0, 0.0, 1.0, 0.0)
        k = (0.0, 0.0, 0.0, 1.0)
        self.assertEqual(hamilton_product(i, j), k)
        self.assertEqual(hamilton_product(j, k), i)
        self.assertEqual(hamilton_product(k, i), j)
        self.assertEqual(hamilton_product(j, i), (-0.0, -0.0, -0.0, -1.0))

    def test_non_commutative(self):
        q1 = (1.0, 2.0, 3.0, 4.0)
        q2 = (5.0, 6.0, 7.0, 8.0)
        self.assertNotEqual(hamilton_product(q1, q2), hamilton_product(q2, q1))

    def test_integer_inputs_are_coerced(self):
        q1 = (1, 0, 0, 0)
        q2 = (0, 1, 0, 0)
        result = hamilton_product(q1, q2)
        self.assertEqual(result, (0.0, 1.0, 0.0, 0.0))
        self.assertTrue(all(isinstance(c, float) for c in result))

    def test_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            hamilton_product((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            hamilton_product((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 0.0))

    def test_rejects_non_numeric(self):
        with self.assertRaises(TypeError):
            hamilton_product(("a", 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(TypeError):
            hamilton_product(None, (1.0, 0.0, 0.0, 0.0))


class ConjugateTests(unittest.TestCase):
    def test_conjugate(self):
        q = (1.0, -2.0, 3.5, 0.0)
        self.assertEqual(conjugate(q), (1.0, 2.0, -3.5, -0.0))

    def test_conjugate_of_real(self):
        q = (5.0, 0.0, 0.0, 0.0)
        self.assertEqual(conjugate(q), (5.0, -0.0, -0.0, -0.0))

    def test_double_conjugate(self):
        q = (0.25, -1.0, 2.0, 3.0)
        result = conjugate(conjugate(q))
        for a, b in zip(result, q):
            self.assertEqual(a, b)

    def test_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            conjugate((1.0, 0.0, 0.0))


class NormTests(unittest.TestCase):
    def test_unit_quaternion(self):
        q = (1.0, 0.0, 0.0, 0.0)
        self.assertEqual(norm(q), 1.0)

    def test_known_value(self):
        q = (1.0, 2.0, 3.0, 4.0)
        self.assertAlmostEqual(norm(q), math.sqrt(30.0))

    def test_zero(self):
        q = (0.0, 0.0, 0.0, 0.0)
        self.assertEqual(norm(q), 0.0)

    def test_negative_components(self):
        q = (-1.0, -2.0, -3.0, -4.0)
        self.assertAlmostEqual(norm(q), math.sqrt(30.0))

    def test_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            norm((1.0, 0.0, 0.0))


class RotationTests(unittest.TestCase):
    def test_identity(self):
        q = (1.0, 0.0, 0.0, 0.0)
        angle, axis = rotation(q)
        self.assertEqual(angle, 0.0)
        self.assertEqual(axis, (1.0, 0.0, 0.0))

    def test_negative_identity(self):
        q = (-1.0, 0.0, 0.0, 0.0)
        angle, axis = rotation(q)
        self.assertEqual(angle, 0.0)
        self.assertEqual(axis, (1.0, 0.0, 0.0))

    def test_90_degree_x_rotation(self):
        # cos(pi/4) = sqrt(2)/2, sin(pi/4) = sqrt(2)/2
        s = math.sqrt(2.0) / 2.0
        q = (s, s, 0.0, 0.0)
        angle, axis = rotation(q)
        self.assertAlmostEqual(angle, math.pi / 2.0)
        for actual, expected in zip(axis, (1.0, 0.0, 0.0)):
            self.assertAlmostEqual(actual, expected)

    def test_120_degree_axis(self):
        # Rotation of 120 degrees around axis (1,1,1)/sqrt(3)
        half_angle = math.pi / 3.0
        w = math.cos(half_angle)
        s = math.sin(half_angle) / math.sqrt(3.0)
        q = (w, s, s, s)
        angle, axis = rotation(q)
        self.assertAlmostEqual(angle, 2.0 * half_angle)
        for actual, expected in zip(axis, (1.0 / math.sqrt(3.0),) * 3):
            self.assertAlmostEqual(actual, expected)

    def test_rejects_non_unit(self):
        with self.assertRaises(ValueError):
            rotation((2.0, 0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            rotation((0.0, 0.0, 0.0, 0.0))

    def test_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            rotation((1.0, 0.0, 0.0))

    def test_returns_floats(self):
        q = (math.sqrt(2.0) / 2.0, math.sqrt(2.0) / 2.0, 0.0, 0.0)
        angle, axis = rotation(q)
        self.assertIsInstance(angle, float)
        self.assertTrue(all(isinstance(c, float) for c in axis))


if __name__ == "__main__":
    unittest.main()
