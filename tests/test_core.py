"""
Модульные тесты для ядра Wing Analyzer (CSTUtils, SectionTools).
Запускать в окружении wing, где установлены все зависимости.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import numpy as np
from wing_analyzer import CSTUtils, SectionTools


class TestCSTUtils(unittest.TestCase):
    """Тесты для CST (Class-Shape Transformation)."""

    def test_bernstein_poly_degree_0(self):
        result = CSTUtils.bernstein_poly(0, 0, 0.5)
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_bernstein_poly_degree_1(self):
        self.assertAlmostEqual(CSTUtils.bernstein_poly(1, 0, 0.3), 0.7, places=6)
        self.assertAlmostEqual(CSTUtils.bernstein_poly(1, 1, 0.3), 0.3, places=6)

    def test_shape_zeros(self):
        coeffs = [0.0, 0.0, 0.0]
        x = np.linspace(0, 1, 5)
        y = CSTUtils.shape(x, coeffs, N1=0.5, N2=1.0, dz=0.0)
        self.assertTrue(np.allclose(y, 0.0, atol=1e-10))

    def test_shape_with_dz(self):
        coeffs = [0.0, 0.0, 0.0]
        y = CSTUtils.shape(np.array([1.0]), coeffs, dz=0.1)
        self.assertAlmostEqual(y[0], 0.1, places=6)

    def test_shape_length(self):
        coeffs = [0.1, 0.05, 0.02]
        x = np.linspace(0, 1, 10)
        y = CSTUtils.shape(x, coeffs)
        self.assertEqual(len(y), 10)

    def test_fit_returns_array(self):
        x = np.linspace(0, 1, 20)
        y_true = CSTUtils.shape(x, [0.08, 0.04, 0.02], N1=0.5, N2=1.0)
        coeffs = CSTUtils.fit(x, y_true, degree=2)
        self.assertEqual(len(coeffs), 3)


class TestSectionTools(unittest.TestCase):
    """Тесты для инструментов обработки сечений."""

    def setUp(self):
        self.upper = np.array([
            [0.0, 0.0],
            [0.3, 0.1],
            [0.7, 0.08],
            [1.0, 0.0]
        ])
        self.lower = np.array([
            [0.0, 0.0],
            [0.3, -0.1],
            [0.7, -0.08],
            [1.0, 0.0]
        ])

    def test_get_mid_tail_point(self):
        points = np.vstack([self.upper, self.lower])
        mid_tail, angle, chord_len = SectionTools.get_mid_tail_point(points)
        self.assertIsNotNone(mid_tail)
        self.assertGreater(chord_len, 0)
        self.assertAlmostEqual(angle, 0.0, delta=0.1)

    def test_connect_wing_section_basic(self):
        all_points = np.vstack([self.upper, self.lower])
        contour, connections, is_closed, msg, up, low = SectionTools.connect_wing_section(all_points)
        self.assertTrue(is_closed)
        self.assertEqual(len(contour), len(connections))
        self.assertIsNotNone(up)
        self.assertIsNotNone(low)

    def test_arc_length_parameterization(self):
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.5, 0.0])
        x_new, y_new = SectionTools.arc_length_parameterization(x, y, 5)
        self.assertEqual(len(x_new), 5)
        self.assertEqual(len(y_new), 5)
        self.assertAlmostEqual(x_new[0], 0.0)
        self.assertAlmostEqual(x_new[-1], 1.0)

    def test_auto_approximate_section_returns_none_for_few_points(self):
        few_upper = np.array([[0.0, 0.0], [1.0, 0.0]])
        few_lower = np.array([[0.0, 0.0], [1.0, 0.0]])
        up, low = SectionTools.auto_approximate_section(few_upper, few_lower)
        self.assertIsNone(up)
        self.assertIsNone(low)


if __name__ == '__main__':
    unittest.main()