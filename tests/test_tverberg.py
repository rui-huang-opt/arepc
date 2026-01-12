import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal


class TestTverberg(unittest.TestCase):
    def test_affine_dependence(self):
        from arepc.geometry.tverberg import affine_dependence

        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        ad = affine_dependence(points)
        assert_array_almost_equal(ad.sum(), 0)
        assert_array_almost_equal(ad @ points, 0)

        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        ad = affine_dependence(points)
        assert_array_almost_equal(ad.sum(), 0)
        assert_array_almost_equal(ad @ points, 0)

    def test_convex_combination(self):
        from arepc.geometry.tverberg import convex_combination

        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        target = np.array([0.25, 0.25])
        cc = convex_combination(target, points)
        assert_array_almost_equal(cc.sum(), 1)
        assert_array_almost_equal(cc @ points, target)

        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        target = np.array([0.25, 0.25, 0.25])
        cc = convex_combination(target, points)
        assert_array_almost_equal(cc.sum(), 1)
        assert_array_almost_equal(cc @ points, target)

    def test_radon(self):
        from arepc.geometry.tverberg import radon, convex_combination

        points = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cp = radon(points)

        r = cp.point
        P, N = cp.proof

        # Partitions should be non-empty
        self.assertTrue(len(P) > 0)
        self.assertTrue(len(N) > 0)

        # Radon point should be correct
        assert_array_almost_equal(r, np.array([0.5, 0.5]))

        # Verify r is in the convex hulls of both partitions
        for part in cp.proof:
            cc = convex_combination(r, points[part])
            assert_array_almost_equal(points[part].T @ cc, r)

    def test_prune(self):
        from arepc.geometry.tverberg import prune, convex_combination

        points = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        target = np.array([0.5, 0.5])
        indices = list(range(len(points)))

        new_indices = prune(target, indices, points)

        self.assertLessEqual(len(new_indices), 3)

        cc = convex_combination(target, points[new_indices])
        assert_array_almost_equal(points[new_indices].T @ cc, target)

        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.2, 0.2, 0.2],
            ]
        )
        target = np.array([0.25, 0.25, 0.25])
        indices = list(range(len(points)))

        new_indices = prune(target, indices, points)

        self.assertLessEqual(len(new_indices), 4)

        cc = convex_combination(target, points[new_indices])
        assert_array_almost_equal(points[new_indices].T @ cc, target)

    def test_iterated_tverberg(self):
        from arepc.geometry.tverberg import iterated_tverberg

        rng = np.random.default_rng(0)
        data = rng.random((20, 2))

        cp = iterated_tverberg(data)

        self.assertIsInstance(cp.proof, list)
        for block in cp.proof:
            self.assertIsInstance(block, list)
            for idx in block:
                self.assertIsInstance(idx, int)
                self.assertTrue(0 <= idx < len(data))


if __name__ == "__main__":
    unittest.main()
