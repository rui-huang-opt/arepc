import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal


class TestGMedian(unittest.TestCase):
    def test_geometric_median(self):
        from arepc.geometry.gmedian import geometric_median

        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        gm = geometric_median(points)
        expected = np.array([0.5, 0.5])
        assert_array_almost_equal(gm, expected)

        points = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, -2.0],
            ]
        )
        gm = geometric_median(points)
        expected = np.array([0.0, 0.0, 0.0])
        assert_array_almost_equal(gm, expected)


if __name__ == "__main__":
    unittest.main()
