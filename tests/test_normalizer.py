import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal


class TestGMedian(unittest.TestCase):
    def test_sparsemax(self):
        from arepc.normalizer import sparsemax

        logits = np.array([0.0, 2.0, 1.0, -1.0])
        probs = sparsemax(logits)
        expected = np.array([0.0, 1.0, 0.0, 0.0])
        assert_array_almost_equal(probs, expected)

        logits = np.array([1.0, 1.0, 1.0, 1.0])
        probs = sparsemax(logits)
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        assert_array_almost_equal(probs, expected)

        logits = np.array([0.0, 0.0, -10.0, -10.0])
        probs = sparsemax(logits)
        expected = np.array([0.5, 0.5, 0.0, 0.0])
        assert_array_almost_equal(probs, expected)

    def test_entmax15(self):
        from arepc.normalizer import entmax15

        logits = np.array([0.0, 10.0, 5.0, -5.0])
        probs = entmax15(logits)
        expected = np.array([0.0, 1.0, 0.0, 0.0])
        assert_array_almost_equal(probs, expected)

        logits = np.array([1.0, 1.0, 1.0, 1.0])
        probs = entmax15(logits)
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        assert_array_almost_equal(probs, expected)

        logits = np.array([0.0, 0.0, -10.0, -10.0])
        probs = entmax15(logits)
        expected = np.array([0.5, 0.5, 0.0, 0.0])
        assert_array_almost_equal(probs, expected)


if __name__ == "__main__":
    unittest.main()
