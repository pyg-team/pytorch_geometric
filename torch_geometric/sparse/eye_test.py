from unittest import TestCase

from numpy.testing import assert_equal

from .eye import eye


class EyeTest(TestCase):
    def test_eye_n(self):
        out = eye(3)
        expected_out = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        assert_equal(out.to_dense().numpy(), expected_out)

    def test_eye_n_m(self):
        out = eye(3, 4)
        expected = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]

        assert_equal(out.to_dense().numpy(), expected)

    def test_eye_m_n(self):
        out = eye(4, 3)
        expected = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]

        assert_equal(out.to_dense().numpy(), expected)
