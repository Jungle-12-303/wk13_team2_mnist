# -*- coding: utf-8 -*-
"""Sigmoid activation tests."""

import numpy as np

from activations import Sigmoid


class TestSigmoid:
    """Sigmoid.forward(), Sigmoid.backward() behavior checks."""

    def test_sigmoid_forward_range_and_midpoint(self):
        sigmoid = Sigmoid()
        x = np.array([[-1.0, 0.0, 1.0]])
        out = sigmoid.forward(x)

        assert out.shape == x.shape
        assert (out > 0).all() and (out < 1).all()
        np.testing.assert_almost_equal(out[0, 1], 0.5)

    def test_sigmoid_backward(self):
        sigmoid = Sigmoid()
        x = np.array([[0.0, 1.0]])
        out = sigmoid.forward(x)
        dout = np.ones_like(x)
        dx = sigmoid.backward(dout)

        expected = out * (1 - out)
        assert dx.shape == x.shape
        np.testing.assert_array_almost_equal(dx, expected)
