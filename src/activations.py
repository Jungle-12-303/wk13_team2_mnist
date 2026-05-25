# -*- coding: utf-8 -*-
"""Activation functions used by the NumPy MNIST model."""

import numpy as np


class ReLU:
    """
    ReLU(Rectified Linear Unit).

    입력값이 0보다 크면 그대로 통과시키고, 0 이하이면 0으로 바꿉니다.
    역전파 때는 순전파에서 살아남은 위치(x > 0)에만 gradient를 흘려 보냅니다.
    """

    def forward(self, x):
        """
        Args:
            x: 어떤 shape이든 가능한 입력 배열

        Returns:
            x와 같은 shape의 배열. 양수는 그대로, 0 이하 값은 0입니다.
        """
        # mask는 "이 위치로 gradient가 지나갈 수 있는가?"를 기억하는 표입니다.
        self.mask = x > 0
        return np.where(self.mask, x, 0)

    def backward(self, dout):
        """
        Args:
            dout: 뒤쪽 layer에서 전달된 gradient

        Returns:
            ReLU 입력 x에 대한 gradient
        """
        # 순전파에서 0으로 막힌 위치는 기울기도 0이 됩니다.
        return dout * self.mask


class Softmax:
    """
    Softmax output layer.

    각 샘플의 logit을 클래스별 확률로 바꿉니다.
    exp를 계산하기 전에 row별 최댓값을 빼면 overflow를 피할 수 있습니다.
    """

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_classes) logit 배열

        Returns:
            (batch_size, num_classes) 확률 배열. 각 row의 합은 1입니다.
        """
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out

    def backward(self, dout):
        """
        Softmax와 Cross Entropy를 함께 미분하면 gradient가 단순해집니다.
        train()에서 이미 그 gradient를 만들기 때문에 여기서는 그대로 넘깁니다.
        """
        return dout
