# -*- coding: utf-8 -*-
"""Activation functions used by the NumPy MNIST model."""

import numpy as np


class ReLU:
    """
    ReLU(Rectified Linear Unit).

    0보다 큰 값은 그대로 통과시키고, 0 이하 값은 0으로 바꿉니다.
    ReLU 계열은 보통 He initialization과 잘 맞습니다.
    """

    def forward(self, x):
        """
        Args:
            x: any-shaped NumPy array

        Returns:
            x와 같은 shape의 배열. x <= 0인 위치는 0입니다.
        """
        # x > 0은 x와 같은 shape의 bool 배열을 만듭니다.
        # np.where(condition, a, b)는 True 위치에는 a, False 위치에는 b를 넣습니다.
        self.mask = x > 0
        return np.where(self.mask, x, 0)

    def backward(self, dout):
        """Forward에서 0으로 막힌 위치에는 gradient도 흐르지 않습니다."""
        return dout * self.mask


class LeakyReLU:
    """
    Leaky ReLU.

    ReLU와 비슷하지만 x <= 0에서도 alpha만큼 작은 기울기를 남깁니다.
    음수 구간 gradient가 완전히 죽는 문제를 줄일 때 사용합니다.
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        """
        Args:
            x: any-shaped NumPy array

        Returns:
            x > 0이면 x, x <= 0이면 alpha * x를 담은 배열
        """
        self.mask = x > 0
        # np.where(condition, a, b)
        # - 입력: bool 배열 condition, 선택할 값 a/b
        # - 처리: condition 위치별로 a 또는 b를 고릅니다.
        # - 출력: condition과 같은 shape의 배열
        return np.where(self.mask, x, self.alpha * x)

    def backward(self, dout):
        """양수 구간은 1, 음수 구간은 alpha만큼 gradient를 흘립니다."""
        dx = np.where(self.mask, 1.0, self.alpha)
        return dout * dx


class Sigmoid:
    """
    Sigmoid activation function.

    입력값을 0과 1 사이로 바꿉니다.
    수식은 sigmoid(x) = 1 / (1 + exp(-x)) 입니다.
    """

    def forward(self, x):
        """
        Args:
            x: any-shaped NumPy array

        Returns:
            x와 같은 shape의 배열. 모든 값은 0~1 범위입니다.
        """
        # overflow 방지를 위해 x >= 0과 x < 0을 나누어 계산합니다.
        # np.empty_like(x)는 x와 같은 shape/dtype의 빈 배열을 만듭니다.
        out = np.empty_like(x, dtype=np.float64)
        positive = x >= 0

        # np.exp(z)
        # - 입력: 배열 z
        # - 처리: 각 원소에 e의 거듭제곱을 적용합니다.
        # - 출력: 입력과 같은 shape의 양수 배열
        out[positive] = 1 / (1 + np.exp(-x[positive]))
        exp_x = np.exp(x[~positive])
        out[~positive] = exp_x / (1 + exp_x)

        self.out = out
        return self.out

    def backward(self, dout):
        """Sigmoid 미분값은 sigmoid(x) * (1 - sigmoid(x))입니다."""
        return dout * self.out * (1 - self.out)


class Tanh:
    """
    Hyperbolic tangent activation.

    입력값을 -1과 1 사이로 바꿉니다.
    Sigmoid보다 0을 중심으로 출력이 퍼져서 hidden layer에서 자주 사용됩니다.
    """

    def forward(self, x):
        """
        Args:
            x: any-shaped NumPy array

        Returns:
            x와 같은 shape의 배열. 모든 값은 -1~1 범위입니다.
        """
        # np.tanh(x)
        # - 입력: 배열 x
        # - 처리: 각 원소에 tanh 함수를 적용합니다.
        # - 출력: 입력과 같은 shape의 배열
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        """Tanh 미분값은 1 - tanh(x)^2입니다."""
        return dout * (1 - self.out**2)


class Softmax:
    """
    Softmax output helper.

    logits를 class probability로 바꿉니다. loss.py에서도 같은 안정화 공식을 사용합니다.
    """

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_classes) logits

        Returns:
            (batch_size, num_classes) probabilities. 각 row의 합은 1입니다.
        """
        # np.max(..., axis=1, keepdims=True)는 row별 최댓값을 (batch_size, 1)로 돌려줍니다.
        # 이 값을 빼면 exp 계산에서 overflow가 나는 것을 막을 수 있습니다.
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out

    def backward(self, dout):
        """Softmax + Cross Entropy gradient는 loss 쪽에서 만들어 그대로 통과시킵니다."""
        return dout


def get_activation(name):
    """Return an activation layer instance from a string name."""
    normalized = name.lower()
    if normalized == "relu":
        return ReLU()
    if normalized == "leaky_relu":
        return LeakyReLU()
    if normalized == "sigmoid":
        return Sigmoid()
    if normalized == "tanh":
        return Tanh()
    raise ValueError(f"Unknown activation: {name}")
