# -*- coding: utf-8 -*-
"""Layer implementations for the NumPy neural network."""

import numpy as np


class Affine:
    """
    Fully connected layer.

    Formula: y = xW + b.
    Forward에서 저장한 입력 x는 backward에서 dW와 dx를 계산할 때 재사용합니다.
    """

    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)

        Returns:
            (batch_size, output_dim)
        """
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        """
        Args:
            dout: (batch_size, output_dim)

        Returns:
            dx: (batch_size, input_dim)
        """
        self.dW = self.x.T @ dout
        # np.sum(dout, axis=0)는 batch 방향을 합쳐 bias별 gradient를 만듭니다.
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T


class BatchNorm:
    """
    Batch Normalization.

    train=True에서는 현재 mini-batch 평균/분산을 사용하고 running 통계를 갱신합니다.
    train=False에서는 running_mean/running_var만 사용합니다.
    """

    def __init__(self, gamma, beta, momentum=0.9):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.running_mean = np.zeros_like(beta)
        self.running_var = np.zeros_like(beta)
        self.eps = 1e-7

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, feature_dim)
            train: 학습 모드 여부

        Returns:
            x와 같은 shape의 normalized output
        """
        if train:
            # np.mean/np.var(..., axis=0)는 feature별 batch 평균/분산을 계산합니다.
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            self.x_centered = x - mean
            self.std = np.sqrt(var + self.eps)
            self.x_norm = self.x_centered / self.std

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            self.x_centered = x - self.running_mean
            self.std = np.sqrt(self.running_var + self.eps)
            self.x_norm = self.x_centered / self.std

        return self.gamma * self.x_norm + self.beta

    def backward(self, dout):
        """
        Args:
            dout: 뒤쪽 layer에서 온 gradient

        Returns:
            dx: BatchNorm 입력에 대한 gradient
        """
        batch_size = dout.shape[0]
        self.dbeta = np.sum(dout, axis=0)
        self.dgamma = np.sum(dout * self.x_norm, axis=0)

        dx_norm = dout * self.gamma
        dx = (
            (1.0 / batch_size)
            / self.std
            * (
                batch_size * dx_norm
                - np.sum(dx_norm, axis=0)
                - self.x_norm * np.sum(dx_norm * self.x_norm, axis=0)
            )
        )
        return dx


class Dropout:
    """
    Inverted Dropout.

    train=True일 때 일부 값을 0으로 끄고, 살아남은 값을 1/(1-drop_ratio)로 키웁니다.
    이렇게 하면 train=False에서는 아무것도 하지 않고 x를 그대로 반환해도 평균 크기가 맞습니다.
    """

    def __init__(self, drop_ratio=0.5):
        if not 0.0 <= drop_ratio < 1.0:
            raise ValueError("drop_ratio must be in [0.0, 1.0).")
        self.drop_ratio = drop_ratio
        self.mask = None

    def forward(self, x, train=True):
        """
        Args:
            x: input array
            train: True면 random mask 적용, False면 dropout 비활성화

        Returns:
            x와 같은 shape의 배열
        """
        if not train or self.drop_ratio == 0.0:
            self.mask = np.ones_like(x)
            return x

        keep_prob = 1.0 - self.drop_ratio
        # np.random.rand(*x.shape)는 x와 같은 shape의 0~1 난수 배열을 만듭니다.
        self.mask = (np.random.rand(*x.shape) < keep_prob) / keep_prob
        return x * self.mask

    def backward(self, dout):
        """Forward에서 꺼진 위치에는 gradient도 흐르지 않습니다."""
        return dout * self.mask
