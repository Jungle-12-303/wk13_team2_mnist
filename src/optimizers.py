# -*- coding: utf-8 -*-
"""Optimizers that update model parameters with gradients."""

import numpy as np


class SGD:
    """
    Stochastic Gradient Descent.

    가장 기본적인 optimizer입니다. 파라미터를 gradient의 반대 방향으로
    learning rate만큼 이동시켜 loss가 작아지도록 합니다.
    """

    def __init__(self, lr=0.01):
        """Args: lr: 한 번 업데이트할 때 gradient에 곱할 학습률"""
        self.lr = lr

    def update(self, params, grads):
        """params dict의 모든 배열을 제자리(in-place)에서 갱신합니다."""
        for key in params:
            params[key] -= self.lr * grads[key]


class Adam:
    """
    Adam optimizer.

    gradient의 이동평균(m)과 제곱 이동평균(v)을 함께 사용해 파라미터마다
    업데이트 크기를 자동으로 조절합니다.
    """

    def __init__(self, lr=0.001):
        """Args: lr: Adam 업데이트의 기본 학습률"""
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.m, self.v = {}, {}
        self.t = 0

    def update(self, params, grads):
        """Adam 공식에 따라 params dict의 모든 파라미터를 갱신합니다."""
        self.t += 1

        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            grad = grads[key]
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad**2)

            # 학습 초반에는 이동평균이 0 쪽으로 치우치므로 bias correction을 적용합니다.
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
