# -*- coding: utf-8 -*-
"""Optimizers that update model parameters with gradients."""

import numpy as np


class SGD:
    """Stochastic Gradient Descent."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]


class Momentum:
    """
    SGD with Momentum.

    이전 update 방향을 velocity에 저장해 두었다가 현재 gradient와 섞어 사용합니다.
    같은 방향의 gradient가 반복되면 더 빠르게 이동합니다.
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grads[key]
            params[key] += self.velocity[key]


class Adam:
    """
    Adam optimizer.

    gradient 이동평균(m)과 gradient 제곱 이동평균(v)을 함께 사용합니다.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m, self.v = {}, {}
        self.t = 0

    def update(self, params, grads):
        self.t += 1

        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            grad = grads[key]
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad**2)

            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def create_optimizer(name, lr=0.001, **kwargs):
    """Factory helper used by experiment configs."""
    normalized = name.lower()
    if normalized == "sgd":
        return SGD(lr=lr)
    if normalized in ("momentum", "sgd_momentum"):
        return Momentum(lr=lr, momentum=kwargs.get("momentum", 0.9))
    if normalized == "adam":
        return Adam(
            lr=lr,
            beta1=kwargs.get("beta1", 0.9),
            beta2=kwargs.get("beta2", 0.999),
            eps=kwargs.get("eps", 1e-8),
        )
    raise ValueError(f"Unknown optimizer: {name}")
