# -*- coding: utf-8 -*-
"""Layer implementations for the NumPy neural network."""

import numpy as np


class Affine:
    """
    Fully connected layer.

    수식은 y = xW + b 입니다. MNIST 이미지 한 장(784차원)을 hidden/output
    차원으로 선형 변환할 때 사용합니다.
    """

    def __init__(self, W, b):
        # W와 b는 model.params 안의 배열과 같은 객체를 참조합니다.
        self.W = W
        self.b = b

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)

        Returns:
            (batch_size, output_dim)
        """
        # backward에서 dW = x.T @ dout을 계산해야 하므로 입력을 저장합니다.
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        """
        Args:
            dout: (batch_size, output_dim), 뒤쪽 layer에서 온 gradient

        Returns:
            dx: (batch_size, input_dim), 앞쪽 layer로 보낼 gradient
        """
        self.dW = self.x.T @ dout
        # np.sum(dout, axis=0)
        # - 입력: (batch_size, output_dim) gradient
        # - 처리: batch 방향을 모두 더해 bias 하나당 gradient를 구함
        # - 출력: (output_dim,) 배열
        self.db = np.sum(dout, axis=0)
        dx = dout @ self.W.T
        return dx


class BatchNorm:
    """
    Batch Normalization.

    mini-batch 안에서 feature별 평균과 분산을 맞춰 학습을 안정화합니다.
    학습 중에는 현재 batch 통계를 쓰고, 추론 중에는 running 통계를 씁니다.
    """

    def __init__(self, gamma, beta, momentum=0.9):
        """
        Args:
            gamma: 정규화된 값을 다시 scale하는 학습 파라미터
            beta: 정규화된 값에 더하는 shift 학습 파라미터
            momentum: running_mean/running_var를 부드럽게 갱신하는 비율
        """
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        # np.zeros_like(beta)
        # - 입력: 기준 배열 beta
        # - 처리: beta와 같은 shape/dtype을 가진 0 배열을 만듦
        # - 출력: beta와 shape이 같은 0 배열
        self.running_mean = np.zeros_like(beta)
        self.running_var = np.zeros_like(beta)
        self.eps = 1e-7

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, feature_dim)
            train: True면 현재 batch 통계, False면 running 통계 사용

        Returns:
            x와 같은 shape의 정규화된 출력
        """
        if train:
            # np.mean(x, axis=0)
            # - 입력: (batch_size, feature_dim) 배열
            # - 처리: batch 방향 평균을 구해 feature별 평균을 계산
            # - 출력: (feature_dim,) 배열
            batch_mean = np.mean(x, axis=0)
            # np.var(x, axis=0)
            # - 입력: (batch_size, feature_dim) 배열
            # - 처리: batch 방향 분산을 구해 feature별 분산을 계산
            # - 출력: (feature_dim,) 배열
            batch_var = np.var(x, axis=0)

            self.x_centered = x - batch_mean
            # np.sqrt(...)
            # - 입력: 분산 + eps 배열
            # - 처리: 각 원소의 제곱근을 계산해 표준편차로 바꿈
            # - 출력: 입력과 같은 shape의 배열
            self.std = np.sqrt(batch_var + self.eps)
            self.x_norm = self.x_centered / self.std

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * batch_var
            )
        else:
            # 추론 때는 batch 하나에 흔들리지 않도록 학습 중 누적한 통계를 씁니다.
            self.x_centered = x - self.running_mean
            # np.sqrt는 running variance를 표준편차로 바꾸는 데 사용합니다.
            self.std = np.sqrt(self.running_var + self.eps)
            self.x_norm = self.x_centered / self.std

        return self.gamma * self.x_norm + self.beta

    def backward(self, dout):
        """
        BatchNorm 입력 x, scale gamma, shift beta에 대한 gradient를 계산합니다.

        Args:
            dout: 뒤쪽 layer에서 온 gradient

        Returns:
            dx: BatchNorm 입력 x에 대한 gradient
        """
        batch_size = dout.shape[0]

        # np.sum(..., axis=0)은 batch 안의 gradient를 feature별로 합칩니다.
        self.dbeta = np.sum(dout, axis=0)
        self.dgamma = np.sum(dout * self.x_norm, axis=0)

        dx_norm = dout * self.gamma

        # x_norm = (x - mean) / std 를 한 번에 미분한 compact formula입니다.
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
    Dropout.

    학습 중 일부 neuron 출력을 무작위로 0으로 만들어 과적합을 줄입니다.
    이 구현은 추론 시 출력에 (1 - drop_ratio)를 곱하는 기본 dropout 방식입니다.
    """

    def __init__(self, drop_ratio=0.5):
        """Args: drop_ratio: 학습 중 0으로 만들 neuron 비율"""
        self.drop_ratio = drop_ratio

    def forward(self, x, train=True):
        """
        Args:
            x: 입력 배열
            train: True면 random mask 적용, False면 평균 출력 크기로 scale
        """
        if train:
            # np.random.rand(*x.shape)
            # - 입력: 만들고 싶은 차원 크기들. *x.shape은 x의 shape을 풀어서 전달합니다.
            # - 처리: 0 이상 1 미만의 균등분포 난수를 생성
            # - 출력: x와 같은 shape의 난수 배열
            # drop_ratio보다 큰 위치만 True가 되어 살아남습니다.
            self.mask = np.random.rand(*x.shape) > self.drop_ratio
            return x * self.mask

        return x * (1 - self.drop_ratio)

    def backward(self, dout):
        """forward에서 꺼진 neuron 위치에는 gradient도 흐르지 않습니다."""
        return dout * self.mask
