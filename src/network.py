# -*- coding: utf-8 -*-
"""Neural network assembly for MNIST classification."""

from collections import OrderedDict

import numpy as np

from activations import ReLU, Softmax
from layers import Affine, BatchNorm, Dropout
from losses import cross_entropy_loss


class NeuralNetwork:
    """
    MNIST classifier built only with NumPy.

    기본 구조는 784 -> 512 -> 256 -> 10 입니다.
    hidden layer는 Affine -> BatchNorm(optional) -> ReLU -> Dropout(optional)
    순서로 구성합니다.
    """

    def __init__(self, use_batchnorm=True, use_dropout=True, dropout_ratio=0.5):
        """
        Args:
            use_batchnorm: hidden layer마다 BatchNorm을 넣을지 여부
            use_dropout: hidden layer마다 Dropout을 넣을지 여부
            dropout_ratio: Dropout에서 끌 neuron 비율
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.params = {}
        self.grads = {}

        layer_sizes = [784, 512, 256, 10]

        # ReLU와 잘 맞는 He initialization입니다.
        # 입력 차원이 클수록 작은 값으로 시작해 activation 폭주를 줄입니다.
        for idx in range(1, len(layer_sizes)):
            input_dim = layer_sizes[idx - 1]
            output_dim = layer_sizes[idx]
            # np.random.randn(input_dim, output_dim)
            # - 입력: 만들 배열의 shape
            # - 처리: 평균 0, 표준편차 1인 정규분포 난수를 생성
            # - 출력: (input_dim, output_dim) weight 배열
            #
            # np.sqrt(2.0 / input_dim)
            # - 입력: 숫자 하나
            # - 처리: 제곱근을 계산
            # - 출력: weight scale로 쓸 scalar
            self.params[f"W{idx}"] = (
                np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
            )
            # np.zeros(output_dim)
            # - 입력: 만들 배열 길이
            # - 처리: 모든 값이 0인 배열 생성
            # - 출력: (output_dim,) bias 배열
            self.params[f"b{idx}"] = np.zeros(output_dim)

            if self.use_batchnorm and idx < len(layer_sizes) - 1:
                # np.ones(output_dim)은 모든 값이 1인 (output_dim,) 배열을 만듭니다.
                # BatchNorm의 gamma는 처음에 값을 그대로 통과시키기 위해 1로 둡니다.
                self.params[f"gamma{idx}"] = np.ones(output_dim)
                # beta는 처음에 shift를 주지 않기 위해 0으로 둡니다.
                self.params[f"beta{idx}"] = np.zeros(output_dim)

        self.layers = OrderedDict()
        for idx in range(1, len(layer_sizes)):
            self.layers[f"Affine{idx}"] = Affine(
                self.params[f"W{idx}"], self.params[f"b{idx}"]
            )

            is_hidden_layer = idx < len(layer_sizes) - 1
            if is_hidden_layer:
                if self.use_batchnorm:
                    self.layers[f"BatchNorm{idx}"] = BatchNorm(
                        self.params[f"gamma{idx}"], self.params[f"beta{idx}"]
                    )
                self.layers[f"ReLU{idx}"] = ReLU()
                if self.use_dropout:
                    self.layers[f"Dropout{idx}"] = Dropout(dropout_ratio)

        self.softmax = Softmax()
        # np.zeros_like(value)는 각 파라미터와 같은 shape의 gradient 저장 공간을 만듭니다.
        self.grads = {key: np.zeros_like(value) for key, value in self.params.items()}

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, 784) normalized MNIST images
            train: BatchNorm/Dropout 학습 모드 여부

        Returns:
            (batch_size, 10) class probabilities
        """
        out = x
        for layer in self.layers.values():
            if isinstance(layer, (BatchNorm, Dropout)):
                out = layer.forward(out, train=train)
            else:
                out = layer.forward(out)
        return self.softmax.forward(out)

    def backward(self, dout):
        """
        Run backpropagation through the whole network and fill self.grads.

        Args:
            dout: Softmax + CrossEntropy를 합친 출력층 gradient
        """
        dout = self.softmax.backward(dout)

        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        for name, layer in self.layers.items():
            if isinstance(layer, Affine):
                idx = name.replace("Affine", "")
                self.grads[f"W{idx}"] = layer.dW
                self.grads[f"b{idx}"] = layer.db
            elif isinstance(layer, BatchNorm):
                idx = name.replace("BatchNorm", "")
                self.grads[f"gamma{idx}"] = layer.dgamma
                self.grads[f"beta{idx}"] = layer.dbeta

    def loss(self, x, y):
        """Return the cross entropy loss for the current model prediction."""
        y_pred = self.forward(x, train=True)
        return cross_entropy_loss(y_pred, y)

    def predict(self, x):
        """Predict probabilities in inference mode."""
        return self.forward(x, train=False)
