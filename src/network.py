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
            self.params[f"W{idx}"] = (
                np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
            )
            self.params[f"b{idx}"] = np.zeros(output_dim)

            if self.use_batchnorm and idx < len(layer_sizes) - 1:
                self.params[f"gamma{idx}"] = np.ones(output_dim)
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
