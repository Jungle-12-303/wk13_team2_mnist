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
        # x > 0은 x와 같은 shape의 bool 배열을 만듭니다.
        # 예: [[-1, 2]] -> [[False, True]]
        # mask는 "이 위치로 gradient가 지나갈 수 있는가?"를 기억하는 표입니다.
        self.mask = x > 0
        # np.where(condition, a, b)
        # - condition: bool 배열
        # - a: condition이 True인 위치에 넣을 값
        # - b: condition이 False인 위치에 넣을 값
        # - 출력: condition과 같은 shape의 배열
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


class Sigmoid:
    """
    Sigmoid activation function.

    입력값을 0과 1 사이의 값으로 바꿉니다.
    수식은 sigmoid(x) = 1 / (1 + exp(-x)) 입니다.

    값이 아주 작으면 0에 가까워지고, 아주 크면 1에 가까워집니다.
    이진 분류나 "켜짐/꺼짐" 같은 확률 느낌의 값을 만들 때 자주 등장합니다.
    """

    def forward(self, x):
        """
        Args:
            x: 어떤 shape이든 가능한 입력 배열

        Returns:
            x와 같은 shape의 배열. 모든 값은 0보다 크고 1보다 작습니다.
        """
        # np.exp(-x)
        # - 입력: x와 같은 shape의 배열
        # - 처리: 각 원소에 자연상수 e의 거듭제곱을 적용합니다.
        #         여기서는 -x를 넣으므로 exp(-x)를 계산합니다.
        # - 출력: x와 같은 shape의 양수 배열
        #
        # 1 / (1 + exp(-x))는 NumPy broadcasting으로 모든 원소에 각각 적용됩니다.
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        """
        Args:
            dout: 뒤쪽 layer에서 전달된 gradient

        Returns:
            Sigmoid 입력 x에 대한 gradient
        """
        # Sigmoid의 미분값은 sigmoid(x) * (1 - sigmoid(x))입니다.
        # forward에서 self.out에 sigmoid(x)를 저장했으므로 다시 계산할 필요가 없습니다.
        return dout * self.out * (1 - self.out)


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
        # np.max(x, axis=1, keepdims=True)
        # - 입력: (batch_size, num_classes) 배열 x
        # - 처리: 각 row(샘플)에서 가장 큰 logit을 찾음
        # - 출력: keepdims=True라서 (batch_size, 1) shape 유지
        shifted = x - np.max(x, axis=1, keepdims=True)
        # np.exp(shifted)
        # - 입력: shifted와 같은 shape의 배열
        # - 처리: 각 원소에 자연상수 e의 거듭제곱을 적용
        # - 출력: shifted와 같은 shape의 양수 배열
        exp_x = np.exp(shifted)
        # np.sum(exp_x, axis=1, keepdims=True)
        # - 입력: (batch_size, num_classes) 배열
        # - 처리: 각 row의 클래스 점수 합계를 구함
        # - 출력: (batch_size, 1) 배열. 나누기 때 row별로 broadcast됩니다.
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out

    def backward(self, dout):
        """
        Softmax와 Cross Entropy를 함께 미분하면 gradient가 단순해집니다.
        train()에서 이미 그 gradient를 만들기 때문에 여기서는 그대로 넘깁니다.
        """
        return dout
