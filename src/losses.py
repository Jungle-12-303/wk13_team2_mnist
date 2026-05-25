# -*- coding: utf-8 -*-
"""Loss functions."""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error, averaged over a mini-batch.

    Args:
        y_pred: (batch_size, num_classes) softmax probability
        y_true: (batch_size,) integer labels such as 0~9

    Returns:
        Scalar loss. 정답 클래스의 확률이 높을수록 0에 가까워집니다.
    """
    batch_size = y_pred.shape[0]

    # log(0)은 -inf가 되므로 아주 작은 값으로 확률 범위를 제한합니다.
    clipped = np.clip(y_pred, 1e-7, 1.0)

    # 각 샘플에서 정답 클래스 확률만 골라 log를 취합니다.
    correct_log_probs = -np.log(clipped[np.arange(batch_size), y_true])
    return np.mean(correct_log_probs)
