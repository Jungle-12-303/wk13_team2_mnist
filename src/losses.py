# -*- coding: utf-8 -*-
"""Loss functions and Softmax/Cross Entropy helpers."""

import numpy as np


def softmax(logits):
    """
    Numerically stable softmax.

    Args:
        logits: (batch_size, num_classes) raw model outputs

    Returns:
        Same shape probability array. Each row sums to 1.
    """
    # np.max(logits, axis=1, keepdims=True)
    # - 입력: class 점수 배열
    # - 처리: 각 샘플(row)의 최댓값을 찾습니다.
    # - 출력: (batch_size, 1), row별 빼기가 가능하도록 차원을 유지합니다.
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _looks_like_probabilities(y_pred):
    """Return True when y_pred already appears to be a softmax probability array."""
    if y_pred.ndim != 2:
        return False
    row_sums = np.sum(y_pred, axis=1)
    return np.all(y_pred >= 0) and np.allclose(row_sums, 1.0, atol=1e-5)


def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error averaged over a mini-batch.

    y_pred는 logits여도 되고 softmax probability여도 됩니다.
    logits가 들어오면 이 함수 안에서 stable softmax를 먼저 적용합니다.
    """
    batch_size = y_pred.shape[0]
    probs = y_pred if _looks_like_probabilities(y_pred) else softmax(y_pred)

    # np.clip(probs, eps, 1.0)
    # - 입력: 확률 배열
    # - 처리: eps보다 작은 값은 eps로 올려 log(0)을 막습니다.
    # - 출력: probs와 같은 shape의 배열
    eps = 1e-7
    clipped = np.clip(probs, eps, 1.0)
    correct_log_probs = -np.log(clipped[np.arange(batch_size), y_true])
    return np.mean(correct_log_probs)


def softmax_cross_entropy_gradient(y_pred, y_true):
    """
    Return dLoss/dLogits for Softmax + Cross Entropy.

    Args:
        y_pred: logits or probability array from model.forward(...)
        y_true: (batch_size,) integer labels

    Returns:
        (batch_size, num_classes) gradient array
    """
    batch_size = y_pred.shape[0]
    probs = y_pred if _looks_like_probabilities(y_pred) else softmax(y_pred)
    dout = probs.copy()
    dout[np.arange(batch_size), y_true] -= 1
    dout /= batch_size
    return dout
