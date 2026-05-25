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

    # np.clip(y_pred, min, max)
    # - 입력: 확률 배열 y_pred
    # - 처리: min보다 작으면 min, max보다 크면 max로 잘라냄
    # - 출력: y_pred와 같은 shape의 배열
    # log(0)은 -inf가 되므로 아주 작은 값으로 확률 범위를 제한합니다.
    clipped = np.clip(y_pred, 1e-7, 1.0)

    # np.arange(batch_size)
    # - 입력: 정수 batch_size
    # - 처리: 0부터 batch_size - 1까지의 정수 배열을 만듦
    # - 출력: (batch_size,) 배열. 각 샘플의 row index로 사용합니다.
    #
    # np.log(...)
    # - 입력: 정답 클래스 확률 배열
    # - 처리: 각 확률에 자연로그를 적용
    # - 출력: 입력과 같은 shape의 로그값 배열
    # 각 샘플에서 정답 클래스 확률만 골라 log를 취합니다.
    correct_log_probs = -np.log(clipped[np.arange(batch_size), y_true])
    # np.mean(correct_log_probs)
    # - 입력: 각 샘플의 loss 배열
    # - 처리: 전체 평균을 계산
    # - 출력: scalar loss 하나
    return np.mean(correct_log_probs)
