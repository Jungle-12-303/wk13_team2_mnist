# -*- coding: utf-8 -*-
"""Training, evaluation, and plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np

from losses import cross_entropy_loss


def train(model, optimizer, x_train, y_train, epochs=20, batch_size=128):
    """
    Mini-batch training loop.

    한 batch마다 다음 순서로 학습합니다.
    1. forward: 현재 파라미터로 예측 확률 계산
    2. loss: 예측이 정답과 얼마나 다른지 계산
    3. backward: loss를 줄이려면 각 파라미터를 어느 방향으로 바꿀지 계산
    4. update: optimizer가 실제 파라미터 값을 갱신

    Returns:
        loss_history: epoch별 평균 loss list
    """
    loss_history = []
    train_size = x_train.shape[0]

    for _ in range(epochs):
        # np.random.permutation(train_size)
        # - 입력: 데이터 개수 train_size
        # - 처리: 0부터 train_size - 1까지의 index를 무작위 순서로 섞음
        # - 출력: (train_size,) 정수 배열
        indices = np.random.permutation(train_size)
        epoch_losses = []

        for start in range(0, train_size, batch_size):
            batch_idx = indices[start : start + batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            y_pred = model.forward(x_batch, train=True)
            loss = cross_entropy_loss(y_pred, y_batch)
            epoch_losses.append(loss)

            # Softmax + CrossEntropy를 합치면 gradient는 y_pred - one_hot(y)가 됩니다.
            dout = y_pred.copy()
            # np.arange(x_batch.shape[0])
            # - 입력: 현재 batch 크기
            # - 처리: 0, 1, ..., batch_size - 1 index 생성
            # - 출력: (batch_size,) 배열. y_batch와 함께 정답 위치를 고르는 데 씁니다.
            dout[np.arange(x_batch.shape[0]), y_batch] -= 1
            dout /= x_batch.shape[0]

            model.backward(dout)
            optimizer.update(model.params, model.grads)

        # np.mean(epoch_losses)는 이번 epoch의 batch loss 평균 scalar를 반환합니다.
        loss_history.append(float(np.mean(epoch_losses)))

    return loss_history


def evaluate(model, x, y):
    """Return accuracy (%) and the total number of trainable parameters."""
    y_pred = model.predict(x)
    # np.argmax(y_pred, axis=1)
    # - 입력: (N, num_classes) 확률 배열
    # - 처리: 각 row에서 가장 큰 확률의 class index를 찾음
    # - 출력: (N,) 예측 label 배열
    #
    # np.mean(bool_array)
    # - 입력: True/False 배열
    # - 처리: True를 1, False를 0처럼 보고 평균을 계산
    # - 출력: 맞춘 비율 scalar
    accuracy = np.mean(np.argmax(y_pred, axis=1) == y) * 100
    total_params = sum(p.size for p in model.params.values())
    return accuracy, total_params


def plot_loss_history(loss_history):
    """Plot the training loss curve."""
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()
