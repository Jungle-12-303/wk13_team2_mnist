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
            dout[np.arange(x_batch.shape[0]), y_batch] -= 1
            dout /= x_batch.shape[0]

            model.backward(dout)
            optimizer.update(model.params, model.grads)

        loss_history.append(float(np.mean(epoch_losses)))

    return loss_history


def evaluate(model, x, y):
    """Return accuracy (%) and the total number of trainable parameters."""
    y_pred = model.predict(x)
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
