# -*- coding: utf-8 -*-
"""
MNIST 데이터 로드 유틸리티.

데이터 파일이 로컬에 없으면 TensorFlow/Keras 공개 URL에서 내려받아 `data/`에 저장합니다.
"""

import os
import urllib.request

import numpy as np


def load_mnist(data_dir="data"):
    """
    MNIST 손글씨 숫자 데이터셋을 로드합니다.
    data/mnist.npz가 있으면 로컬 파일을 사용하고, 없으면 URL에서 다운로드 후 data/에 저장합니다.

    Returns:
        (x_train, y_train), (x_test, y_test)
        - x: (N, 784) float32, 0~1 정규화
        - y: (N,) int, 0~9 레이블
    """
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, "mnist.npz")

    if not os.path.isfile(local_path):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        urllib.request.urlretrieve(url, local_path)

    # np.load(local_path)
    # - 입력: .npz 또는 .npy 파일 경로
    # - 처리: NumPy가 저장한 배열 파일을 읽음
    # - 출력: 여기서는 dict처럼 key로 배열을 꺼낼 수 있는 NpzFile 객체
    with np.load(local_path) as data:
        # astype(np.float32)
        # - 입력: 원본 이미지 배열
        # - 처리: 배열 원소 타입을 float32로 변환
        # - 출력: 같은 shape의 float32 배열
        #
        # reshape(-1, 784)
        # - 입력: (N, 28, 28) 이미지 배열
        # - 처리: N은 자동으로 맞추고, 각 이미지를 784칸짜리 벡터로 펼침
        # - 출력: (N, 784) 배열
        x_train = data["x_train"].astype(np.float32).reshape(-1, 784) / 255.0
        x_test = data["x_test"].astype(np.float32).reshape(-1, 784) / 255.0
        y_train = data["y_train"]
        y_test = data["y_test"]

    return (x_train, y_train), (x_test, y_test)
