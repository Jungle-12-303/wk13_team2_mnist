# MNIST NumPy MLP

PyTorch나 TensorFlow 없이 **NumPy만 사용해 MNIST 손글씨 숫자 분류용 MLP** 를 구현하고, 구조와 학습 설정을 바꿔가며 성능을 비교한 프로젝트입니다.

최종 실험에서 모든 모델이 97% 이상의 test accuracy를 달성했으며, 최고 성능 모델은 `relu_512_256_128`로 **98.56%** 를 기록했습니다. 가장 빠른 고성능 모델은 `batchnorm_off`로 **98.28% / 110.5초** 를 기록했습니다.

## 제출 파일

- `src/`: NumPy 기반 MLP 구현 소스코드
- `REPORT.md`: 과제 제출용 실험 보고서
- `figures/`: 실험 결과 그래프 및 표 이미지
- `mnist_lab.ipynb`: 데이터 로드, 학습, 실험 실행, 결과 시각화 노트북
- `requirements.txt`: 실행에 필요한 Python 패키지 목록

## 프로젝트 구조

```text
.
├── src/
│   ├── activations.py
│   ├── data.py
│   ├── layers.py
│   ├── losses.py
│   ├── network.py
│   ├── optimizers.py
│   └── training.py
├── tests/
├── figures/
├── mnist_lab.ipynb
├── requirements.txt
├── README.md
└── REPORT.md
```

## 주요 구현 내용

| 파일 | 내용 |
| --- | --- |
| `src/network.py` | Affine, BatchNorm, Activation, Dropout을 조합하는 MLP 모델 |
| `src/layers.py` | Affine, BatchNorm, Dropout 계층 구현 |
| `src/activations.py` | ReLU, LeakyReLU, Sigmoid, Tanh, Softmax |
| `src/losses.py` | Cross Entropy 및 Softmax Cross Entropy gradient |
| `src/optimizers.py` | SGD, Momentum, Adam |
| `src/training.py` | mini-batch 학습, 평가, 실험 실행, 결과 저장 |

## 실행 방법

필요 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

노트북을 실행합니다.

```bash
jupyter notebook mnist_lab.ipynb
```

테스트를 실행합니다.

```bash
pytest
```

## 실험 요약

| 모델 | Test accuracy | Params | Time |
| --- | ---: | ---: | ---: |
| `relu_512_256_128` | 98.56% | 569,226 | 301.9s |
| `lr_decay_10_15` | 98.52% | 235,914 | 125.0s |
| `relu_256_128` | 98.41% | 235,914 | 125.3s |
| `batchnorm_off` | 98.28% | 235,146 | 110.5s |
| `relu_1024_512` | 98.26% | 1,336,842 | 622.0s |

자세한 실험 조건, 그래프, 회고는 [REPORT.md](REPORT.md)를 확인하면 됩니다.
