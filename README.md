# MNIST NumPy MLP 실험

PyTorch나 TensorFlow 없이 NumPy만 사용해서 MNIST 손글씨 숫자 분류 모델을 구현하고, 여러 모델 조합을 실험하는 프로젝트입니다.

이번 실험의 목표는 기존 구성 요소에 Sigmoid 활성화 함수를 추가한 뒤, 모델 구조와 학습 설정을 바꿔 보면서 test accuracy 95% 이상, 가능하면 97% 이상을 달성하는 것입니다.

## 실행 파일

- `mnist_lab.ipynb`: 데이터 로드, 단일 학습, 여러 실험 실행, 결과표와 그래프 출력
- `src/network.py`: Affine, BatchNorm, Activation, Dropout을 조합하는 MLP 모델
- `src/activations.py`: ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
- `src/layers.py`: Affine, BatchNorm, Dropout
- `src/losses.py`: Softmax, Cross Entropy, Softmax+Cross Entropy gradient
- `src/optimizers.py`: SGD, Momentum, Adam
- `src/training.py`: 학습 루프, 평가, 실험 실행, 결과 저장, loss curve 비교
- `data/mnist.npz`: MNIST 데이터 파일

## 학습 루프 규칙

모든 실험은 `training.train()` 함수의 미니배치 학습 루프를 사용합니다. 각 미니배치에서는 아래 순서를 지킵니다.

1. Forward: `model.forward(x_batch, train=True)`
2. Loss: `cross_entropy_loss(y_pred, y_batch)`
3. Backward: `softmax_cross_entropy_gradient(y_pred, y_batch)` 계산 후 `model.backward(dout)`
4. Update: `optimizer.update(model.params, model.grads)`

테스트와 평가 단계에서는 `evaluate()`가 `model.predict()`를 호출하고, 내부적으로 `train=False`로 forward를 실행합니다. 따라서 Dropout은 꺼지고 BatchNorm은 학습 중 저장된 running mean과 running variance를 사용합니다.

Softmax와 Cross Entropy는 수치 안정성을 위해 max-shift와 epsilon clipping을 적용합니다. 첫 번째 미니배치에서는 파라미터와 gradient shape가 같은지도 확인합니다.

## 실험 구성

노트북의 **10. Multi Experiment Suite** 구역에서 여러 실험을 따로 실행할 수 있습니다. `SELECTED_GROUP` 값만 바꾸면 원하는 실험군을 선택할 수 있습니다.

```python
SELECTED_GROUP = "quick"
```

선택 가능한 실험군은 다음과 같습니다.

| 실험군 | 내용 |
| --- | --- |
| `quick` | ReLU와 Sigmoid 모델을 짧게 실행해서 전체 코드 흐름 확인 |
| `architecture` | 은닉층 수와 차원 변경: `[256, 128]`, `[512, 256, 128]`, `[1024, 512]` 등 |
| `dropout` | Dropout 비율 `0.0`, `0.1`, `0.2`, `0.5` 비교 |
| `learning_rate` | learning rate `0.0005`, `0.001`, `0.002`, learning rate decay 비교 |
| `batchnorm` | BatchNorm 적용 전후를 ReLU와 Sigmoid에서 비교 |
| `full` | 위 실험을 모두 실행 |

Sigmoid 모델은 은닉층 activation을 `"sigmoid"`로 설정하고, 초기화는 Sigmoid에 더 잘 맞는 `"xavier"`를 사용하도록 구성했습니다. ReLU 모델은 `"he"` 초기화를 기본으로 사용합니다.

## 기록 항목

각 실험은 다음 항목을 결과로 저장합니다.

| 항목 | 설명 |
| --- | --- |
| 모델 구조 | `architecture`, 은닉층 수, 각 층 차원, activation |
| 학습 설정 | optimizer, learning rate, epochs, batch size |
| 정규화/규제 | BatchNorm 사용 여부, Dropout 비율 |
| 초기화 | He, Xavier 등 |
| 결과 | train loss, train accuracy, test accuracy, 파라미터 수 |
| 시간 | 학습에 걸린 시간 `time_sec` |

실험 실행 후 결과는 아래 파일로 저장됩니다.

- `results_<실험군>.csv`
- `results_<실험군>.json`

예를 들어 `SELECTED_GROUP = "dropout"`으로 실행하면 `results_dropout.csv`, `results_dropout.json`이 생성됩니다.

## 그래프

노트북의 시각화 셀은 결과를 보기 좋게 비교할 수 있도록 다음 그래프를 출력하고 `figures/` 폴더에 PNG로 저장합니다.

- epoch별 loss curve
- 실험별 test accuracy bar chart
- learning rate 변경에 따른 accuracy 비교
- Dropout 적용 전후 및 비율별 accuracy 비교

보고서에는 노트북 출력 화면을 캡처하거나 `figures/`에 저장된 PNG 파일을 첨부하면 됩니다.

## 실행 순서

1. `mnist_lab.ipynb`를 엽니다.
2. 환경 설정 셀을 실행합니다.
3. 데이터 로드 셀을 실행합니다.
4. 단일 모델 학습을 먼저 실행해 기본 동작을 확인합니다.
5. **10. Multi Experiment Suite**에서 `SELECTED_GROUP`을 선택합니다.
6. 실험 실행 셀을 실행합니다.
7. 결과표와 그래프 셀을 실행합니다.

빠르게 확인할 때는 `quick`을 먼저 실행하고, 최종 비교용으로는 `architecture`, `dropout`, `learning_rate`, `batchnorm` 또는 `full`을 실행하면 됩니다.

## Colab 또는 로컬 실행

필요 패키지는 `requirements.txt`로 설치합니다.

```bash
pip install -r requirements.txt
```

로컬에서 실행할 때는 프로젝트 루트에서 Jupyter Notebook을 켠 뒤 `mnist_lab.ipynb`를 실행하면 됩니다.

```bash
jupyter notebook mnist_lab.ipynb
```
