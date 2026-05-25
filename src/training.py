# -*- coding: utf-8 -*-
"""Training, evaluation, plotting, and experiment helpers."""

import csv
import json
import time

import matplotlib.pyplot as plt
import numpy as np

from losses import cross_entropy_loss, softmax_cross_entropy_gradient
from network import MLP
from optimizers import create_optimizer


def check_param_grad_shapes(model):
    """
    Assert that every parameter has a gradient with the same shape.

    Raises:
        AssertionError: when a key is missing or a shape is different.
    """
    for key in model.params:
        if key not in model.grads:
            raise AssertionError(f"Missing gradient for parameter: {key}")
        if model.params[key].shape != model.grads[key].shape:
            raise AssertionError(
                f"Shape mismatch for {key}: "
                f"param={model.params[key].shape}, grad={model.grads[key].shape}"
            )


def _is_bad_loss(loss):
    """Return True when loss is NaN or infinite."""
    return np.isnan(loss) or np.isinf(loss)


def train(
    model,
    optimizer,
    x_train,
    y_train,
    epochs=20,
    batch_size=128,
    lr_decay_epochs=None,
    lr_decay_factor=0.1,
    experiment_name="experiment",
    check_shapes=True,
):
    """
    Mini-batch training loop.

    The required order is preserved:
        1. Forward:  model.forward(x_batch, train=True)
        2. Loss:     cross_entropy_loss(y_pred, y_batch)
        3. Backward: softmax + cross entropy gradient, then model.backward(dout)
        4. Update:   optimizer.update(model.params, model.grads)

    Returns:
        loss_history: epoch-level average train loss list
    """
    lr_decay_epochs = set(lr_decay_epochs or [])
    loss_history = []
    train_size = x_train.shape[0]

    for epoch in range(1, epochs + 1):
        if epoch in lr_decay_epochs:
            optimizer.lr *= lr_decay_factor

        indices = np.random.permutation(train_size)
        epoch_losses = []

        for iteration, start in enumerate(range(0, train_size, batch_size), start=1):
            batch_idx = indices[start : start + batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            y_pred = model.forward(x_batch, train=True)
            loss = cross_entropy_loss(y_pred, y_batch)

            if _is_bad_loss(loss):
                print("[ERROR] loss became NaN/inf")
                print(f"experiment={experiment_name}")
                print(f"epoch={epoch}, iteration={iteration}, lr={optimizer.lr}")
                print(f"recent_loss={loss}")
                return loss_history

            epoch_losses.append(loss)

            dout = softmax_cross_entropy_gradient(y_pred, y_batch)
            model.backward(dout)

            if check_shapes and epoch == 1 and iteration == 1:
                check_param_grad_shapes(model)

            optimizer.update(model.params, model.grads)

        loss_history.append(float(np.mean(epoch_losses)))

    return loss_history


def evaluate(model, x, y):
    """
    Return accuracy (%) and total trainable parameter count.

    Evaluation always calls model.forward(x, train=False) through model.predict(...).
    """
    y_pred = model.predict(x)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == y) * 100
    total_params = sum(p.size for p in model.params.values())
    return accuracy, total_params


def classification_metrics(y_true, y_pred, num_classes=10):
    """
    Return multiclass precision, recall, and F1 metrics in percent.

    Macro averages treat all classes equally. Weighted averages scale each
    class score by its support, which is useful when class counts differ.
    """
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion[int(true_label), int(pred_label)] += 1

    tp = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)

    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )

    total = support.sum()
    weights = support / total if total else np.zeros_like(support)

    return {
        "precision_macro": float(np.mean(precision) * 100),
        "recall_macro": float(np.mean(recall) * 100),
        "f1_macro": float(np.mean(f1) * 100),
        "precision_weighted": float(np.sum(precision * weights) * 100),
        "recall_weighted": float(np.sum(recall * weights) * 100),
        "f1_weighted": float(np.sum(f1 * weights) * 100),
        "per_class_precision": (precision * 100).tolist(),
        "per_class_recall": (recall * 100).tolist(),
        "per_class_f1": (f1 * 100).tolist(),
        "support": support.astype(int).tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_with_metrics(model, x, y, num_classes=10):
    """
    Return accuracy, parameter count, predictions, and classification metrics.
    """
    logits = model.predict(x)
    y_pred = np.argmax(logits, axis=1)
    accuracy = np.mean(y_pred == y) * 100
    total_params = sum(p.size for p in model.params.values())
    metrics = classification_metrics(y, y_pred, num_classes=num_classes)
    metrics.update(
        {
            "accuracy": float(accuracy),
            "params": int(total_params),
            "y_pred": y_pred,
        }
    )
    return metrics


def plot_loss_history(loss_history):
    """Plot one experiment's training loss curve."""
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()


def plot_compare_loss_histories(histories):
    """
    Plot multiple experiment loss curves on one graph.

    Args:
        histories: {"experiment_name": loss_history}
    """
    for name, history in histories.items():
        plt.plot(history, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


DEFAULT_EXPERIMENT_CONFIGS = [
    {
        "name": "baseline",
        "hidden_dims": [256, 128],
        "activation": "relu",
        "use_batchnorm": True,
        "dropout_rate": 0.2,
        "weight_init": "he",
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 20,
        "batch_size": 128,
        "seed": 42,
    },
    {
        "name": "deeper_512_256_128",
        "hidden_dims": [512, 256, 128],
        "activation": "relu",
        "use_batchnorm": True,
        "dropout_rate": 0.2,
        "weight_init": "he",
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 20,
        "batch_size": 128,
        "seed": 42,
    },
    {
        "name": "wider_1024_512",
        "hidden_dims": [1024, 512],
        "activation": "relu",
        "use_batchnorm": True,
        "dropout_rate": 0.2,
        "weight_init": "he",
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 20,
        "batch_size": 128,
        "seed": 42,
    },
    {
        "name": "no_batchnorm",
        "hidden_dims": [256, 128],
        "activation": "relu",
        "use_batchnorm": False,
        "dropout_rate": 0.2,
        "weight_init": "he",
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 20,
        "batch_size": 128,
        "seed": 42,
    },
    {
        "name": "no_dropout",
        "hidden_dims": [256, 128],
        "activation": "relu",
        "use_batchnorm": True,
        "dropout_rate": 0.0,
        "weight_init": "he",
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 20,
        "batch_size": 128,
        "seed": 42,
    },
    {
        "name": "dropout_0_1",
        "hidden_dims": [256, 128],
        "activation": "relu",
        "use_batchnorm": True,
        "dropout_rate": 0.1,
        "weight_init": "he",
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 20,
        "batch_size": 128,
        "seed": 42,
    },
    {
        "name": "dropout_0_5",
        "hidden_dims": [256, 128],
        "activation": "relu",
        "use_batchnorm": True,
        "dropout_rate": 0.5,
        "weight_init": "he",
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 20,
        "batch_size": 128,
        "seed": 42,
    },
    {
        "name": "leaky_relu",
        "hidden_dims": [256, 128],
        "activation": "leaky_relu",
        "use_batchnorm": True,
        "dropout_rate": 0.2,
        "weight_init": "he",
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 20,
        "batch_size": 128,
        "seed": 42,
    },
    {
        "name": "sigmoid_xavier",
        "hidden_dims": [256, 128],
        "activation": "sigmoid",
        "use_batchnorm": True,
        "dropout_rate": 0.2,
        "weight_init": "xavier",
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 20,
        "batch_size": 128,
        "seed": 42,
    },
    {
        "name": "tanh_xavier",
        "hidden_dims": [256, 128],
        "activation": "tanh",
        "use_batchnorm": True,
        "dropout_rate": 0.2,
        "weight_init": "xavier",
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 20,
        "batch_size": 128,
        "seed": 42,
    },
    {
        "name": "adam_lr_0_0005",
        "hidden_dims": [256, 128],
        "activation": "relu",
        "use_batchnorm": True,
        "dropout_rate": 0.2,
        "weight_init": "he",
        "optimizer": "adam",
        "lr": 0.0005,
        "epochs": 20,
        "batch_size": 128,
        "seed": 42,
    },
    {
        "name": "batch_size_64",
        "hidden_dims": [256, 128],
        "activation": "relu",
        "use_batchnorm": True,
        "dropout_rate": 0.2,
        "weight_init": "he",
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 20,
        "batch_size": 64,
        "seed": 42,
    },
    {
        "name": "batch_size_256",
        "hidden_dims": [256, 128],
        "activation": "relu",
        "use_batchnorm": True,
        "dropout_rate": 0.2,
        "weight_init": "he",
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 20,
        "batch_size": 256,
        "seed": 42,
    },
    {
        "name": "adam_lr_decay",
        "hidden_dims": [256, 128],
        "activation": "relu",
        "use_batchnorm": True,
        "dropout_rate": 0.2,
        "weight_init": "he",
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 20,
        "batch_size": 128,
        "lr_decay_epochs": [10, 15],
        "lr_decay_factor": 0.1,
        "seed": 42,
    },
]


def _activation_label(activation):
    return ",".join(activation) if isinstance(activation, (list, tuple)) else activation


def run_experiment(config, x_train, y_train, x_test, y_test):
    """
    Build a model/optimizer from config, train it, evaluate it, and return a result dict.
    """
    seed = config.get("seed", 42)
    np.random.seed(seed)

    model = MLP(
        input_dim=config.get("input_dim", 784),
        hidden_dims=config["hidden_dims"],
        output_dim=config.get("output_dim", 10),
        activation=config.get("activation", "relu"),
        use_batchnorm=config.get("use_batchnorm", True),
        dropout_rate=config.get("dropout_rate", 0.0),
        weight_init=config.get("weight_init", "he"),
        seed=seed,
    )
    optimizer = create_optimizer(
        config.get("optimizer", "adam"),
        lr=config.get("lr", 0.001),
        momentum=config.get("momentum", 0.9),
    )

    started = time.time()
    loss_history = train(
        model,
        optimizer,
        x_train,
        y_train,
        epochs=config.get("epochs", 20),
        batch_size=config.get("batch_size", 128),
        lr_decay_epochs=config.get("lr_decay_epochs"),
        lr_decay_factor=config.get("lr_decay_factor", 0.1),
        experiment_name=config.get("name", "experiment"),
    )
    time_sec = time.time() - started

    train_eval = evaluate_with_metrics(model, x_train, y_train)
    test_eval = evaluate_with_metrics(model, x_test, y_test)
    train_acc = train_eval["accuracy"]
    test_acc = test_eval["accuracy"]
    params = test_eval["params"]

    result = {
        "name": config.get("name", "experiment"),
        "architecture": model.architecture,
        "hidden_layers": len(config["hidden_dims"]),
        "hidden_dims": "-".join(str(dim) for dim in config["hidden_dims"]),
        "activation": _activation_label(config.get("activation", "relu")),
        "init": config.get("weight_init", "he"),
        "optimizer": config.get("optimizer", "adam"),
        "lr": config.get("lr", 0.001),
        "epochs": config.get("epochs", 20),
        "batch_size": config.get("batch_size", 128),
        "batchnorm": config.get("use_batchnorm", True),
        "dropout": config.get("dropout_rate", 0.0),
        "train_loss": loss_history[-1] if loss_history else None,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_precision_macro": train_eval["precision_macro"],
        "train_recall_macro": train_eval["recall_macro"],
        "train_f1_macro": train_eval["f1_macro"],
        "test_precision_macro": test_eval["precision_macro"],
        "test_recall_macro": test_eval["recall_macro"],
        "test_balanced_acc": test_eval["recall_macro"],
        "test_f1_macro": test_eval["f1_macro"],
        "test_precision_weighted": test_eval["precision_weighted"],
        "test_recall_weighted": test_eval["recall_weighted"],
        "test_f1_weighted": test_eval["f1_weighted"],
        "test_per_class_precision": test_eval["per_class_precision"],
        "test_per_class_recall": test_eval["per_class_recall"],
        "test_per_class_f1": test_eval["per_class_f1"],
        "test_support": test_eval["support"],
        "test_confusion_matrix": test_eval["confusion_matrix"],
        "params": params,
        "time_sec": time_sec,
        "loss_history": loss_history,
    }
    return result


def save_results(results, csv_path="results.csv", json_path="results.json"):
    """Save experiment results to CSV and JSON."""
    summary_keys = [
        "name",
        "architecture",
        "hidden_layers",
        "hidden_dims",
        "activation",
        "optimizer",
        "lr",
        "epochs",
        "batch_size",
        "batchnorm",
        "dropout",
        "init",
        "train_loss",
        "train_acc",
        "test_acc",
        "test_precision_macro",
        "test_recall_macro",
        "test_balanced_acc",
        "test_f1_macro",
        "test_precision_weighted",
        "test_recall_weighted",
        "test_f1_weighted",
        "params",
        "time_sec",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_keys)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in summary_keys})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def summarize_experiment_results(results, baseline_name="baseline"):
    """Print best, efficient, and baseline-comparison summaries."""
    if not results:
        print("No experiment results.")
        return

    best = max(results, key=lambda item: item["test_acc"])
    print("Best accuracy model:")
    print(f"name={best['name']}, test_acc={best['test_acc']:.2f}%")

    over_97 = [item for item in results if item["test_acc"] >= 97.0]
    if over_97:
        efficient = min(over_97, key=lambda item: item["params"])
        print("\nMost efficient model over 97%:")
        print(
            f"name={efficient['name']}, params={efficient['params']}, "
            f"test_acc={efficient['test_acc']:.2f}%"
        )
    else:
        print("\nMost efficient model over 97%:")
        print("No model reached 97% test accuracy.")

    baseline = next((item for item in results if item["name"] == baseline_name), None)
    if baseline is None:
        return

    print("\nCompared with baseline:")
    for item in results:
        if item["name"] == baseline_name:
            continue
        print(
            f"{item['name']}: "
            f"acc_diff={item['test_acc'] - baseline['test_acc']:+.2f}%, "
            f"param_diff={item['params'] - baseline['params']:+d}, "
            f"time_diff={item['time_sec'] - baseline['time_sec']:+.2f}s"
        )


def run_experiments(
    configs,
    x_train,
    y_train,
    x_test,
    y_test,
    csv_path="results.csv",
    json_path="results.json",
):
    """Run many configs, save results, plot comparable loss curves, and print summary."""
    results = []
    histories = {}

    for config in configs:
        print(f"[RUN] {config['name']}")
        result = run_experiment(config, x_train, y_train, x_test, y_test)
        results.append(result)
        histories[result["name"]] = result["loss_history"]
        print(
            f"  train_acc={result['train_acc']:.2f}% "
            f"test_acc={result['test_acc']:.2f}% "
            f"recall={result['test_recall_macro']:.2f}% "
            f"f1={result['test_f1_macro']:.2f}% "
            f"time={result['time_sec']:.2f}s"
        )

    save_results(results, csv_path=csv_path, json_path=json_path)
    summarize_experiment_results(results)
    plot_compare_loss_histories(histories)
    return results, histories
