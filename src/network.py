# -*- coding: utf-8 -*-
"""Configurable NumPy MLP for MNIST classification."""

from collections import OrderedDict

import numpy as np

from activations import get_activation
from layers import Affine, BatchNorm, Dropout
from losses import cross_entropy_loss


class MLP:
    """
    Flexible multi-layer perceptron.

    Hidden layer order:
        Affine -> BatchNorm(optional) -> Activation -> Dropout(optional)

    Output layer:
        Affine -> logits

    Softmax and Cross Entropy are handled in losses.py, so forward returns logits.
    """

    def __init__(
        self,
        input_dim=784,
        hidden_dims=None,
        output_dim=10,
        activation="relu",
        use_batchnorm=True,
        dropout_rate=0.2,
        weight_init="he",
        seed=None,
        batchnorm_momentum=0.9,
    ):
        self.input_dim = input_dim
        self.hidden_dims = [512, 256] if hidden_dims is None else list(hidden_dims)
        self.output_dim = output_dim
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.weight_init = weight_init
        self.seed = seed
        self.batchnorm_momentum = batchnorm_momentum

        if seed is not None:
            np.random.seed(seed)

        self.params = {}
        self.grads = {}
        self.layers = OrderedDict()
        self.layer_dims = [input_dim] + self.hidden_dims + [output_dim]
        self.activation_names = self._normalize_activations(activation)

        self._init_params()
        self._build_layers()
        self.grads = {key: np.zeros_like(value) for key, value in self.params.items()}

    def _normalize_activations(self, activation):
        """Allow one activation string or one activation per hidden layer."""
        if isinstance(activation, str):
            return [activation] * len(self.hidden_dims)

        activation_names = list(activation)
        if len(activation_names) != len(self.hidden_dims):
            raise ValueError("activation list length must match hidden_dims length.")
        return activation_names

    def _weight_scale(self, fan_in, fan_out):
        """Return std scale for the selected initialization strategy."""
        init = self.weight_init.lower()
        if init == "he":
            return np.sqrt(2.0 / fan_in)
        if init == "xavier":
            return np.sqrt(2.0 / (fan_in + fan_out))
        if init == "normal_small":
            return 0.01
        if init == "normal_large":
            return 1.0
        raise ValueError(f"Unknown weight_init: {self.weight_init}")

    def _init_params(self):
        """Create W/b for every Affine layer and gamma/beta for BatchNorm layers."""
        for idx in range(1, len(self.layer_dims)):
            fan_in = self.layer_dims[idx - 1]
            fan_out = self.layer_dims[idx]
            scale = self._weight_scale(fan_in, fan_out)

            self.params[f"W{idx}"] = np.random.randn(fan_in, fan_out) * scale
            self.params[f"b{idx}"] = np.zeros(fan_out)

            is_hidden = idx < len(self.layer_dims) - 1
            if is_hidden and self.use_batchnorm:
                self.params[f"gamma{idx}"] = np.ones(fan_out)
                self.params[f"beta{idx}"] = np.zeros(fan_out)

    def _build_layers(self):
        """Build OrderedDict so forward/backward order is explicit and reproducible."""
        for idx in range(1, len(self.layer_dims)):
            self.layers[f"Affine{idx}"] = Affine(
                self.params[f"W{idx}"], self.params[f"b{idx}"]
            )

            is_hidden = idx < len(self.layer_dims) - 1
            if not is_hidden:
                continue

            if self.use_batchnorm:
                self.layers[f"BatchNorm{idx}"] = BatchNorm(
                    self.params[f"gamma{idx}"],
                    self.params[f"beta{idx}"],
                    momentum=self.batchnorm_momentum,
                )
            self.layers[f"Activation{idx}"] = get_activation(self.activation_names[idx - 1])
            if self.dropout_rate > 0.0:
                self.layers[f"Dropout{idx}"] = Dropout(self.dropout_rate)

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, input_dim)
            train: BatchNorm/Dropout mode flag

        Returns:
            (batch_size, output_dim) logits
        """
        out = x
        for layer in self.layers.values():
            if isinstance(layer, (BatchNorm, Dropout)):
                out = layer.forward(out, train=train)
            else:
                out = layer.forward(out)
        return out

    def backward(self, dout):
        """Run backpropagation and collect gradients into self.grads."""
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        for name, layer in self.layers.items():
            if isinstance(layer, Affine):
                idx = name.replace("Affine", "")
                self.grads[f"W{idx}"] = layer.dW
                self.grads[f"b{idx}"] = layer.db
            elif isinstance(layer, BatchNorm):
                idx = name.replace("BatchNorm", "")
                self.grads[f"gamma{idx}"] = layer.dgamma
                self.grads[f"beta{idx}"] = layer.dbeta

        return dout

    def loss(self, x, y):
        """Return cross entropy loss from logits."""
        logits = self.forward(x, train=True)
        return cross_entropy_loss(logits, y)

    def predict(self, x):
        """Return logits in inference mode. argmax(logits) is the predicted class."""
        return self.forward(x, train=False)

    @property
    def architecture(self):
        """Human-readable architecture string such as 784-256-128-10."""
        return "-".join(str(dim) for dim in self.layer_dims)


class NeuralNetwork(MLP):
    """
    Backward-compatible wrapper for older code/tests.

    The old default structure was 784 -> 512 -> 256 -> 10.
    """

    def __init__(self, use_batchnorm=True, use_dropout=True, dropout_ratio=0.5):
        super().__init__(
            input_dim=784,
            hidden_dims=[512, 256],
            output_dim=10,
            activation="relu",
            use_batchnorm=use_batchnorm,
            dropout_rate=dropout_ratio if use_dropout else 0.0,
            weight_init="he",
        )
