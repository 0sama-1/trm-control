import torch
import torch.nn as nn

from utils import fanin_init, layer_init

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu":  nn.ELU,
    "silu": nn.SiLU,
}


def _apply_init(layer: nn.Linear, init_type: str) -> nn.Linear:
    if init_type == "fanin":
        return fanin_init(layer)
    if init_type == "orthogonal":
        return layer_init(layer)
    if init_type == "uniform_small":
        nn.init.uniform_(layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(layer.bias,   -3e-3, 3e-3)
        return layer
    raise ValueError(f"Unknown init: {init_type!r}. Choose from: fanin, orthogonal, uniform_small")


class DNNTrunk(nn.Module):
    """Configurable MLP backbone. Accepts flat kwargs so Hydra can instantiate it
    directly via _target_ without the caller needing to know the concrete class."""

    def __init__(self, in_dim: int, hidden_sizes: list[int], activation: str, hidden_init: str, **_):
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation)
        if act_cls is None:
            raise ValueError(
                f"Unknown activation: {activation!r}. "
                f"Choose from: {list(_ACTIVATIONS)}"
            )

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            linear = _apply_init(nn.Linear(prev, h), hidden_init)
            layers.extend([linear, act_cls()])
            prev = h

        self.net = nn.Sequential(*layers)
        self._out_dim = prev

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_head(in_dim: int, out_dim: int, output_init: str) -> nn.Linear:
    """Output head with a named weight initialisation scheme."""
    return _apply_init(nn.Linear(in_dim, out_dim), output_init)
