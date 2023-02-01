from __future__ import annotations

from collections.abc import Callable, Sequence
from math import ceil, sqrt

import torch
from thop import profile
from torch import nn
from torchinfo.torchinfo import summary

from analogvnn.nn.Linear import Linear
from analogvnn.nn.module.FullSequential import FullSequential

k1: float = 0.1
k2: float = 0.1


def calculate_mzi_latency(m: int, n: int, p: int, f_mod: float = 10 ** 9, tau_mzi: float = 10 ** -6,
                          tau_gpu: float = 10 ** -6) -> float:
    avg_k = (k1 + k2) / 2
    return (p / f_mod + avg_k * tau_mzi) * ceil(m / k1) * ceil(n / k2) + tau_gpu * m * p * ceil(n / k2)


def calculate_crossbar_latency(m: int, n: int, p: int, f_mod: float = 10 ** 9) -> float:
    avg_k = (k1 + k2) / 2
    tau_read = avg_k / f_mod
    return (n / f_mod + tau_read) * ceil(m / k1) * ceil(p / k2)


def linear_photonics_latency(self: nn.Linear | Linear, input_size: Sequence[int],
                             latency_function: Callable = calculate_mzi_latency) -> float:
    # print(f"linear_photonics_latency: {input_size}", flush=True)

    return latency_function(m=self.out_features, n=self.in_features, p=input_size[-2])


def conv_photonics_latency(self: nn.Conv2d, input_size: Sequence[int],
                           latency_function: Callable = calculate_mzi_latency) -> float:
    # print(f"conv_photonics_latency: {input_size}", flush=True)

    scale_factor = self.out_channels * self.in_channels / self.groups
    return scale_factor * latency_function(m=self.kernel_size[0], n=self.kernel_size[1], p=self.kernel_size[0])


def photonics_latency(self: nn.Module, input_size: Sequence[int],
                      latency_function: Callable = calculate_mzi_latency) -> float:
    # print(f"photonics_latency: {input_size}", flush=True)

    # using pip --- thop
    # TODO: check if layer_info.macs from summary function returns correct mac count
    #       if so then use that instead of thop

    macs, _ = profile(self, torch.zeros(input_size))
    m = n = ceil(sqrt(macs / (k1 * k2)))

    return latency_function(m=m, n=n, p=input_size[-2])


nn.Module.photonics_latency = photonics_latency
Linear.photonics_latency = linear_photonics_latency
nn.Linear.photonics_latency = linear_photonics_latency
nn.Conv2d.photonics_latency = conv_photonics_latency


class LinearModel(FullSequential):
    def __init__(self, activation_class):
        super(LinearModel, self).__init__()

        self.activation_class = activation_class
        self.all_layers = []
        self.all_layers.append(nn.Flatten(start_dim=1))
        self.add_layer(Linear(in_features=28 * 28, out_features=256))
        self.add_layer(Linear(in_features=256, out_features=128))
        self.add_layer(Linear(in_features=128, out_features=10))

        self.add_sequence(*self.all_layers)

    def add_layer(self, layer):
        self.all_layers.append(layer)
        self.all_layers.append(self.activation_class())


def calculate_photonics_of(module: nn.Module, input_size: Sequence[int]) -> float:
    # print(f"calculate_photonics_of: {input_size}", flush=True)

    layer_info_list = summary(module, input_size, verbose=0, depth=5).summary_list
    latencies = []
    for layer_info in layer_info_list:
        layer_latency = layer_info.module.photonics_latency(layer_info.input_size, calculate_crossbar_latency)
        print(f"{layer_info.module} latency: {layer_latency}")
        latencies.append(layer_latency)

    return sum(latencies)


def main() -> None:
    model = LinearModel(activation_class=nn.ReLU)
    model.compile()
    latency = calculate_photonics_of(model, (1, 1, 28, 28))
    print(f"Photonic Latency: {latency} secs")


if __name__ == '__main__':
    main()
