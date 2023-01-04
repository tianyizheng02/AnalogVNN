from typing import Union, Optional

import torch
from torch import Tensor, nn

from analogvnn.backward.BackwardIdentity import BackwardIdentity
from analogvnn.nn.module.Layer import Layer
from analogvnn.utils.to_tensor_parameter import to_float_tensor, to_nongrad_parameter


class UniformNoise(Layer, BackwardIdentity):
    __constants__ = ['low', 'high', 'leakage', 'precision']
    low: nn.Parameter
    high: nn.Parameter
    leakage: nn.Parameter
    precision: nn.Parameter

    def __init__(
            self,
            low: Optional[float] = None,
            high: Optional[float] = None,
            leakage: Optional[float] = None,
            precision: Optional[int] = None
    ):
        super(UniformNoise, self).__init__()

        if (low is None or high is None) + (leakage is None) + (precision is None) != 1:
            raise ValueError("only 2 out of 3 arguments are needed (scale, leakage, precision)")

        self.low, self.high, self.leakage, self.precision = to_float_tensor(
            low, high, leakage, precision
        )

        if (self.low is None or self.high is None) and self.leakage is not None and self.precision is not None:
            self.low, self.high = self.calc_high_low(self.leakage, self.precision)

        if self.low is not None and self.high is not None and self.leakage is None and self.precision is not None:
            self.leakage = self.calc_leakage(self.low, self.high, self.precision)

        if self.low is not None and self.high is not None and self.leakage is not None and self.precision is None:
            self.precision = self.calc_precision(self.low, self.high, self.precision)

        self.low, self.high, self.leakage, self.precision = to_nongrad_parameter(
            self.low, self.high, self.leakage, self.precision
        )

    @staticmethod
    def calc_high_low(leakage, precision):
        v = 1 / (1 - leakage) * (1 / precision)
        return -v / 2, v / 2

    @staticmethod
    def calc_precision(low, high, leakage):
        return 1 / (1 - leakage) * (1 / (high - low))

    @staticmethod
    def calc_leakage(low, high, precision):
        return 1 - min(1, (1 / precision) * (1 / (high - low)))

    @property
    def mean(self):
        return (self.high + self.low) / 2

    @property
    def stddev(self):
        return (self.high - self.low) / 12 ** 0.5

    @property
    def variance(self):
        return (self.high - self.low).pow(2) / 12

    def pdf(self, x: Tensor) -> Tensor:
        return torch.exp(self.log_prob(x=x))

    def log_prob(self, x: Tensor) -> Tensor:
        lb = self.low.le(x).type_as(self.low)
        ub = self.high.gt(x).type_as(self.low)
        return torch.log(lb.mul(ub)) - torch.log(self.high - self.low)

    def cdf(self, x: Tensor) -> Union[Tensor, float]:
        result = (x - self.low) / (self.high - self.low)
        return result.clamp(min=0, max=1)

    def forward(self, x: Tensor) -> Tensor:
        return torch.distributions.Uniform(low=x + self.low, high=x + self.high).sample()

    def extra_repr(self) -> str:
        return f'high={float(self.high):.4f}' \
               f', low={float(self.low):.4f}' \
               f', leakage={float(self.leakage):.4f}' \
               f', precision={int(self.precision)}'