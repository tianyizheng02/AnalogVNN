import abc

from torch import Tensor, nn

from analogvnn.backward.BackwardModule import BackwardModule
from analogvnn.nn.module.Layer import Layer


class InitImplement:
    @staticmethod
    def initialise(tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform(tensor)

    @staticmethod
    def initialise_(tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform_(tensor)


class Activation(Layer, BackwardModule, InitImplement, metaclass=abc.ABCMeta):
    pass