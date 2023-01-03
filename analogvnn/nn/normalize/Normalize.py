from abc import ABC

from analogvnn.backward.BackwardIdentity import BackwardIdentity
from analogvnn.nn.module.Layer import Layer


class Normalize(Layer, BackwardIdentity, ABC):
    pass
