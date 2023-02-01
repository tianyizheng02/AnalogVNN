from typing import Optional

from torch import Tensor

from analogvnn.nn.Linear import LinearBackpropagation


class LinearFeedbackAlignment(LinearBackpropagation):
    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        return NotImplemented
