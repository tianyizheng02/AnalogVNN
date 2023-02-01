from typing import Optional

import torch
from torch import Tensor

from analogvnn.fn.to_matrix import to_matrix
from analogvnn.nn.Linear import LinearBackpropagation


class LinearFeedbackAlignment(LinearBackpropagation):
    fixed_weights = None

    def get_fixed_matrix(self, size=None, random: bool = False) -> Tensor:
        if size is None:
            size = self.weight.size()
        if self.fixed_weights is None or random:
            self.fixed_weights = torch.rand(*size)
        return self.fixed_weights

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        """Backward pass of the linear layer.

        Args:
            grad_output (Optional[Tensor]): The gradient of the output.

        Returns:
            Optional[Tensor]: The gradient of the input.
        """

        grad_output = to_matrix(grad_output)

        weight = to_matrix(self.get_fixed_matrix())
        grad_input = grad_output @ weight

        self.set_grad_of(self.weight, torch.mm(grad_output.t(), self.inputs))
        self.set_grad_of(self.bias, grad_output.sum(0))
        return grad_input
