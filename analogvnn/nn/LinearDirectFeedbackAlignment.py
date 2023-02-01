from typing import Optional

from torch import Tensor, mm

from analogvnn.fn.to_matrix import to_matrix
from analogvnn.nn.LinearFeedbackAlignment import LinearFeedbackAlignment


class LinearDirectFeedbackAlignment(LinearFeedbackAlignment):
    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        grad_output = to_matrix(grad_output)

        weight = to_matrix(self.get_fixed_matrix(size=(grad_output.size()[-1], self.weight.size()[0])))
        grad_input = grad_output @ weight

        self.set_grad_of(self.weight, mm(grad_output.t(), self.inputs))
        self.set_grad_of(self.bias, grad_output.sum(0))
        return grad_input
