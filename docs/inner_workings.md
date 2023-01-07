# Inner Workings

There are three major new classes in AnalogVNN, which are as follows

## PseudoParameters

"PseudoParameters" is a subclass of "Parameter" class of PyTorch.

"PseudoParameters" class lets you convent a digital parameter to analog parameter by converting
the parameter of layer of "Parameter" class to "PseudoParameters".

PseudoParameters requires a ParameterizingModel to parameterize the parameters (weights and biases) of the
layer to get parameterized data

PyTorch's ParameterizedParameters vs AnalogVNN's PseudoParameters:
- Similarity (Forward or Parameterizing the data):
    > Data -> ParameterizingModel -> Parameterized Data
- Difference (Backward or Gradient Calculations):
  - ParameterizedParameters
    > Parameterized Data -> ParameterizingModel -> Data
  - PseudoParameters 
    > Parameterized Data -> Data

So, by using PseudoParameters class the gradients of the parameter are calculated in such a way that
the ParameterizingModel was never present.

To convert parameters of a layer or model to use PseudoParameters, then use:

  ```python
    PseudoParameters.parameterize(Model, "parameter_name", transformation=ParameterizingModel)
  ```

OR

  ```python
    PseudoParameters.parametrize_module(Model, transformation=ParameterizingModel)
  ```

## Forward and Backward Graphs

Documentation Coming Soon...