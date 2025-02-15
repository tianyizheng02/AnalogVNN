# Tutorial

<a href="https://colab.research.google.com/github/Vivswan/AnalogVNN/blob/v1.0.0/docs/_static/AnalogVNN_Demo.ipynb" style="font-size:24px;">
Run in Google Colab:
<img alt="Google Colab" src="https://www.tensorflow.org/images/colab_logo_32px.png" style="vertical-align: bottom;">
</a>

![3 Layered Linear Photonic Analog Neural Network](_static/analogvnn_model.png)

To convert a digital model to its analog counterpart the following steps needs to be followed:

1. Adding the analog layers to the digital model. For example, to create the Photonic Linear Layer with
   {ref}`extra_classes:Reduce Precision`, {ref}`extra_classes:Normalization` and {ref}`extra_classes:noise`:
    1. Create the model similar to how you would create a digital model but using
       {py:class}`analogvnn.nn.module.FullSequential.FullSequential` as superclass
        ```python
        class LinearModel(FullSequential):
            def __init__(self, activation_class, norm_class, precision_class, precision, noise_class, leakage):
                super(LinearModel, self).__init__()

                self.activation_class = activation_class
                self.norm_class = norm_class
                self.precision_class = precision_class
                self.precision = precision
                self.noise_class = noise_class
                self.leakage = leakage

                self.all_layers = []
                self.all_layers.append(nn.Flatten(start_dim=1))
                self.add_layer(Linear(in_features=28 * 28, out_features=256))
                self.add_layer(Linear(in_features=256, out_features=128))
                self.add_layer(Linear(in_features=128, out_features=10))

                self.add_sequence(*self.all_layers)
        ```

       Note: {py:func}`analogvnn.nn.module.Sequential.Sequential.add_sequence` is used to create and set forward and
       backward graphs in AnalogVNN, more information in {doc}`inner_workings`

    2. To add the Reduce Precision, Normalization, and Noise before and after the main Linear layer, `add_layer`
       function is used.
        ```python
        def add_layer(self, layer):
            self.all_layers.append(self.norm_class())
            self.all_layers.append(self.precision_class(precision=self.precision))
            self.all_layers.append(self.noise_class(leakage=self.leakage, precision=self.precision))
            self.all_layers.append(layer)
            self.all_layers.append(self.noise_class(leakage=self.leakage, precision=self.precision))
            self.all_layers.append(self.norm_class())
            self.all_layers.append(self.precision_class(precision=self.precision))
            self.all_layers.append(self.activation_class())
            self.activation_class.initialise_(layer.weight)
        ```
2. Creating an Analog Parameters Model for analog parameters (analog weights and biases)
    ```python
    class WeightModel(FullSequential):
        def __init__(self, norm_class, precision_class, precision, noise_class, leakage):
            super(WeightModel, self).__init__()
            self.all_layers = []

            self.all_layers.append(norm_class())
            self.all_layers.append(precision_class(precision=precision))
            self.all_layers.append(noise_class(leakage=leakage, precision=precision))

            self.eval()
            self.add_sequence(*self.all_layers)
   ```

   Note: Since the `WeightModel` will only be used for converting the data to analog data to be used in the main
   `LinearModel`, we can use `eval()` to make sure the `WeightModel` is never been trained

3. Simply getting data and setting up the model as we will normally do in PyTorch with some minor changes for automatic
   evaluations
    ```python
    torch.backends.cudnn.benchmark = True
    device, is_cuda = is_cpu_cuda.is_using_cuda
    print(f"Device: {device}")
    print()

    # Loading Data
    print(f"Loading Data...")
    train_loader, test_loader, input_shape, classes = load_vision_dataset(
        dataset=torchvision.datasets.MNIST,
        path="_data/",
        batch_size=128,
        is_cuda=is_cuda
    )

    # Creating Models
    print(f"Creating Models...")
    nn_model = LinearModel(
        activation_class=GeLU,
        norm_class=Clamp,
        precision_class=ReducePrecision,
        precision=2 ** 4,
        noise_class=GaussianNoise,
        leakage=0.5
    )
    weight_model = WeightModel(
        norm_class=Clamp,
        precision_class=ReducePrecision,
        precision=2 ** 4,
        noise_class=GaussianNoise,
        leakage=0.5
    )

    # Setting Model Parameters
    nn_model.loss_function = nn.CrossEntropyLoss()
    nn_model.accuracy_function = cross_entropy_accuracy
    nn_model.compile(device=device)
    weight_model.compile(device=device)
    ```
4. Using Analog Parameters Model to convert digital parameters to analog parameters using
   {py:func}`analogvnn.parameter.PseudoParameter.PseudoParameter.parametrize_module`
    ```python
    PseudoParameter.parametrize_module(nn_model, transformation=weight_model)
    ```
5. Adding optimizer
    ```python
    nn_model.optimizer = optim.Adam(params=nn_model.parameters())
    ```
6. Then you are good to go to train and test the model
    ```python
    # Training
    print(f"Starting Training...")
    for epoch in range(10):
        train_loss, train_accuracy = nn_model.train_on(train_loader, epoch=epoch)
        test_loss, test_accuracy = nn_model.test_on(test_loader, epoch=epoch)

        str_epoch = str(epoch + 1).zfill(1)
        print_str = f'({str_epoch})' \
                    f' Training Loss: {train_loss:.4f},' \
                    f' Training Accuracy: {100. * train_accuracy:.0f}%,' \
                    f' Testing Loss: {test_loss:.4f},' \
                    f' Testing Accuracy: {100. * test_accuracy:.0f}%\n'
        print(print_str)
    print("Run Completed Successfully...")
    ```

Full Sample code for this process can be found at {doc}`sample_code`
