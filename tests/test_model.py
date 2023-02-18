import torch
import torch.nn as nn
from mlcvlab.models.nn2 import NN2
import numpy as np
import mlcvlab.nn.activations as act


def show_array(arr, name):
    sep = '-' * 40
    print(sep)
    print(f"{name:^40}")
    print(sep)
    print()
    print(arr)
    print()
    print(f"Shape: {arr.shape}")
    print()
    print()

def _pad_input(x):
    ones = np.ones(shape=(1,x.shape[1]), dtype=x.dtype)
    return np.concatenate((ones, x), axis=0)

def _get_user_weights(*layers):
    weights = []
    for layer in layers:
        w = layer.weight.detach().numpy()
        b = layer.bias.detach().numpy().reshape(-1,1)
        w = np.concatenate((b, w), axis=1)
        weights.append(w)
    return weights


# Define a pytorch model
class TorchNN2(nn.Module):
    def __init__(self):
        super(TorchNN2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        y_hat = self.layers(x)
        return y_hat


def test_nn2_single_input():
    # Define input
    torch_X = 20 * torch.rand(1, 2) + 10

    # Run input through model
    torch_nn2 = TorchNN2()
    torch_y_hat = torch_nn2(torch_X).detach().numpy()

    # Show arrs on error
    show_array(torch_X, 'torch X')
    show_array(torch_y_hat, 'torch y_hat')

    # Define our model
    user_nn2 = NN2()

    # Prepare our inputs, (our convention is transposed compared to PyTorch)
    user_X = torch_X.detach().numpy().T

    # Use the same weights as what the PyTorch model was using and set them for this model
    user_w1, user_w2 = _get_user_weights(torch_nn2.layers[0], torch_nn2.layers[2])
    user_nn2.layers[0].W = user_w1
    user_nn2.layers[1].W = user_w2

    # Run the input through our model
    user_y_hat = user_nn2.nn2(user_X)

    show_array(user_X, 'user X')
    show_array(user_y_hat, 'user y_hat')

    assert np.allclose(user_y_hat, torch_y_hat.T)


def test_nn2_multiple_input():
    # Define input
    torch_X = 20 * torch.rand(4, 2) + 10

    # Run input through model
    torch_nn2 = TorchNN2()
    torch_y_hat = torch_nn2(torch_X).detach().numpy()

    # Show arrs on error
    show_array(torch_X, 'torch X')
    show_array(torch_y_hat, 'torch y_hat')

    # Define our model
    user_nn2 = NN2()

    # Prepare our inputs, (our convention is transposed compared to PyTorch)
    user_X = torch_X.detach().numpy().T

    # Use the same weights as what the PyTorch model was using and set them for this model
    user_w1, user_w2 = _get_user_weights(torch_nn2.layers[0], torch_nn2.layers[2])
    user_nn2.layers[0].W = user_w1
    user_nn2.layers[1].W = user_w2

    # Run the input through our model
    user_y_hat = user_nn2.nn2(user_X)

    show_array(user_X, 'user X')
    show_array(user_y_hat, 'user y_hat')

    assert np.allclose(user_y_hat, torch_y_hat.T)






