if __name__ == '__main__':
    # Some path hacking if you want to run this inside this folder
    import sys
    from pathlib import Path
    pathhack = Path(__file__).resolve().parent.parent.parent / '2023-SP-101-hw1-cjbzfd/'
    sys.path.append(pathhack.as_posix())

import torch
import torch.nn as nn
from mlcvlab.models.nn2 import NN2
from mlcvlab.models.nn1 import NN1
import numpy as np
import mlcvlab.nn.losses as losses


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



def _get_user_weights(*layers):
    weights = []
    for layer in layers:
        w = layer.weight.detach().numpy()
        b = layer.bias.detach().numpy().reshape(-1,1)
        w = np.concatenate((b, w), axis=1)
        z = np.zeros((1,w.shape[-1]))
        w = np.concatenate((z, w), axis=0)
        weights.append(w)
    return weights


def _get_user_grads(*layers):
    grads = []
    for layer in layers:
        w = layer.weight.grad.detach().numpy()
        b = layer.bias.grad.detach().numpy().reshape(-1,1)
        w = np.concatenate((b, w), axis=1)
        z = np.zeros((1,w.shape[-1]))
        w = np.concatenate((z, w), axis=0)
        grads.append(w)
    return grads

# Define a pytorch model
class TorchNN2(nn.Module):
    def __init__(self, in_features=2, hidden_size=4, out_features=2):
        super(TorchNN2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        y_hat = self.layers(x)
        return y_hat


# Define a pytorch model
class TorchNN1(nn.Module):
    def __init__(self, in_features=2, out_features=4):
        super(TorchNN1, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
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

    show_array(user_w1, 'user_w1')
    show_array(user_w2, 'user_w2')

    # Run the input through our model
    user_y_hat = user_nn2.nn2(user_X)

    show_array(user_X, 'user X')
    show_array(user_y_hat, 'user y_hat')

    assert np.allclose(user_y_hat, torch_y_hat.T)


def test_nn2_multiple_input():
    # Define input
    torch_X = 20 * torch.rand(10, 2) + 10

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

    show_array(user_w1, 'user_w1')
    show_array(user_w2, 'user_w2')

    # Run the input through our model
    user_y_hat = user_nn2.nn2(user_X)

    show_array(user_X, 'user X')
    show_array(user_y_hat, 'user y_hat')

    assert np.allclose(user_y_hat, torch_y_hat.T)


def test_nn1_single_input():
    # Define input
    torch_X = 20 * torch.rand(1, 2) + 10

    # Run input through model
    torch_nn1 = TorchNN1()
    torch_y_hat = torch_nn1(torch_X).detach().numpy()

    # Show arrs on error
    show_array(torch_X, 'torch X')
    show_array(torch_y_hat, 'torch y_hat')

    # Define our model
    user_nn1 = NN1()

    # Prepare our inputs, (our convention is transposed compared to PyTorch)
    user_X = torch_X.detach().numpy().T

    # Use the same weights as what the PyTorch model was using and set them for this model
    user_w1 = _get_user_weights(torch_nn1.layers[0])[0]
    user_nn1.layers[0].W = user_w1

    # Run the input through our model
    user_y_hat = user_nn1.nn1(user_X)

    show_array(user_X, 'user X')
    show_array(user_y_hat, 'user y_hat')

    assert np.allclose(user_y_hat, torch_y_hat.T)



def test_nn1_multi_input():
    # Define input
    torch_X = 20 * torch.rand(10, 2) + 10

    # Run input through model
    torch_nn1 = TorchNN1()
    torch_y_hat = torch_nn1(torch_X).detach().numpy()

    # Show arrs on error
    show_array(torch_X, 'torch X')
    show_array(torch_y_hat, 'torch y_hat')

    # Define our model
    user_nn1 = NN1()

    # Prepare our inputs, (our convention is transposed compared to PyTorch)
    user_X = torch_X.detach().numpy().T

    # Use the same weights as what the PyTorch model was using and set them for this model
    user_w1 = _get_user_weights(torch_nn1.layers[0])[0]
    user_nn1.layers[0].W = user_w1

    # Run the input through our model
    user_y_hat = user_nn1.nn1(user_X)

    show_array(user_X, 'user X')
    show_array(user_y_hat, 'user y_hat')

    assert np.allclose(user_y_hat, torch_y_hat.T)



def test_nn2_grad():
    torch_X = 20 * torch.rand(1, 2) + 10

    # Run input through pytorch model
    torch_nn2 = TorchNN2(2, 3, 3)
    torch_yhat = torch_nn2(torch_X)
    torch_y = torch.zeros_like(torch_yhat)

    mse = nn.MSELoss(reduction='sum')
    torch_loss = torch.sqrt(mse(torch_y, torch_yhat))
    torch_loss.backward()

    torch_weights = _get_user_weights(*torch_nn2.layers[::2])
    torch_grads   = _get_user_grads(*torch_nn2.layers[::2])

    mlcv_X = torch_X.detach().numpy().T
    mlcv_y = torch_y.detach().numpy().T 

    # Run input through MLCV model
    # Test loss
    mlcv_nn2 = NN2()

    for idx, torch_weight in enumerate(torch_weights):
        mlcv_nn2.layers[idx].W = torch_weight

    mlcv_yhat = mlcv_nn2.nn2(
        mlcv_X
    )
    mlcv_loss = losses.l2(mlcv_y, mlcv_yhat)
    print(f"MLCV Loss: {mlcv_loss}")

    assert np.allclose(
        torch_yhat.detach().numpy().T,
        mlcv_yhat
    )

    assert np.allclose(
        mlcv_loss,
        torch_loss.detach().numpy().item()
    )
    
    mlcv_grads = mlcv_nn2.grad(mlcv_X, mlcv_y, W=None)


    for idx, (torch_grad, mlcv_grad) in enumerate(zip(torch_grads, mlcv_grads)):
        show_array(torch_grad, f'Torch dW{idx}')
        show_array(mlcv_grad, f'MLCV dW{idx}')
        delta = mlcv_grad - torch_grad
        show_array(delta, f'delta {idx}')
        assert np.allclose(torch_grad, mlcv_grad, atol=1e-6)

    


def test_nn1_grad():
    torch_X = 20 * torch.rand(1, 2) + 10

    # Run input through pytorch model
    torch_nn2 = TorchNN1()
    torch_yhat = torch_nn2(torch_X)
    torch_y = torch.zeros_like(torch_yhat)

    mse = nn.MSELoss(reduction='sum')
    torch_loss = torch.sqrt(mse(torch_y, torch_yhat))
    torch_loss.backward()

    torch_weights = _get_user_weights(*torch_nn2.layers[::2])
    torch_grads   = _get_user_grads(*torch_nn2.layers[::2])

    mlcv_X = torch_X.detach().numpy().T
    mlcv_y = torch_y.detach().numpy().T 

    # Run input through MLCV model
    # Test loss
    mlcv_nn1 = NN1()

    for idx, torch_weight in enumerate(torch_weights):
        mlcv_nn1.layers[idx].W = torch_weight

    mlcv_yhat = mlcv_nn1.nn1(
        mlcv_X
    )
    mlcv_loss = losses.l2(mlcv_y, mlcv_yhat)
    print(f"MLCV Loss: {mlcv_loss}")

    assert np.allclose(
        torch_yhat.detach().numpy().T,
        mlcv_yhat
    )

    assert np.allclose(
        mlcv_loss,
        torch_loss.detach().numpy().item()
    )
    
    mlcv_grads = mlcv_nn1.grad(mlcv_X, mlcv_y, W=None)


    for idx, (torch_grad, mlcv_grad) in enumerate(zip(torch_grads, mlcv_grads)):
        show_array(torch_grad, f'Torch dW{idx}')
        show_array(mlcv_grad, f'MLCV dW{idx}')
        delta = mlcv_grad - torch_grad
        show_array(delta, f'delta {idx}')
        assert np.allclose(torch_grad, mlcv_grad, atol=1e-6)

# def test_nn1_grad():
#     # Define input
#     torch_X = 20 * torch.rand(1, 4) + 10

#     # Run input through model
#     torch_nn1 = TorchNN1()
#     torch_y_hat = torch_nn1(torch_X)
#     torch_y = torch.zeros_like(torch_y_hat)

#     loss = nn.MSELoss()(torch_y, torch_y_hat)
#     loss.backward()

#     show_array(torch_nn1.layers[0].weight.grad, 'Torch W1 grad')
#     show_array(torch_nn1.layers[0].bias.grad, 'Torch B1 grad')

#     # Show arrs on error
#     # show_array(torch_X, 'torch X')
#     # show_array(torch_y_hat, 'torch y_hat')

#     # # Define our model
#     # user_nn1 = NN1()

#     # # Prepare our inputs, (our convention is transposed compared to PyTorch)
#     # user_X = torch_X.detach().numpy().T

#     # # Use the same weights as what the PyTorch model was using and set them for this model
#     # user_w1 = _get_user_weights(torch_nn1.layers[0])
#     # user_nn1.layers[0].W = user_w1

#     # # Run the input through our model
#     # user_y_hat = user_nn1.nn1(user_X)

#     # show_array(user_X, 'user X')
#     # show_array(user_y_hat, 'user y_hat')

#     assert False
#     assert np.allclose(user_y_hat, torch_y_hat.T)

if __name__ == '__main__':
    test_nn2_single_input()
    test_nn2_multiple_input()
    test_nn2_grad()
    test_nn1_single_input()