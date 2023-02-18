import torch
import numpy as np
import torch.nn as nn
from torch import tensor
from mlcvlab.nn import losses
from typing import Union


def test_l2():
    """Test the output and gradient of the mlcvlab.nn.activations.relu function.

    size=1 b/c scalar output is expected for hw1.
    """
    def sqrt_MSE(Y_hat: tensor, Y: tensor) -> tensor:
        loss = nn.MSELoss(reduction='sum')
        return torch.sqrt(loss(Y_hat, Y))

    _run_loss_test(
        torch_func=sqrt_MSE,
        mlcv_z_func=losses.l2,
        mlcv_grad_func=losses.l2_grad,
        size=3
    )


def _run_loss_test(
    torch_func: nn.Module, 
    mlcv_z_func: callable, 
    mlcv_grad_func: callable,
    size: Union[int, tuple[int, ...]]
) -> None:
    """Computes and compares the results of an activation function and the associated gradient

    Args:
        torch_func (nn.Module): PyTorch activation function module
        mlcv_z_func (callable): Activation function
        mlcv_grad_func (callable): Activation gradient function
        size (Union[int, tuple[int, ...]]): Size of random vector to test,
            if int is passed shape is changed to (shape, 1)
    """
    if isinstance(size, int):
        size = (size, 1)

    Y_hat = np.random.uniform(-10, 10, size=size)  # Input
    Y = np.random.uniform(-10, 10, size=size)  # Target

    torch_z, torch_grad = _z_grad_ground_truth(Y, Y_hat, torch_func)
    user_z = mlcv_z_func(Y, Y_hat)
    user_grad = mlcv_grad_func(Y, Y_hat)

    print(f"{user_z = }")
    print(f"{torch_z = }")
    assert np.allclose(user_z, torch_z)

    print(f"{user_grad = }")
    print(f"{torch_grad = }")
    assert np.allclose(user_grad, torch_grad)


def _z_grad_ground_truth(Y: np.array, Y_hat: np.array, func: nn.Module) -> tuple[np.array, np.array]:
    """Returns a ground truth value for the gradient of Y_hat when used as input for function


    Args:
        Y (np.array): Target values for loss function
        Y_hat (np.array): Input to function to calculate gradient of
        func (nn.Module): Function to calculate gradient with, must support the pytorch ".backward()" convention

    Returns:
        _type_: _description_
    """

    Y_hat = torch.tensor(Y_hat, requires_grad=True)
    Y = torch.tensor(Y)
    z = func(Y_hat, Y)
    z.sum().backward()

    grad = Y_hat.grad.detach().numpy()
    z = z.detach().numpy().item()

    return z, grad


if __name__ == '__main__':
    test_l2()