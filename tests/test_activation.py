import torch
import numpy as np
import torch.nn as nn
from torch import tensor
from mlcvlab.nn import activations
from typing import Union



def test_relu():
    """Test the output and gradient of the mlcvlab.nn.activations.relu function"""
    _run_activation_test(
        torch_func=nn.ReLU,
        mlcv_z_func=activations.relu,
        mlcv_grad_func=activations.relu_grad,
        size=10000
    )

def test_sigmoid():
    """Test the output and gradient of the mlcvlab.nn.activations.sigmoid function"""
    _run_activation_test(
        torch_func=nn.Sigmoid,
        mlcv_z_func=activations.sigmoid,
        mlcv_grad_func=activations.sigmoid_grad,
        size=10000
    )


def test_softmax():
    pass


def test_tanh():
    """Test the output and gradient of the mlcvlab.nn.activations.tanh function"""
    _run_activation_test(
        torch_func=nn.Tanh,
        mlcv_z_func=activations.tanh,
        mlcv_grad_func=activations.tanh_grad,
        size=10000
    )


def _run_activation_test(
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
    X = np.random.uniform(-10, 10, size=size)
    torch_z, torch_grad = _z_grad_ground_truth(X, torch_func())
    user_z = mlcv_z_func(X)
    user_grad = mlcv_grad_func(user_z)
    assert np.allclose(user_z, torch_z)
    assert np.allclose(user_grad, torch_grad)


def _z_grad_ground_truth(X: np.array, func: nn.Module) -> tuple[np.array, np.array]:
    """Returns a ground truth value for the gradient of X when used as input for function


    Args:
        X (np.array): Input to function to calculate gradient of
        func (nn.Module): Function to calculate gradient with, must support the pytorch ".backward()" convention

    Returns:
        _type_: _description_
    """

    X = torch.tensor(X, requires_grad=True)
    z = func(X)
    z.sum().backward()

    grad = X.grad.detach().numpy()
    z = z.detach().numpy()

    return z, grad
