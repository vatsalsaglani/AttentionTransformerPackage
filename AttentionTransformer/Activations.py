import torch 
import torch.nn as nn
import torch.nn.functional as F


activation_dict = {
    # rectified linear unit
    'relu': torch.relu,
    # randomized rectified linear unit
    'rrelu': torch.rrelu,
    # relu 6 pad = 6
    'relu6': nn.ReLU6(),
    # Exponential linear unit
    'elu': nn.ELU(),
    # continuously differentiable exponential linear units
    'celu': nn.CELU(),
    # self-normalizing exponential linear units
    'selu': nn.SELU(),
    # gaussian error linear units
    'gelu': F.gelu,
    # parametric rectified linear units
    'prelu': nn.PReLU()

}