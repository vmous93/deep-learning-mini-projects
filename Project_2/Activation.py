#!/usr/bin/env python
# coding: utf-8


import torch
from Module import *


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.value = 0
        self.name = 'Sigmoid'

    def forward(self, input):
        # Sigmoid(x) = 1 / 1 + exp(-x)
        self.value = input
        self.value = 1 / (1 + torch.exp(-self.value))
        return self.value

    def backward(self, gradwrtoutput):
        # sigmoid(x) * 1 - sigmoid(x) * gradwrtoutput
        return self.forward(self.value) * (1 - self.forward(self.value)) * gradwrtoutput


class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.value = 0
        self.name = 'Tanh'

    def forward(self, input):
        # Tanh(x) = exp(x) - exp(-x) / exp(x) + exp(-x)
        self.value = input
        self.value = (torch.exp(self.value) - torch.exp(-self.value)) / (torch.exp(self.value) + torch.exp(-self.value))
        return self.value

    def backward(self, gradwrtoutput):
        # 1 - Tanh(x)**2 * gradwrtoutput
        return (1 - (self.forward(self.value) ** 2)) * gradwrtoutput 





