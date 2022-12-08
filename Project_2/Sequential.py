#!/usr/bin/env python
# coding: utf-8


import torch
from Module import *


class Sequential(Module):
    def __init__(self, models):
        super().__init__()
        self.name = 'Sequential'
        self.model = models
        self.params = []

    def forward(self, x):
        # call forward function of the current module
        for module in self.model:
            x = module.forward(x)
        return x
    
    def backward(self, grad_pred):
        # Strat from the last module
        for module in reversed(self.model):
            grad_pred = module.backward(grad_pred)
        return grad_pred
    
    def update(self):
         # update parameters of the module if needed
        for module in self.model:
            module.update()
    def param(self):
        # collect the parameters
        for module in self.modules:
            self.params.append(model.param())
        return self.params
    
    def zero_grad(self):
        # call zero_grad of the module
        for module in self.model:
            module.zero_grad()

