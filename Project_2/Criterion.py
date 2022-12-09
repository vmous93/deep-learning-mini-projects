#!/usr/bin/env python
# coding: utf-8

import torch
from Module import *


class MSE(Module):
    def init(self):
        super().init()
        self.name = "MSE"
    
    def forward(self, y, y_pred):
        # MSE(y_pred, y) = 1/batchSize * sum(y_pred - y)^2
        loss = sum(((y_pred - y) ** 2)) / y.size()[0]
        return loss 
    
    def backward(self, y, y_pred):
        # sum(2 * (y_pred - y)) / batchSize
        return sum(2*(y_pred-y))/y.size()[0]


class BCE(Module):
    def __init__(self):
        super().__init__()
        self.name = "BCE"

    def forward(self, y, y_pred):
        # BCE(y, y_pred) = - sum(y_pred * log(y) + (1 - y_pred) * log(1 - y))
        return -sum(y_pred * torch.clamp(torch.log(y), min= -1) + (1 - y_pred) * torch.clamp(torch.log(1 - y), min=-1))

    def backward(self, y, y_pred):
        # sum(y_pred - y) / batchSize
        return sum(y_pred - y)/y.size()[0]





