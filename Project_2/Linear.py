#!/usr/bin/env python
# coding: utf-8


import torch
from Module import *


class Linear(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.name = 'Linear'
        #Initialize the parameters
        self.x = torch.zeros(output_size)        
        self.input = input_size
        self.output = output_size
        self.weight = torch.rand(size=(self.input, self.output))
        self.bias = torch.rand(self.output)
        self.grad_w = torch.empty(self.weight.size()) 
        self.grad_b = torch.empty(self.bias.size())
        
        #These parameters are needed for SGD
        self.w_updt = None
        self.b_updt = None
        
        #These parameters are needed for Adam
        self.m_dw, self.v_dw = None, None
        self.m_db, self.v_db = None, None


    def forward(self, x):
        self.x = x
        #Y = W * X + B
        return x.mm(self.weight) + self.bias
        
    def backward(self, gradwrtoutput):
        #gradient of linear
        b = gradwrtoutput.mm(self.weight.t())
        #update the gradients
        self.grad_w += self.x.t().mm(gradwrtoutput) 
        self.grad_b += gradwrtoutput.sum(dim=0)
        return b
    
        
    def update(self, w=None, b=None, opSGD = False, opAdam = False):
        lr = 0.001
        #if SGD is used
        if opSGD == True:
            self.weight= w
            self.bias= b
        #if Adam is used
        elif opAdam == True:
            self.weight= w
            self.bias= b
        #if no optimization is used
        else:
            self.weight = self.weight - (lr)*(self.grad_w) 
            self.bias = self.bias - (lr)*self.grad_b
        
    
    def param(self):
        #update and save the parameters
        return [(self.weight, self.grad_w), (self.bias, self.grad_b)]

    def zero_grad(self):
        # set gradients to zero
        self.grad_w.zero_() 
        self.grad_b.zero_()




