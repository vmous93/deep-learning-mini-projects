#!/usr/bin/env python
# coding: utf-8


import torch



class Module(object):
    def __init__(self):
        super().__init__()
        
    def forward(self, *input) :
        raise NotImplementedError
        
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
        
    def param(self):
        return []
    
    def update(self):
        pass
    
    def zero_grad(self):
        pass





