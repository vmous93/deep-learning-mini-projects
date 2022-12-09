#!/usr/bin/env python
# coding: utf-8

import torch

class SGD():
    def __init__(self, models, lr):
        
        self.lr = lr 
        self.momentum = 0.9
        self.models = models
        
    def update(self):
        for module in self.models.model:
            if module.name == 'Linear':
                # If not initialized
                if module.w_updt is None:
                    module.w_updt = torch.zeros((module.weight).size())
                    module.b_updt = torch.zeros((module.bias).size())

                # Use momentum 
                module.w_updt = self.momentum * module.w_updt + (1 - self.momentum) * module.grad_w
                module.b_updt = self.momentum * module.b_updt + (1 - self.momentum) * module.grad_b
                # Move against the gradient to minimize loss
                
                w_update = module.weight - self.lr * module.w_updt
                b_update = module.bias - self.lr * module.b_updt
                #Update weight and bias of the module
                module.update(w_update, b_update, opSGD=True)
                

class AdamOptim():
    def __init__(self,models, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr = lr
        self.models = models
        self.t = 0
    
    def update(self,w=None, b=None, gw=None, db=None):
        self.t = 0
        for module in self.models.model:
            if module.name == 'Linear':
                converged = False
                while not converged:
                    w = module.weight
                    b = module.bias
                    w_0_old = module.weight
                    gw = module.grad_w
                    gb = module.grad_b
                    # If not initialized
                    if module.m_dw == None:
                        module.m_dw, module.v_dw = torch.zeros((module.weight).shape), torch.zeros((module.weight).shape)
                        module.m_db, module.v_db = torch.zeros((module.bias).shape), torch.zeros((module.bias).shape)
                    
                    ## momentum beta 1
                    #weights
                    module.m_dw = self.beta1*module.m_dw + (1-self.beta1)*gw
                    #biases
                    module.m_db = self.beta1*module.m_db + (1-self.beta1)*gb

                    # beta 2
                    # weights 
                    module.v_dw = self.beta2*module.v_dw + (1-self.beta2)*(gw**2)
                    # biases
                    module.v_db = self.beta2*module.v_db + (1-self.beta2)*(db**2)

                    # bias correction
                    m_dw_corr = module.m_dw/(1-self.beta1**self.t)
                    m_db_corr = module.m_db/(1-self.beta1**self.t)
                    v_dw_corr = module.v_dw/(1-self.beta2**self.t)
                    v_db_corr = module.v_db/(1-self.beta2**self.t)

                    # update weights and biases
                    w = w - self.lr* m_dw_corr/(torch.sqrt(v_dw_corr)+self.epsilon)
                    b = b - self.lr* m_db_corr/(torch.sqrt(v_db_corr)+self.epsilon)
                    #if the weight doesn't have changed
                    if torch.all(w.eq(w_0_old)) == True:
                        break
                    
                    self.t+=1
                    #if t > batch size
                    if self.t > 5:
                        break
                    
                #Update weight and bias of the module
                module.update(w, b, opAdam=True)
                



