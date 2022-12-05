#!/usr/bin/env python
# coding: utf-8


import torch
from torch import nn
from torch.nn import Linear, Conv2d
from torch.nn import functional as F
from torch.nn import Dropout, MaxPool2d, MaxPool1d
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d
from torch.nn.modules.activation import ReLU, Sigmoid, Softmax
import dlc_practical_prologue as prologue
import warnings
warnings.filterwarnings('ignore')



train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)


# MLPNet with Weight sharing


class Pre_WS_MLPNet(nn.Module):

    def __init__(self):
        super(Pre_WS_MLPNet, self).__init__()
        
        self.fullyConnected = nn.Sequential(
          Linear(196, 200),
          ReLU(True),
          BatchNorm1d(200),
          Linear(200, 84),
          ReLU(True),
          BatchNorm1d(84),
          Linear(84, 10),
          Sigmoid())
        
    def forward(self, x1, x2):
        x1 = x1.view(-1, 196)
        x1 = self.fullyConnected(x1)
        x2 = x2.view(-1, 196)
        x2 = self.fullyConnected(x2)
        
        
        return torch.cat((x1, x2), 1)
    

class WS_MLPNet(nn.Module):
    def __init__(self):
        super(WS_MLPNet, self).__init__()
        
        self.shareWeight = Pre_WS_MLPNet()
        self.fullyConnected = nn.Sequential(
            nn.Linear(20, 200),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(200),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(200),
            nn.Linear(200, 1),
            Sigmoid()
        )

    def forward(self, x):
        x = self.shareWeight(x[:, 0], x[:,1])
        x = self.fullyConnected(x)
        
        return x


# LeNet5 with Weight sharing


class WS_LeNet5(nn.Module):

    def __init__(self):
        super(WS_LeNet5, self).__init__()
        
        self.convolutional_layer = nn.Sequential(            
            nn.Conv2d(1, 16, 3),
            ReLU(True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3),
            ReLU(True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 120, 2),
            ReLU(True),
            nn.BatchNorm2d(120)
            )

        self.linear_layer = nn.Sequential(
            nn.Linear(240, 84),
            ReLU(True),
            nn.BatchNorm1d(84),
            nn.Linear(84, 1),
            Sigmoid())
     
    def forward(self,x):
        x1 , x2 = x[:,0], x[:,1]
        x1 = self.convolutional_layer(x1.unsqueeze(1))
        x2 = self.convolutional_layer(x2.unsqueeze(1))
        x1 = x1.view(-1, 120)
        x2 = x2.view(-1, 120)
        X = torch.concat((x1, x2), 1)
        X = self.linear_layer(X)
        
        return X
    


# Training


def train_model_WS(model, train_input, train_target, lr,mini_batch_size, nb_epochs = 25):
    criterion = nn.MSELoss()
    tr_errs = []
    for e in range(nb_epochs):
        acc_loss = 0
        nb_errors = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output.view(-1, mini_batch_size), train_target.narrow(0, b, mini_batch_size).float())
            acc_loss = acc_loss + loss.item()           
            
            optimizer = torch.optim.Adam(model.parameters(), lr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                    
            output = torch.as_tensor((output - 0.5) > 0, dtype=torch.int32)
            for k in range(mini_batch_size):
                if output[k] != train_target.narrow(0, b, mini_batch_size)[k]:
                    nb_errors = nb_errors + 1

        error_rate = nb_errors/train_input.size(0)
        
        tr_errs.append(error_rate*100)
    return error_rate*100, tr_errs
    

# Testing


def compute_nb_errors(model, test_input, target, mini_batch_size):
    nb_errors = 0
    for b in range(0, test_input.size(0), mini_batch_size):
        output = model(test_input.narrow(0, b, mini_batch_size))
        output = torch.as_tensor((output - 0.5) > 0, dtype=torch.int32)
        for k in range(mini_batch_size):
            if output[k] != test_target.narrow(0, b, mini_batch_size)[k]:
                nb_errors = nb_errors + 1
    error = nb_errors/test_input.size(0)
    return error*100


# Run


def ws_run(mini_batch_size, learning_rate, mod):
    tr_er_list = []
    tes_er_list = []
    for k in range(10):
        if mod == 0: model = WS_MLPNet()
        else: model = WS_LeNet5()
        train_errors, tr_plt = train_model_WS(model, train_input, train_target, learning_rate, mini_batch_size)
        tr_er_list.append(round(train_errors,2))
        test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)
        tes_er_list.append(round(test_errors,2))
        del model
    
    tr_acc =  [100 - element for element in tr_er_list]
    tes_acc = [100 - element for element in tes_er_list] 
    return tr_er_list, tes_er_list, tr_acc, tes_acc, tr_plt
