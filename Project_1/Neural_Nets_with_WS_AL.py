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


# WS + AL MLPNet


# A MLP Network for classification
class Pre_WSAL_MLP(nn.Module):
    def __init__(self):
        super(Pre_WSAL_MLP, self).__init__()
        self.fullyConnected = nn.Sequential(
          Linear(196, 200),
          ReLU(True),
          BatchNorm1d(200),
          Linear(200, 84),
          ReLU(True),
          BatchNorm1d(84),
          Linear(84, 10),
          Softmax())
        
    def forward(self, x):
        x = x.view(-1, 196)
        x = self.fullyConnected(x)
        
        return x



# A MLP Network for comparing the result of classification
class WS_AL_MLPNet(nn.Module):
    def __init__(self):
        super(WS_AL_MLPNet, self).__init__()
        self.weightSharing = Pre_WSAL_MLP()

        self.fullyConnected = nn.Sequential(
          Linear(20, 200),
          ReLU(True),
          BatchNorm1d(200),
          nn.Dropout(p=0.5),
          Linear(200, 84),
          ReLU(True),
          BatchNorm1d(84),
          nn.Dropout(p=0.5),
          Linear(84, 1),
          Sigmoid())
        
    def forward(self, x):
        # Return the results of each classification for computing the auxiliary losses
        x1 = self.weightSharing(x[:, 0])
        x2 = self.weightSharing(x[:, 1])
        X = torch.cat((x1, x2), 1)
        res = self.fullyConnected(X)
        return x1,x2, res

# WS + AL LeNet5



# LeNet for classification
class Pre_WSAL_LeNet5(nn.Module):

    def __init__(self):
        super(Pre_WSAL_LeNet5, self).__init__()
        
        self.convolutional_layer = nn.Sequential(            
            nn.Conv2d(1, 16, 3),
            ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3),
            ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 120, 2),
            ReLU(True),
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(120, 84),
            ReLU(True),
            nn.BatchNorm1d(84),
            nn.Linear(84, 10),
            Softmax())
     
    def forward(self, x):
        x = self.convolutional_layer(x)
        x = x.view(-1, 120)
        x = self.linear_layer(x)
        
        return x


 # LeNet for comparing the result of classification
class WSAL_LeNet5(nn.Module):
    def __init__(self):
        super(WSAL_LeNet5, self).__init__()
        self.weightSharing = Pre_WSAL_LeNet5()
        self.fullyConnected = nn.Sequential(
          Linear(20, 200),
          ReLU(True),
          nn.Dropout(p=0.7),
          Linear(200, 84),
          ReLU(True),
          nn.Dropout(p=0.7),
          Linear(84, 1),
          Sigmoid())
        
    def forward(self, x):
        # Return the results of each classification for computing the auxiliary losses
        x1 = self.weightSharing(x[:, 0].unsqueeze(1))
        x2 = self.weightSharing(x[:, 1].unsqueeze(1))
        X = torch.cat((x1, x2), 1)
        res = self.fullyConnected(X)
        
        return x1,x2, res
    


# Train


def train_model_WSAL(model, train_input, train_target, train_classes, lr, mini_batch_size, nb_epochs = 25):
    cri1 = nn.CrossEntropyLoss()
    cri3 = nn.MSELoss()
    tr_errs = []
    for e in range(nb_epochs):
        acc_loss = 0
        nb_errors = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            x1,x2, res = model(train_input.narrow(0, b, mini_batch_size))
            loss_x1 = cri1(x1, train_classes[:,0].narrow(0, b, mini_batch_size))
            loss_x2 = cri1(x2, train_classes[:,1].narrow(0, b, mini_batch_size))
            loss_res = cri3(res.view(-1, mini_batch_size), train_target.narrow(0, b, mini_batch_size).float())
            # Computing the auxiliary loss
            loss = 0.05*(loss_x1 + loss_x2) + 0.3*loss_res
            acc_loss = acc_loss + loss.item()
            
            optimizer = torch.optim.Adam(model.parameters(), lr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #Converting the predicted labels to zero and one        
            res = torch.as_tensor((res - 0.5) > 0, dtype=torch.int32)
            for k in range(mini_batch_size):
                if res[k] != train_target.narrow(0, b, mini_batch_size)[k]:
                    nb_errors = nb_errors + 1

        error_rate =nb_errors/train_input.size(0)
   
        tr_errs.append(error_rate*100)
    return error_rate*100, tr_errs
    



def compute_nb_errors(model, test_input, test_target, mini_batch_size):
    nb_errors = 0
    for b in range(0, test_input.size(0), mini_batch_size):
        _,_, output = model(test_input.narrow(0, b, mini_batch_size))
        output = torch.as_tensor((output - 0.5) > 0, dtype=torch.int32)
        for k in range(mini_batch_size):
            if output[k] != test_target.narrow(0, b, mini_batch_size)[k]:
                nb_errors = nb_errors + 1
    error = nb_errors/test_input.size(0)
    return error*100



def WSAL_run(mini_batch_size, learning_rate, mod):
    tr_er_list = []
    tes_er_list = []
    for k in range(10):
        if mod == 0: model = WS_AL_MLPNet()
        else: model = WSAL_LeNet5()
        train_error, tr_plt = train_model_WSAL(model, train_input, train_target,train_classes , learning_rate, mini_batch_size)
        tr_er_list.append(round(train_error,2))
        test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)
        tes_er_list.append(round(test_errors,2))
        del model
    
    tr_acc =  [100 - element for element in tr_er_list]
    tes_acc = [100 - element for element in tes_er_list] 
    return tr_er_list, tes_er_list, tr_acc, tes_acc, tr_plt

