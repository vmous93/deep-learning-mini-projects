#!/usr/bin/env python
# coding: utf-8

import torch
import matplotlib.pyplot as plt

#Generate data set
def generate_disc_set(nb):
    
    center = torch.tensor([0.5,0.5])
    radius = 1/torch.sqrt(torch.tensor(2*torch.pi))
    input =  torch.empty(nb, 2).uniform_(0, 1)
    
    target = (((((input - center)**2).sum(1)) - radius**2) < 0).long()
    mean, std = input.mean(), input.std()
    input = input.sub_(mean).div_(std)
    
    return input, target


#plot error
def plot_err(train , test, nb_epochs, title):
    y = [i for i in range(nb_epochs)]
    plt.plot(y, train, label='Train')
    plt.plot(y, test, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()


#Plot the result
def plot_figure(data, label, title):
    fig, ax = plt.subplots(figsize=(5,5))
    scatter = ax.scatter(data[:, 0], data[:, 1],c=label, cmap='bwr')
    legend = ax.legend(*scatter.legend_elements())    
    plt.title(title)
    plt.show()


#compute accuracy of model
def compute_accuracy(model, data_features, data_target):
    predicted_classes = prediction(model, data_features)
    nb_data_errors = sum(data_target.data != predicted_classes.flatten())
    return nb_data_errors/data_features.size(0)*100

#forward input data through the network and predict the labels
def prediction(model, data):
    output = model.forward(data)
    if model.model[-1].name == 'Sigmoid':
        output = (output>0.5).long()
    elif model.model[-1].name == 'Tanh':
        output = (output>0).long()
    return output
#Train   
def train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, optimizer, opSGD, opAdam, criterion):   
    
    train_err = []
    test_err = []
    test_results = []
    for epochs in range(0, nb_epochs):
        train_err.append(compute_accuracy(Model, train_input,train_target).item())
        test_err.append(compute_accuracy(Model, test_input,test_target).item())
        loss_sum = 0
        test_results.append(prediction(Model, test_input))
        for b in range(train_input.size(0) // batch_size):
            output = Model.forward(train_input.narrow(0, b * batch_size, batch_size))
            y = train_target.narrow(0, b * batch_size, batch_size)
            mse_loss = criterion.forward(y, output.flatten())
            mse_grad = criterion.backward(y, output.flatten())
            loss_sum = loss_sum + mse_loss.item()
            Model.zero_grad()
            Model.backward(mse_grad)     

            if opSGD == True:
                optimizer.update()
            elif opAdam == True:
                optimizer.update()
            else: Model.update()
            Model.zero_grad()

    print("Train accuracy: ", round(100 - compute_accuracy(Model, train_input, train_target).item(), 2),"\nTest accuracy: ", round(100 - compute_accuracy(Model, test_input, test_target).item(), 2))    
    
    return test_results[-1], train_err, test_err





