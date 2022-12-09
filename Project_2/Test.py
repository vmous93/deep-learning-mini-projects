#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from Train import *
from Linear import *
from Sequential import *
from Activation import *
from Optimizer import *
from Criterion import *
from Module import *
torch.set_grad_enabled(False)


# In[2]:


train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)
batch_size = 5


# In[3]:


plot_figure(test_input, test_target, "Data")


# # Activation: ReLU, Criterion: MSE, Optimizer: SGD

# In[4]:


nb_epochs = 300
Model = Sequential([Linear(2,25), ReLU(), Linear(25,25), ReLU(), Linear(25,25), ReLU(), Linear(25,1), Sigmoid()])
sgd = SGD(Model, 0.001)
print("*** Activation: ReLU ***", "\n*** Criterion: MSE ***", "\n*** Optimizer: SGD ***")
res, train_err, test_err = train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, sgd, True, False, MSE()) 
plot_figure(test_input, res, "Testing with (ReLU, MSE, SGD) ")
#plot_err(train_err, test_err, nb_epochs, "Network: (ReLU, MSE, SGD)")
del Model
del sgd


# # Activation: ReLU, Criterion: MSE, Optimizer: Adam

# In[5]:


nb_epochs = 150
batch_size = 5
Model = Sequential([Linear(2,25), ReLU(), Linear(25,25), ReLU(), Linear(25,25), ReLU(), Linear(25,1), Sigmoid()])
adam = AdamOptim(Model, 0.01)
print("*** Activation: ReLU ***", "\n*** Criterion: MSE ***", "\n*** Optimizer: Adam ***")
res, train_err, test_err = train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, adam, False, True, MSE())   
plot_figure(test_input, res, "Testing with (ReLU, MSE, Adam) ")
#plot_err(train_err, test_err, nb_epochs, "Network: (ReLU, MSE, Adam)")

del Model
del adam


# # Activation: ReLU, Criterion: MSE, Optimizer: None

# In[6]:


nb_epochs = 300
Model = Sequential([Linear(2,25), ReLU(), Linear(25,25), ReLU(), Linear(25,25), ReLU(), Linear(25,1), Sigmoid()])
print("*** Activation: ReLU ***", "\n*** Criterion: MSE ***", "\n*** Optimizer: None ***")
res, train_err, test_err = train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, None, False, False, MSE())
plot_figure(test_input, res, "Testing with (ReLU, MSE, Without optimizer) ")
#plot_err(train_err, test_err, nb_epochs, "Network: (ReLU, MSE, Without optimizer)")

del Model


# # Activation: ReLU, Criterion: BCE, Optimizer: SGD

# In[9]:


nb_epochs = 300
Model = Sequential([Linear(2,25), ReLU(), Linear(25,25), ReLU(), Linear(25,25), ReLU(), Linear(25,1), Sigmoid()])
sgd = SGD(Model, 0.001)
print("*** Activation: ReLU ***", "\n*** Criterion: BCE ***", "\n*** Optimizer: SGD ***")
res, train_err, test_err = train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, sgd, True, False, BCE())   
plot_figure(test_input, res, "Testing with (ReLU, BCE, SGD) ")
#plot_err(train_err, test_err, nb_epochs, "Network: (ReLU, BCE, SGD)")

del Model
del sgd


# # Activation: ReLU, Criterion: BCE, Optimizer: Adam

# In[10]:


nb_epochs = 150
Model = Sequential([Linear(2,25), ReLU(), Linear(25,25), ReLU(), Linear(25,25), ReLU(), Linear(25,1), Sigmoid()])
adam = AdamOptim(Model, 0.01)
print("*** Activation: ReLU ***", "\n*** Criterion: BCE ***", "\n*** Optimizer: Adam ***")
res, train_err, test_err = train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, adam, False, True, BCE())   
plot_figure(test_input, res, "Testing with (ReLU, BCE, Adam) ")
#plot_err(train_err, test_err, nb_epochs, "Network: (ReLU, BCE, Adam)")
del Model
del adam


# # Activation: ReLU, Criterion: BCE, Optimizer: None

# In[11]:


nb_epochs = 300
Model = Sequential([Linear(2,25), ReLU(), Linear(25,25), ReLU(), Linear(25,25), ReLU(), Linear(25,1), Sigmoid()])
print("*** Activation: ReLU ***", "\n*** Criterion: BCE ***", "\n*** Optimizer: None ***")
res, train_err, test_err = train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, None, False, False, BCE())   
plot_figure(test_input, res, "Testing with (ReLU, BCE, Without optimizer) ")
#plot_err(train_err, test_err, nb_epochs, "Network: (ReLU, MSE, Without optimizer)")
del Model


# # Activation: Tanh, Criterion: MSE, Optimizer: SGD

# In[12]:


nb_epochs = 300
Model = Sequential([Linear(2,25), Tanh(), Linear(25,25), Tanh(), Linear(25,25), Tanh(), Linear(25,1), Sigmoid()])
sgd = SGD(Model, 0.001)
print("*** Activation: Tanh ***", "\n*** Criterion: MSE ***", "\n*** Optimizer: SGD ***")
res, train_err, test_err = train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, sgd, True, False, MSE())   
plot_figure(test_input, res, "Testing with (Tanh, MSE, SGD) ")
#plot_err(train_err, test_err, nb_epochs, "Network: (Tanh, MSE, SGD)")
del Model
del sgd


# # Activation: Tanh, Criterion: MSE, Optimizer: Adam

# In[13]:


nb_epochs = 150
Model = Sequential([Linear(2,25), Tanh(), Linear(25,25), Tanh(), Linear(25,25), Tanh(), Linear(25,1), Sigmoid()])
adam = AdamOptim(Model, 0.01)
print("*** Activation: Tanh ***", "\n*** Criterion: MSE ***", "\n*** Optimizer: Adam ***")
res, train_err, test_err = train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, adam, False, True, MSE())   
plot_figure(test_input, res, "Testing with (Tanh, MSE, Adam) ")
#plot_err(train_err, test_err, nb_epochs, "Network: (Tanh, MSE, Adam)")

del Model
del adam


# # Activation: Tanh, Criterion: MSE, Optimizer: None

# In[28]:


nb_epochs = 300
Model = Sequential([Linear(2,25), Tanh(), Linear(25,25), Tanh(), Linear(25,25), Tanh(), Linear(25,1), Sigmoid()])
print("*** Activation: Tanh ***", "\n*** Criterion: MSE ***", "\n*** Optimizer: None ***")
res, train_err, test_err = train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, None, False, False, MSE())   
plot_figure(test_input, res, "Testing with (Tanh, MSE, Without optimizer) ")
#plot_err(train_err, test_err, nb_epochs, "Network: (Tanh, MSE, Without optimizer)")

del Model


# # Activation: Tanh, Criterion: BCE, Optimizer: SGD

# In[15]:


nb_epochs = 300
Model = Sequential([Linear(2,25), Tanh(), Linear(25,25), Tanh(), Linear(25,25), Tanh(), Linear(25,1), Sigmoid()])
sgd = SGD(Model, 0.001)
print("*** Activation: Tanh ***", "\n*** Criterion: BCE ***", "\n*** Optimizer: SGD ***")
res, train_err, test_err = train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, sgd, True, False, BCE())   
plot_figure(test_input, res, "Testing with (Tanh, BCE, SGD) ")
#plot_err(train_err, test_err, nb_epochs, "Network: (Tanh, BCE, SGD)")

del Model
del sgd


# # Activation: Tanh, Criterion: BCE, Optimizer: Adam

# In[16]:


nb_epochs = 150
Model = Sequential([Linear(2,25), Tanh(), Linear(25,25), Tanh(), Linear(25,25), Tanh(), Linear(25,1), Sigmoid()])
adam = AdamOptim(Model, 0.01)
print("*** Activation: Tanh ***", "\n*** Criterion: BCE ***", "\n*** Optimizer: Adam ***")
res, train_err, test_err = train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, adam, False, True, BCE())   
plot_figure(test_input, res, "Testing with (Tanh, BCE, Adam) ")
#plot_err(train_err, test_err, nb_epochs, "Network: (Tanh, BCE, Adam)")

del Model
del adam


# # Activation: Tanh, Criterion: BCE, Optimizer: None

# In[17]:


nb_epochs = 300
Model = Sequential([Linear(2,25), Tanh(), Linear(25,25), Tanh(), Linear(25,25), Tanh(), Linear(25,1), Sigmoid()])
print("*** Activation: Tanh ***", "\n*** Criterion: BCE ***", "\n*** Optimizer: None ***")
res, train_err, test_err = train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, None, False, False, BCE())   
plot_figure(test_input, res, "Testing with (Tanh, BCE, Without optimizer) ")
#plot_err(train_err, test_err, nb_epochs, "Network: (Tanh, BCE, Without optimizer)")

del Model


# # Activation: LeakyReLU, Criterion: MSE, Optimizer: SGD

# In[18]:


nb_epochs = 300
Model = Sequential([Linear(2,25), LeakyReLU(), Linear(25,25), LeakyReLU(), Linear(25,25), LeakyReLU(), Linear(25,1), Sigmoid()])
sgd = SGD(Model, 0.001)
print("*** Activation: LeakyReLU ***", "\n*** Criterion: MSE ***", "\n*** Optimizer: SGD ***")
res, train_err, test_err = train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, sgd, True, False, MSE())   
plot_figure(test_input, res, "Testing with (LeakyReLU, MSE, SGD) ")
#plot_err(train_err, test_err, nb_epochs, "Network: (LeakyReLU, MSE, SGD)")

del Model
del sgd


# # Activation: LeakyReLU, Criterion: MSE, Optimizer: Adam

# In[19]:


nb_epochs = 150
batch_size = 5
Model = Sequential([Linear(2,25), LeakyReLU(), Linear(25,25), LeakyReLU(), Linear(25,25), LeakyReLU(), Linear(25,1), Sigmoid()])
adam = AdamOptim(Model, 0.01)
print("*** Activation: LeakyReLU ***", "\n*** Criterion: MSE ***", "\n*** Optimizer: Adam ***")
res, train_err, test_err = train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, adam, False, True, MSE())   
plot_figure(test_input, res, "Testing with (LeakyReLU, MSE, Adam) ")
#plot_err(train_err, test_err, nb_epochs, "Network: (LeakyReLU, MSE, Adam)")

del Model
del adam


# # Activation: LeakyReLU, Criterion: MSE, Optimizer: None

# In[21]:


nb_epochs = 300
batch_size = 5
Model = Sequential([Linear(2,25), LeakyReLU(), Linear(25,25), LeakyReLU(), Linear(25,25), LeakyReLU(), Linear(25,1), Sigmoid()])
print("*** Activation: LeakyReLU ***", "\n*** Criterion: MSE ***", "\n*** Optimizer: None ***")
res, train_err, test_err = train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, None, False, False, MSE())   
plot_figure(test_input, res, "Testing with (LeakyReLU, MSE, Without optimizer) ")
#plot_err(train_err, test_err, nb_epochs, "Network: (LeakyReLU, MSE, Without optimizer)")

del Model


# # Activation: LeakyReLU, Criterion: BCE, Optimizer: SGD

# In[22]:


nb_epochs = 300
Model = Sequential([Linear(2,25), LeakyReLU(), Linear(25,25), LeakyReLU(), Linear(25,25), LeakyReLU(), Linear(25,1), Sigmoid()])
sgd = SGD(Model, 0.001)
print("*** Activation: LeakyReLU ***", "\n*** Criterion: BCE ***", "\n*** Optimizer: SGD ***")
res, train_err, test_err = train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, sgd, True, False, BCE())   
plot_figure(test_input, res, "Testing with (LeakyReLU, BCE, SGD) ")
#plot_err(train_err, test_err, nb_epochs, "Network: (LeakyReLU, BCE, SGD)")

del Model
del sgd


# # Activation: LeakyReLU, Criterion: BCE, Optimizer: Adam

# In[23]:


nb_epochs = 150
batch_size = 5
Model = Sequential([Linear(2,25), LeakyReLU(), Linear(25,25), LeakyReLU(), Linear(25,25), LeakyReLU(), Linear(25,1), Sigmoid()])
adam = AdamOptim(Model, 0.01)
print("*** Activation: LeakyReLU ***", "\n*** Criterion: BCE ***", "\n*** Optimizer: Adam ***")
res, train_err, test_err = train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, adam, False, True, BCE())   
plot_figure(test_input, res, "Testing with (LeakyReLU, BCE, Adam) ")
#plot_err(train_err, test_err, nb_epochs, "Network: (LeakyReLU, BCE, Adam)")

del Model
del adam


# # Activation: LeakyReLU, Criterion: BCE, Optimizer: None

# In[25]:


nb_epochs = 300
batch_size = 5
Model = Sequential([Linear(2,25), LeakyReLU(), Linear(25,25), LeakyReLU(), Linear(25,25), LeakyReLU(), Linear(25,1), Sigmoid()])
print("*** Activation: LeakyReLU ***", "\n*** Criterion: BCE ***", "\n*** Optimizer: None ***")
res, train_err, test_err = train(Model, train_input, train_target,test_input, test_target, nb_epochs, batch_size, None, False, False, BCE())   
plot_figure(test_input, res, "Testing with (LeakyReLU, BCE, Without optimizer) ")
#plot_err(train_err, test_err, nb_epochs, "Network: (LeakyReLU, BCE, Without optimizer)")

del Model


# In[ ]:




