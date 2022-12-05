#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Neural_Nets import *
from Neural_Nets_with_Auxiliary_Losses import *
from Neural_Nets_with_Weight_sharing import *
from Neural_Nets_with_WS_AL import *


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


def std_mean(ls):
    n = len(ls)
    mean = sum(ls) / n
    var = sum((x - mean)**2 for x in ls) / n
    std_dev = var**0.5
    return std_dev, mean


# In[4]:


mini_batch_size = 200
learning_rate = 0.01


# # Neural Net

# In[5]:


train_err_LeNet, test_err_LeNet, train_acc_LeNet, test_acc_LeNet, tr_errs_lenet = nn_run(mini_batch_size,
                                                                    learning_rate, mod=1)


# In[6]:


train_err_LeNet_std, train_err_LeNet_mean = std_mean(train_err_LeNet)
test_err_LeNet_std, test_err_LeNet_mean = std_mean(test_err_LeNet)

train_acc_LeNet_std, train_acc_LeNet_mean = std_mean(train_acc_LeNet)
test_acc_LeNet_std, test_acc_LeNet_mean = std_mean(test_acc_LeNet)


# In[7]:


print("*** Network: LeNet5 *** \n\nTrain accuracy: ", train_acc_LeNet,
      "  Std: ",round(train_acc_LeNet_std,3),",  Mean: ",round(train_acc_LeNet_mean,3),
      "\nTest accuracy: ", test_acc_LeNet, "  Std: ", round(test_acc_LeNet_std,3),
      ",  Mean: ", round(test_acc_LeNet_mean,3),
      "\n\nTrain errors: ", train_err_LeNet,
      "  Std: ",round(train_err_LeNet_std,3),",  Mean: ",round(train_err_LeNet_mean,3) ,
      "\nTest errors: ", test_err_LeNet, "  Std: ", round(test_err_LeNet_std,3), ",  Mean: ", round(test_err_LeNet_mean,3))


# In[8]:


train_err_MLP, test_err_MLP, train_acc_MLP, test_acc_MLP, tr_errs_nn = nn_run(mini_batch_size,
                                                            learning_rate, mod=0)
 


# In[9]:


train_err_MLP_std, train_err_MLP_mean = std_mean(train_err_MLP)
test_err_MLP_std, test_err_MLP_mean = std_mean(test_err_MLP)

train_acc_MLP_std, train_acc_MLP_mean = std_mean(train_acc_MLP)
test_acc_MLP_std, test_acc_MLP_mean = std_mean(test_acc_MLP)


# In[10]:


print("\n\n*** Network: MLPNet *** \n\nTrain accuracy: ", train_acc_MLP,
      "  Std: ",round(train_acc_MLP_std,3),",  Mean: ",round(train_acc_MLP_mean,3),
      "\nTest accuracy: ", test_acc_MLP, "  Std: ", round(test_acc_MLP_std,3),
      ",  Mean: ", round(test_acc_MLP_mean,3),
      "\n\nTrain errors: ", train_err_MLP,
      "  Std: ",round(train_err_MLP_std,3),",  Mean: ",round(train_err_MLP_mean,3) ,
      "\nTest errors: ", test_err_MLP, "  Std: ", round(test_err_MLP_std,3), ",  Mean: ", round(test_err_MLP_mean,3))


# # Weight Sharing

# In[11]:


train_err_WS_MLP, test_err_WS_MLP, train_acc_WS_MLP, test_acc_WS_MLP, tr_errs_nn_ws = ws_run(mini_batch_size,
                                                            learning_rate, mod=0)


# In[12]:


train_err_WS_MLP_std, train_err_WS_MLP_mean = std_mean(train_err_WS_MLP)
test_err_WS_MLP_std, test_err_WS_MLP_mean = std_mean(test_err_WS_MLP)

train_acc_WS_MLP_std, train_acc_WS_MLP_mean = std_mean(train_acc_WS_MLP)
test_acc_WS_MLP_std, test_acc_WS_MLP_mean = std_mean(test_acc_WS_MLP)


# In[13]:


print("\n\n*** Network: MLPNet with Weight Sharing *** \n\nTrain accuracy: ", train_acc_WS_MLP,
      "  Std: ",round(train_acc_WS_MLP_std,3),",  Mean: ",round(train_acc_WS_MLP_mean,3),
      "\nTest accuracy: ", test_acc_WS_MLP, "  Std: ", round(test_acc_WS_MLP_std,3),
      ",  Mean: ", round(test_acc_WS_MLP_mean,3),
      "\n\nTrain errors: ", train_err_WS_MLP,
      "  Std: ",round(train_err_WS_MLP_std,3),",  Mean: ",round(train_err_WS_MLP_mean,3) ,
      "\nTest errors: ", test_err_WS_MLP, "  Std: ", round(test_err_WS_MLP_std,3), ",  Mean: ", round(test_err_WS_MLP_mean,3))


# In[14]:


train_err_WS_LeNet, test_err_WS_LeNet, train_acc_WS_LeNet, test_acc_WS_LeNet, tr_errs_lenet_ws = ws_run(mini_batch_size,
                                                            learning_rate, mod=1)


# In[15]:


train_err_WS_LeNet_std, train_err_WS_LeNet_mean = std_mean(train_err_WS_LeNet)
test_err_WS_LeNet_std, test_err_WS_LeNet_mean = std_mean(test_err_WS_LeNet)

train_acc_WS_LeNet_std, train_acc_WS_LeNet_mean = std_mean(train_acc_WS_LeNet)
test_acc_WS_LeNet_std, test_acc_WS_LeNet_mean = std_mean(test_acc_WS_LeNet)


# In[16]:


print("\n\n*** Network: LeNet5 with Weight Sharing *** \n\nTrain accuracy: ", train_acc_WS_LeNet,
      "  Std: ",round(train_acc_WS_LeNet_std,3),",  Mean: ",round(train_acc_WS_LeNet_mean,3),
      "\nTest accuracy: ", test_acc_WS_LeNet, "  Std: ", round(test_acc_WS_LeNet_std,3),
      ",  Mean: ", round(test_acc_WS_LeNet_mean,3),
      "\n\nTrain errors: ", train_err_WS_LeNet,
      "  Std: ",round(train_err_WS_LeNet_std,3),",  Mean: ",round(train_err_WS_LeNet_mean,3) ,
      "\nTest errors: ", test_err_WS_LeNet, "  Std: ", round(test_err_WS_LeNet_std,3), ",  Mean: ", round(test_err_WS_LeNet_mean,3))


# # Auxiliary losses

# In[17]:


train_err_ALoss_MLP, test_err_ALoss_MLP, train_acc_ALoss_MLP, test_acc_ALoss_MLP, tr_errs_nn_al = axLoss_run(mini_batch_size,
                                                            learning_rate, mod=0)


# In[18]:


train_err_ALoss_MLP_std, train_err_ALoss_MLP_mean = std_mean(train_err_ALoss_MLP)
test_err_ALoss_MLP_std, test_err_ALoss_MLP_mean = std_mean(test_err_ALoss_MLP)


train_acc_ALoss_MLP_std, train_acc_ALoss_MLP_mean = std_mean(train_acc_ALoss_MLP)
test_acc_ALoss_MLP_std, test_acc_ALoss_MLP_mean = std_mean(test_acc_ALoss_MLP)


# In[19]:


print("\n\n*** Network: MLPNet with Auxiliary losses *** \n\nTrain accuracy: ", train_acc_ALoss_MLP,
      "  Std: ",round(train_acc_ALoss_MLP_std,3),",  Mean: ",round(train_acc_ALoss_MLP_mean,3),
      "\nTest accuracy: ", test_acc_ALoss_MLP, "  Std: ", round(test_acc_ALoss_MLP_std,3),
      ",  Mean: ", round(test_acc_ALoss_MLP_mean,3),
      "\n\nTrain errors: ", train_err_ALoss_MLP,
      "  Std: ",round(train_err_ALoss_MLP_std,3),",  Mean: ",round(train_err_ALoss_MLP_mean,3) ,
      "\nTest errors: ", test_err_ALoss_MLP, "  Std: ", round(test_err_ALoss_MLP_std,3), ",  Mean: ", round(test_err_ALoss_MLP_mean,3))


# In[20]:


train_err_ALoss_LeNet, test_err_ALoss_LeNet, train_acc_ALoss_LeNet, test_acc_ALoss_LeNet, tr_errs_lenet_al = axLoss_run(mini_batch_size,
                                                            learning_rate, mod=1)


# In[21]:


train_err_ALoss_LeNet_std, train_err_ALoss_LeNet_mean = std_mean(train_err_ALoss_LeNet)
test_err_ALoss_LeNet_std, test_err_ALoss_LeNet_mean = std_mean(test_err_ALoss_LeNet)


train_acc_ALoss_LeNet_std, train_acc_ALoss_LeNet_mean = std_mean(train_acc_ALoss_LeNet)
test_acc_ALoss_LeNet_std, test_acc_ALoss_LeNet_mean = std_mean(test_acc_ALoss_LeNet)


# In[22]:


print("\n\n*** Network: LeNet5 with Auxiliary losses *** \n\nTrain accuracy: ", train_acc_ALoss_LeNet,
      "  Std: ",round(train_acc_ALoss_LeNet_std,3),",  Mean: ",round(train_acc_ALoss_LeNet_mean,3),
      "\nTest accuracy: ", test_acc_ALoss_LeNet, "  Std: ", round(test_acc_ALoss_LeNet_std,3),
      ",  Mean: ", round(test_acc_ALoss_LeNet_mean,3),
      "\n\nTrain errors: ", train_err_ALoss_LeNet,
      "  Std: ",round(train_err_ALoss_LeNet_std,3),",  Mean: ",round(train_err_ALoss_LeNet_mean,3) ,
      "\nTest errors: ", test_err_ALoss_LeNet, "  Std: ", round(test_err_ALoss_LeNet_std,3), ",  Mean: ", round(test_err_ALoss_LeNet_mean,3))


# # Weight sharing and Auxiliary losses

# In[23]:


train_err_WSAL_MLP, test_err_WSAL_MLP, train_acc_WSAL_MLP, test_acc_WSAL_MLP, tr_errs_nn_wsal = WSAL_run(mini_batch_size,
                                                            learning_rate, mod=0)


# In[24]:


train_err_WSAL_MLP_std, train_err_WSAL_MLP_mean = std_mean(train_err_WSAL_MLP)
test_err_WSAL_MLP_std, test_err_WSAL_MLP_mean = std_mean(test_err_WSAL_MLP)


train_acc_WSAL_MLP_std, train_acc_WSAL_MLP_mean = std_mean(train_acc_WSAL_MLP)
test_acc_WSAL_MLP_std, test_acc_WSAL_MLP_mean = std_mean(test_acc_WSAL_MLP)


# In[25]:


print("\n\n*** Network: MLPNet with Weight sharing and Auxiliary losses *** \n\nTrain accuracy: ", train_acc_WSAL_MLP,
      "  Std: ",round(train_acc_WSAL_MLP_std,3),",  Mean: ",round(train_acc_WSAL_MLP_mean,3),
      "\nTest accuracy: ", test_acc_WSAL_MLP, "  Std: ", round(test_acc_WSAL_MLP_std,3),
      ",  Mean: ", round(test_acc_WSAL_MLP_mean,3),
      "\n\nTrain errors: ", train_err_WSAL_MLP,
      "  Std: ",round(train_err_WSAL_MLP_std,3),",  Mean: ",round(train_err_WSAL_MLP_mean,3) ,
      "\nTest errors: ", test_err_WSAL_MLP, "  Std: ", round(test_err_WSAL_MLP_std,3), ",  Mean: ", round(test_err_WSAL_MLP_mean,3))


# In[26]:


train_err_WSAL_LeNet, test_err_WSAL_LeNet, train_acc_WSAL_LeNet, test_acc_WSAL_LeNet, tr_errs_lenet_wsal = WSAL_run(mini_batch_size,
                                                            learning_rate, mod=1)


# In[27]:


train_err_WSAL_LeNet_std, train_err_WSAL_LeNet_mean = std_mean(train_err_WSAL_LeNet)
test_err_WSAL_LeNet_std, test_err_WSAL_LeNet_mean = std_mean(test_err_WSAL_LeNet)


train_acc_WSAL_LeNet_std, train_acc_WSAL_LeNet_mean = std_mean(train_acc_WSAL_LeNet)
test_acc_WSAL_LeNet_std, test_acc_WSAL_LeNet_mean = std_mean(test_acc_WSAL_LeNet)


# In[28]:


print("\n\n*** Network: LeNet5 with Weight sharing and Auxiliary losses *** \n\nTrain accuracy: ", train_acc_WSAL_LeNet,
      "  Std: ",round(train_acc_WSAL_LeNet_std,3),",  Mean: ",round(train_acc_WSAL_LeNet_mean,3),
      "\nTest accuracy: ", test_acc_WSAL_LeNet, "  Std: ", round(test_acc_WSAL_LeNet_std,3),
      ",  Mean: ", round(test_acc_WSAL_LeNet_mean,3),
      "\n\nTrain errors: ", train_err_WSAL_LeNet,
      "  Std: ",round(train_err_WSAL_LeNet_std,3),",  Mean: ",round(train_err_WSAL_LeNet_mean,3) ,
      "\nTest errors: ", test_err_WSAL_LeNet, "  Std: ", round(test_err_WSAL_LeNet_std,3), ",  Mean: ", round(test_err_WSAL_LeNet_mean,3))


# In[31]:


mlpList = [tr_errs_nn, tr_errs_nn_ws, tr_errs_nn_al, tr_errs_nn_wsal]
mlpList_labels = ["Basic MLPNet", "MLPNet + Weight Sharing", "MLPNet + Auxiliary losses", "MLPNet + Weight Sharing and Auxiliary losses"]

fig, axs = plt.subplots(2, 2)
plt.rcParams["figure.figsize"] = (12,9)

for i in range(len(mlpList)):
    
    axs[0, 0].plot(mlpList[0])
    axs[0, 0].set_title(mlpList_labels[0])
    axs[0, 1].plot(mlpList[1], 'tab:orange')
    axs[0, 1].set_title(mlpList_labels[1])
    axs[1, 0].plot(mlpList[2], 'tab:green')
    axs[1, 0].set_title(mlpList_labels[2])
    axs[1, 1].plot(mlpList[3], 'tab:red')
    axs[1, 1].set_title(mlpList_labels[3])

for ax in axs.flat:
    ax.set(xlabel='Epoches', ylabel='Loss')   
fig.tight_layout(pad=5.0)




# In[30]:


mlpList = [tr_errs_lenet, tr_errs_lenet_ws, tr_errs_lenet_al, tr_errs_lenet_wsal]
mlpList_labels = ["Basic LeNet5", "LeNet5 + Weight Sharing", "LeNet5 + Auxiliary losses", "LeNet5 + Weight Sharing and Auxiliary losses"]

fig, axs = plt.subplots(2, 2)
plt.rcParams["figure.figsize"] = (12,9)

for i in range(len(mlpList)):
    
    axs[0, 0].plot(mlpList[0])
    axs[0, 0].set_title(mlpList_labels[0])
    axs[0, 1].plot(mlpList[1], 'tab:orange')
    axs[0, 1].set_title(mlpList_labels[1])
    axs[1, 0].plot(mlpList[2], 'tab:green')
    axs[1, 0].set_title(mlpList_labels[2])
    axs[1, 1].plot(mlpList[3], 'tab:red')
    axs[1, 1].set_title(mlpList_labels[3])

for ax in axs.flat:
    ax.set(xlabel='Epoches', ylabel='Loss')
fig.tight_layout(pad=5.0)

    




# In[ ]:




