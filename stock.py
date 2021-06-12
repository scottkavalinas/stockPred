import yfinance
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import random
from torch.utils.data import DataLoader as D
from torch.utils.data import random_split
import torch.optim as optim
import torchvision.datasets as Set

Batch_size = 2 # train in batches
epoch = 1   # amount of training iterations before testing

price = 10.0
earning = 100000.4
expenses = 110000.5
ratio_1 = 3.0
ratio_2 = 5.0
ratio_3 = 2.0
price_2 = 9.0
stock = {'initial Price':10.0,"earnings":100000.0, "expenses":110000.0,
"ratio 1": 3.0,"ratio 2": 5.0,"ratio 3": 2.0,"Price 2": 9.0}

data = []

for i in range(100):
	data.append({})

for i in data:
	i['initial Price'] = stock['initial Price'] * random.randint(1,5)
	i['earnings'] = stock['earnings'] * random.randint(1,5)
	i['expenses'] = stock['expenses'] * random.randint(1,5)
	i['ratio 1'] = stock['ratio 1'] * random.randint(1,5)
	i['ratio 2'] = stock['ratio 2'] * random.randint(1,5)
	i['ratio 3'] = stock['ratio 3'] * random.randint(1,5)
	i['Price 2'] = stock['Price 2'] * random.randint(1,5)

to_load=[]
for i in data:
	stock_tensor = (torch.tensor([i['initial Price'],i['earnings'],i['expenses'],
		i['ratio 1'],i['ratio 2'],i['ratio 3']],dtype=torch.double),i['Price 2'])	
	to_load.append(stock_tensor)

price_2 = np.array([9.0])

TV_split = [90, 10] 
training_set, validation_set = random_split(to_load, TV_split) #create validation subset

Training_DataLoader = D(training_set, batch_size = Batch_size, shuffle = False) #shuffle to randomize
Validation_DataLoader = D(validation_set, batch_size = Batch_size, shuffle = False)


class FullyConnectedNet(nn.Module):
  def __init__(self):
    super(FullyConnectedNet, self).__init__() 
    self.FC_Layer_1 = nn.Linear(6, 1)
    self.double()
  
  def forward(self,x):
    x = x.view(x.size(0), -1)##################???????????????????? Flatten
    x = F.relu(self.FC_Layer_1(x)) #Activation Function
    return x.double()


log_interval = 100
Fully_connected_EX = FullyConnectedNet()
learning_rate_SGD = 0.001
momentum_SGD = 0.9
#optimize with Stochastic Gradient descent
SGD_Optimizer = optim.SGD(Fully_connected_EX.parameters(), 
lr = learning_rate_SGD, momentum = momentum_SGD)

criterion = nn.MSELoss()

def train_network(epoch):
  setSize = len(Training_DataLoader)
  Fully_connected_EX.train()
  for batchIndex, (inputData, targetPrice) in enumerate(Training_DataLoader):
    #send input and target to GPU
    #inputData = inputData.to(device) 
    #targetLabel = targetLabel.to(device)
    SGD_Optimizer.zero_grad() #compute gradient
    output = Fully_connected_EX(inputData) #get output from the model
    loss = criterion(output,targetPrice) 
    print(loss)
    print(type(loss))
    loss.backward() #Back Propogation
    SGD_Optimizer.step() # Update parameters
    if batchIndex % log_interval == 0:
      print('Training Epoch: {}\n{:.0f}%  Complete\tLoss: {:.6f}'.format(
        epoch, 100. * batchIndex / setSize, loss.item()))

def run():
  #validate_network(0)
  for i in range(1,epoch+1):
    train_network(i)
    #validate_network(i)
  #test_network()
run()