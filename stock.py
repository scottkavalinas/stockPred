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
device = torch.device("cuda")

Batch_size = 1 # train in batches
epoch = 10   # amount of training iterations before testing

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


max_p1 = 0
max_earn = 0
max_exp= 0
max_r1=0
max_r2=0
max_r3=0
max_p2=0
for i in data:
	i['initial Price'] = stock['initial Price'] * random.randint(1,5)
	if i['initial Price'] > max_p1:
		max_p1= i['initial Price']
	i['earnings'] = stock['earnings'] * random.randint(1,5)
	if i['earnings'] > max_earn:
		max_earn=i['earnings']
	i['expenses'] = stock['expenses'] * random.randint(1,5)
	if i['expenses'] > max_exp:
		max_exp=i['expenses']
	i['ratio 1'] = stock['ratio 1'] * random.randint(1,5)
	if i['ratio 1']> max_r1:
		max_r1=i['ratio 1']
	i['ratio 2'] = stock['ratio 2'] * random.randint(1,5)
	if i['ratio 2'] > max_r2:
		max_r2 = i['ratio 2'] 
	i['ratio 3'] = stock['ratio 3'] * random.randint(1,5)
	if i['ratio 3'] > max_r3:
		max_r3=i['ratio 3']
	i['Price 2'] = stock['Price 2'] * random.randint(1,5)
	if i['Price 2'] > max_p2:
		max_p2=i['Price 2']

to_load=[]
for i in data:
	stock_tensor = (torch.tensor([i['initial Price']/max_p1,i['earnings']/max_earn,i['expenses']/max_exp,
		i['ratio 1']/max_r1,i['ratio 2']/max_r2,i['ratio 3']/max_r3],dtype=torch.double),i['Price 2']/max_p2)	
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
  	#x = x.view(x.size(0), -1)#Flatten
  	x = torch.sigmoid(self.FC_Layer_1(x)) #Activation Function
  	return x.double()


log_interval = 1
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
    #targetPrice = targetPrice.to(device)
    SGD_Optimizer.zero_grad() #compute gradient
    output = Fully_connected_EX(inputData) #get output from the model
    loss = criterion(output,targetPrice) 
    loss.backward() #Back Propogation
    SGD_Optimizer.step() # Update parameters
    if batchIndex % log_interval == 0:
      print('Training Epoch: {}\n{:.0f}%  Complete\tLoss: {:.6f}'.format(
        epoch, 100. * batchIndex / setSize, loss.item()))

def validate_network(epoch):
  Fully_connected_EX.eval()
  validation_loss = 0
  correct = 0
  setSize = len(Validation_DataLoader.dataset)
  with torch.no_grad():
    for inputData, targetPrice in (Validation_DataLoader):
      output = Fully_connected_EX(inputData)  #get output from the model
      validation_loss += criterion(output,targetPrice)
      predition_label = output.data.max(1, keepdim=True)[1] #get prediction
      #add correct predictions
      correct += predition_label.eq(targetPrice.data.view_as(predition_label)).sum()

  validation_loss /= setSize
  print('\nValidation set: Training Epoch {}\n Average loss: {:.8f}\n Accuracy: {}/{}= {:.2f}%\n'.format(
    epoch, validation_loss, correct, setSize,
    100. * correct / setSize))


def run():
  validate_network(0)
  for i in range(1,epoch+1):
    train_network(i)
    #validate_network(i)
  #test_network()
run()