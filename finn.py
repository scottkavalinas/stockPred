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

from financials import FinData, Data



data_list = Data('sp500.txt').getData()

Batch_size = 1 # train in batches
epoch = 10   # amount of training iterations before testing
findata = FinData(data_list)
findata.dateIter()
findata.getBalance()
q = findata.getQuarterlyData()

maxVal= 100000000000

to_load = []
for i in q:
  temp = []
  for key in i.keys():
    temp.append(i[key])
  for j in range(len(temp)):
    for k in range(len(temp[j])):

      temp[j][k] = float(temp[j][k]/maxVal) 
      #print(temp[j][k])
  to_load.append((torch.tensor(np.array([temp[0],temp[1],temp[2]])),np.array(temp[3])))

TV_split = [45, 5] 
training_set, validation_set = random_split(to_load, TV_split) #create validation subset

Training_DataLoader = D(training_set, batch_size = Batch_size, shuffle = True) #shuffle to randomize
Validation_DataLoader = D(validation_set, batch_size = Batch_size, shuffle = True)
#Test_DataLoader	 =  D(test_load, batch_size = Batch_size, shuffle = True)

log_interval = 1


class FullyConnectedNet(nn.Module):
  def __init__(self):
    super(FullyConnectedNet, self).__init__() 
    self.FC_Layer_1 = nn.Linear(45, 15)
    #print(self.FC_Layer_1)
    #self.float()
  
  def forward(self,x):
    x=x.float()
    
    x = x.view(x.size(0), -1)#Flatten
    x = torch.sigmoid(self.FC_Layer_1(x)) #Activation Function
    
    #print(x)
    return x.float()

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
    SGD_Optimizer.zero_grad() #compute gradient
    output = Fully_connected_EX(inputData).float() #get output from the model
    output= output.float()
    targetPrice =targetPrice.float()
    loss = criterion(output,targetPrice) 
    loss.backward() #Back Propogation
    SGD_Optimizer.step() # Update parameters
    if batchIndex % log_interval == 1111110:
      print('Training Epoch: {}\n{:.0f}%  Complete\tLoss: {:.6f}'.format(
        epoch, 100. * batchIndex / setSize, loss.item()))

def validate_network(epoch):
  Fully_connected_EX.eval()
  validation_loss = 0
  correct = 0
  setSize = len(Validation_DataLoader.dataset)
  with torch.no_grad():
    for inputData, targetPrice in (Validation_DataLoader):
      inputData = inputData.float()
      targetPrice = targetPrice.float()
      print(targetPrice)
      output = Fully_connected_EX(inputData)*maxVal  #get output from the model
      #print(output)
      
      print("Cash:                      $" +str(output[0][0].item()))
      print("Total Liabilities:         $"+str(output[0][1].item()))
      print("Total Sotckholder Equity:  $"+str(output[0][2].item()))
      print("Other Current Liabilities: $"+str(output[0][3].item()))
      print("Total Assets:              $"+str(output[0][4].item()))
      print("Other Liab:                $"+str(output[0][5].item()))
      print("Treasury Stock:            $"+str(output[0][6].item()))
      print("Other Assets:              $"+str(output[0][7].item()))
      print("Total Current Liabilities: $"+str(output[0][8].item()))
      print("Other Stockholder Equity:  $"+str(output[0][9].item()))
      print("Property Plant Equipment:  $"+str(output[0][10].item()))
      print("Total Current Assets:      $"+str(output[0][11].item()))
      print("Net Tangible Assets:       $"+str(output[0][12].item()))
      print("Net Receivables:           $"+str(output[0][13].item()))
      print("Accounts Payable:          $"+str(output[0][14].item()))
      print()
      validation_loss += criterion(output,targetPrice)
      prediction_label =''
      #prediction_label = output.data.item()
      #prediction_label = output.data.max(0, keepdim=True)[0] #get prediction
      #add correct predictions
      #correct += prediction_label.eq(targetPrice.data.view_as(prediction_label)).sum()

  validation_loss /= setSize
  #print('\nValidation set: Training Epoch {}\n Average loss: {:.8f}\n Accuracy: {}/{}= {:.2f}%\n'.format(
    #epoch, validation_loss, correct, setSize,
    #100. * correct / setSize))

def test_network():
  Fully_connected_EX.eval()
  validation_loss = 0
  correct = 0
  setSize = len(Test_DataLoader.dataset)
  with torch.no_grad():
    for inputData, targetPrice in (Test_DataLoader):
      output = Fully_connected_EX(inputData)  
      validation_loss += criterion(output,targetPrice)
      predition_label = output.data.max(0, keepdim=True)[0]*maxVal
      correct += predition_label.eq(targetPrice.data.view_as(predition_label)).sum()

  validation_loss /= setSize
  print('\nTest set: \n Average loss: {:.8f} \n Accuracy: {}/{} ={:.2f}%\n'.format(
    validation_loss, correct, setSize,
    100. * correct / setSize))

def run():
  validate_network(0)
  for i in range(1,epoch+1):
    train_network(i)
    validate_network(i)
  #test_network()
run()