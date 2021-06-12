import yfinance
import numpy as np
import torch
import random

price = 10.0
earning = 100000.4
expenses = 110000.5
ratio_1 = 3.0
ratio_2 = 5.0
ratio_3 = 2.0
price_2 = 9.0
stock = {'initial Price':10.0,"earnings":100000.0, "expenses":110000.0,"ratio 1": 3.0,"ratio 2": 5.0,"ratio 3": 2.0,"Price 2": 9.0}

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
	

print(data)


price_2 = np.array([9.0])