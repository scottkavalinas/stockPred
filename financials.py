import yfinance
from yfinance import base
from yfinance import Ticker,multi
from yfinance import Tickers
from yfinance import ticker
from yfinance import tickers



class Data:

	def __init__(self,filename):
		self.filename = filename

	def GetData(self):
		finList=[]
		word= ''
		wordFile = open(self.filename)
		for word in wordFile:
			word = word.upper()
			finList.append(word)
		data_list = []
		counter = 0
		alpha = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
		for i in finList:
			x=''
			for j in i:
				if j.lower() in alpha:
					x+=j
			if counter < 5:
				data_list.append(x)
				counter +=1
		return data_list


class FinData:
	def __init__(self,data):
		self.data = data
		self.quarterlies = []
		self.days = [1,2,3,4,5,6,7,8,9]
		self.months =  [1,2,3,4,5,6,7,8,9]
		self.years = ['2020-','2021-']
		self.dates = []

	def dateIter(self):
		day_temp = []
		month_temp = []
		for day_1 in self.days:
			day_temp.append('0'+str(day_1))
		day_temp.append('10')
		for day_2 in self.days:
			day_temp.append('1'+str(day_2))
		day_temp.append('20')
		for day_3 in self.days:
			day_temp.append('2'+str(day_3))
		day_temp.append('30')
		day_temp.append('31')
		for month in self.months:
			month_temp.append('0'+str(month))
		month_temp.append('10')
		month_temp.append('11')
		month_temp.append('12')
		for year in self.years:
			for month in month_temp:
				for i in range(len(day_temp)):
					if month == '02' and int(day_temp[i]) > 28:
						continue
					if int(day_temp[i]) > 30 and (month == '04' or month == '06' or month == '10' or month == '11'):
						continue
					date = year +month+'-' + day_temp[i]
					if date not in self.dates:
						self.dates.append(date)		


	def getBalance(self):
		for i in self.data:
			print(i)
			contain = {}
			contain['name'] = i
			stock = base.TickerBase(i)
			example = stock.get_balance_sheet(freq='quarterly')
			print(example)
			for j in example:
				contain[str(j)] = []
				for k in example[j]:
					if k >0 and len(contain[str(j)])<25:
						contain[str(j)].append(k)
			self.quarterlies.append(contain)

	def getPrices(self):
		for d in range(len(self.dates)):
			for q in self.quarterlies:
				if self.dates[d]+ " 00:00:00" in q.keys():
					for i in self.data:
						stock = base.TickerBase(i)
						period = stock.history(period='1d',interval="1d",start = self.dates[d],end=self.dates[d+3])
						if len(q[self.dates[d]+ " 00:00:00"])<27:
							for h in period['Close'].keys():
								price_1 = period['Close'][h]
							if type(q[self.dates[d]+ " 00:00:00"][-1]) != tuple:
								q[self.dates[d]+ " 00:00:00"].append(("P1",price_1))							
							period_2 = stock.history(period='1d',interval="1d",start = self.dates[d+3],end=self.dates[d+4])
							for h in period_2['Close'].keys():
								price_2 = period_2['Close'][h]
							if q[self.dates[d]+ " 00:00:00"][-1][0] != 'P2':
								q[self.dates[d]+ " 00:00:00"].append(("P2",price_2))
							print(len(q[self.dates[d]+ " 00:00:00"]))
							
	def getQuarterlyData(self):
		print(self.quarterlies)
		return self.quarterlies

data_list = Data('sp500.txt')
data_list=data_list.GetData()
findata = FinData(data_list)
findata.dateIter()
findata.getBalance()
findata.getPrices()
q = findata.getQuarterlyData()

f = open("stockData.txt", "a")
f.writelines(str(q))
f.close()
