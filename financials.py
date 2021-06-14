import yfinance
from yfinance import base
from yfinance import Ticker,multi
from yfinance import Tickers
from yfinance import ticker
from yfinance import tickers

finList=['MSFT']





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
					#"Timestamp('"+date+ " 00:00:00'):"
					if date not in self.dates:
						self.dates.append(date)		


	def getBalance(self):
		for i in self.data:
			contain = {}
			contain['name'] = i
			stock = base.TickerBase(i)
			#print(stock.get_balance_sheet(freq='quarterly'))
			example = stock.get_balance_sheet(freq='quarterly')
			for j in example:
				contain[str(j)] = []
				for k in example[j]:
					if k >0 and len(contain[str(j)])<25:
						contain[str(j)].append(k)
			self.quarterlies.append(contain)
		#print(self.quarterlies)

	'2021-03-31 00:00:00'
	'2020-12-31 00:00:00'
	'2020-09-30 00:00:00'
	'2020-06-30 00:00:00'
	def getPrices(self):
		for d in range(len(self.dates)):
			for q in self.quarterlies:
				#print(q.keys())
				if self.dates[d]+ " 00:00:00" in q.keys():

					for i in self.data:
						stock = base.TickerBase(i)
						period = stock.history(period='1d',interval="1d",start = self.dates[d])
						#print(len(q[self.dates[d]+ " 00:00:00"]))
						if len(q[self.dates[d]+ " 00:00:00"])<26:
							for h in period['Close'].keys():
								price_1 = period['Close'][h]
							q[self.dates[d]+ " 00:00:00"].append(("P1",price_1))
							#adding 5 will always land on a trading day
							period_2 = stock.history(period='1d',interval="1d",start = self.dates[d+5])
							for h in period_2['Close'].keys():
								price_2 = period_2['Close'][h]
							q[self.dates[d]+ " 00:00:00"].append(("P2",price_2))
		print(self.quarterlies)
							




findata = FinData(finList)
findata.getBalance()
findata.dateIter()

findata.getPrices()
