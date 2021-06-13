import yfinance
from yfinance import base
from yfinance import Ticker,multi
from yfinance import Tickers
from yfinance import ticker
from yfinance import tickers

finList=['MSFT','AAPL']

quarterlies = []

for i in finList:
	break
	stock = base.TickerBase(i)
	print(stock.get_earnings(freq='quarterly'))

for i in finList:
	stock = base.TickerBase(i)
	print(stock.get_balance_sheet(freq='quarterly'))

for i in finList:
	break
	stock = Ticker(i)
	print(stock.history(period="1mo", interval="1d",
                start=None, end=None, prepost=False,
                actions=True, auto_adjust=True, proxy=None,
                threads=True, group_by='column', progress=True))