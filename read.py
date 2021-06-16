class Read:

	def __init__(self,filename):
		self.filename = filename

	def GetData(self):
		finList=[]
		
		wordFile = open(self.filename)
		
		for word in wordFile:
			#print(word)
			word = word.splitlines()
			finList.append(word)
		
		return finList

f = Read('stockData.txt').GetData()
print(len(f))
for i in f:
	print()
	print()
	print(i)