class Person:

	def __init__(self, name,age,aliases = []):
		self.name = name
		self.age = age
		self.aliases = aliases

	def __len__(self):
		return self.age

	def __getitem__(self, key):
		# if key is of invalid type or value, the list values will raise the error
		return self.aliases[key]

	def __setitem__(self, key, value):
		self.aliases[key] = value

	def __delitem__(self, key):
		del self.aliases[key]

	def __iter__(self):
		return iter(self.aliases)

	def __reversed__(self):
		return reversed(self.aliases)

	def append(self, value):
		self.aliases.append('system generated')
	def head(self):
		# get the first element
		return self.aliases[0]
	def tail(self):
		# get all elements after the first
		return self.aliases[1:]
	def __gt__(self, other):
		return self.age > other.age
	def __lt__(self, other):
		return self.age < other.age
	def __add__(self, other):
		return Person(self.name + other.name, self.age + other.age, self.aliases + other.aliases)

	def __str__(self):
		s1 = "My age is {}, my name is {}, and my aliases are :".format(self.age,self.name)
		for alias in self.aliases:
			s1 += " '" + str(alias) + "'"
		return s1

	def __repr__(self):
		s1 = "I am {} old, I am {}, I am also known by...:".format(self.age,self.name)
		for alias in self.aliases:
			s1 += " '" + str(alias) + "'"
		return s1

	def __float__(self):
		return float(self.age)
	def __int__(self):
		return int(self.age)
		
	def __copy__(self):
		return Person(self.name, self.age, self.alias)


if __name__ == '__main__'
	"""create two people"""
	dan = Person(name = 'Dan', age = 31, aliases = ['LittleFinger','Mollick', 'I looove R', 'I lost at ping pong', 'I am the new boss'])
	chris = Person(name = 'Chris', age = 33, aliases = ['The Boss','I rule at ping pong', 'I work hard on non work activites', 'I like dry erase boards'])
	"""try object the object specific definitions"""
	chris # __repr__
	print(dan) #__str__
	len(dan) ##__len__
	chris[0] ##__getitem__
	chris[0] = 'Not the boss'##__setitem__
	del dan[3] #__delitem__
	for alias in dan: #__iter__
		print(alias)
	for alias in reversed(dan): #__iter__ & reverse
		print(alias)
	dan.append('will this work?') #append -- we hardcoded the value, so it wont
	dan.tail() #tail
	chris > dan #__gt__
	chris < dan #__lt__
	dan + chris #__add__
	james = dan + chris #__copy__
	float(james) #__float__