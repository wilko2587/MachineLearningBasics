def most_frequent(List, default = 0):
  counter = []

  binary_list = [int(i==List[0]) for i in List] #convert the list to binary (1 or 0)

  for i in List:
    frequency = List.count(i)
    counter.append(frequency)

  if binary_list.count(0) == binary_list.count(1):
    return default #if a draw, use default
  else:
    return List[counter.index(max(counter))] #else, use the most common value


x = most_frequent(['a','b','a','c'])
print(x) # --> should return 'a'

y = most_frequent([1,1,1,0,0])
print(y) #--> should return 1