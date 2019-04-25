import queue 
  
# From class queue, Queue is 
# created as an object Now L 
# is Queue of a maximum  
# capacity of 20 
L = queue.Queue(maxsize=20) 
  
# Data is inserted into Queue 
# using put() Data is inserted 
# at the end 
L.put(5) 
L.put(9) 
L.put(1) 
L.put(7) 
  
# get() takes data out from 
# the Queue from the head  
# of the Queue 
print("L",L)
print(L.get()) 
print(L.get()) 
print(L.get()) 
print(L.get()) 