from CPML import *
import ulab.numpy as np

class myNet(Network): #create a network class similar to pytorch format
    def __init__(self,i,h,o):
        self.layer1=LinearLayer(i,h) #linear input layer
        self.out=LinearLayer(h,o) #output layer
        self.layers=[self.layer1,sigmoid(),self.out] #add activation in the middle
        self.compile() #must always compile changes to the architecture of self.layers to create the network

net=myNet(5,6,2)
#generate random data and labels
data=normal(size=(100,5))
labels=normal(size=(100,2))
print(net.forward(normal(0,0.5,(10,5))).shape) 
epochs=1000
learning_rate=0.001
net.train(data,labels,epochs,learning_rate) 