from CPML import *
import ulab.numpy as np

class myNet(Network):
    def __init__(self,i,h,o):
        self.layer1=LinearLayer(i,h)
        self.out=LinearLayer(h,o)
        self.layers=[self.layer1,sigmoid(),self.out]
        self.compile()

net=myNet(5,6,2)
data=normal(size=(100,5))
labels=normal(size=(100,2))
print(net.forward(normal(0,0.5,(10,5))).shape)
net.train(data,labels,100,0.1)