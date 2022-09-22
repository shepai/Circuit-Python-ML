from CPML import *
import ulab.numpy as np

def activation(nump):
    #activation function for the or gate
    s=np.sum(nump.transpose())
    if s>0:
        return np.zeros(nump.shape)+1 #representing as [on,off] as the output of the network
    return np.zeros(nump.shape)
#create the network
net=Network(2)
net.add_layer(2,vals=np.array([1,0,1,0]),act=activation)
net.add_bias(vals=np.array([0,0]))


#off
x=np.array([[0],[0]])
num=net.forward(x)
print(np.sum(num)/4>0)

#on
x=np.array([[0],[1]])
num=net.forward(x)
print(np.sum(num)/4>0)

#on
x=np.array([[1],[0]])
num=net.forward(x)
print(np.sum(num)/4>0)

#on
x=np.array([[1],[1]])
num=net.forward(x)
print(np.sum(num)/4>0)