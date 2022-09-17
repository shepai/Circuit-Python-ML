from CPML import *
import ulab.numpy as np

def activation(nump):
    #activation function for the and gate
    s=np.sum(nump.transpose()[0])
    if s>=0:
        return np.array([1])
    return np.array([0])
#create the network
net=Network(2)
net.add_layer(2,vals=np.array([1,0,1,0]),act=activation)
net.add_bias(vals=np.array([-1,-1]))
net.show()
""""""

#off
x=np.array([[0],[0]])
num=net.forward(x)
print(num)

#off
x=np.array([[0],[1]])
num=net.forward(x)
print(num)

#off
x=np.array([[1],[0]])
num=net.forward(x)
print(num)

#on
x=np.array([[1],[1]])
num=net.forward(x)
print(num)
