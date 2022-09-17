from CPML import *
import ulab.numpy as np

def activation(nump):
    s=np.sum(nump)
    if s>=0:
        return 1
    return 0
    
net=Network(2)
net.add_layer(2,vals=np.array([1,0,1,0]))
net.add_bias(vals=np.array([-1,-1]))
net.show()

#off
x=np.array([[0],[0]])
num=net.forward(x).transpose()[0]
print(activation(num))

#off
x=np.array([[0],[1]])
num=net.forward(x).transpose()[0]
print(activation(num))

#off
x=np.array([[1],[0]])
num=net.forward(x).transpose()[0]
print(activation(num))

#on
x=np.array([[1],[1]])
num=net.forward(x).transpose()[0]
print(activation(num))
