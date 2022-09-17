from CPML import *
import ulab.numpy as np

net=Network(2)
net.add_layer(2,vals=np.array([0,0,0,0]))
net.add_bias(vals=np.array([-1,1]))


x=np.array([0,0])

num=net.forward(x)
print(num)
