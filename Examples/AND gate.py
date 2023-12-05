from CPML import *
import ulab.numpy as np

net=Network()
layer=LinearLayer(2,2,bias=True)
layer.setLayer(np.array([[1,1],[1,1]]))
layer.setBias(np.array([[-2],[-2]]))
net.layers=[layer]
net.compile()
#off
x=np.array([0,0])
num=net.forward(x)
print(np.sum(num)/2>=0)

#off
x=np.array([0,1])
num=net.forward(x)
print(np.sum(num)/2>=0)

#off
x=np.array([1,0])
num=net.forward(x)
print(np.sum(num)/2>=0)

#on
x=np.array([1,1])
num=net.forward(x)
print(np.sum(num)/2>=0)