from CPML import *
import ulab.numpy as np

data=normal(0,0.5,(100,5))
label=normal(0,0.5,(100,3))

net=Network()
net.layers=[LinearLayer(5,12),ReLU(),LinearLayer(12,20),sigmoid(),LinearLayer(20,3)]
net.compile()

net.train(data,label,100,0.01,10)