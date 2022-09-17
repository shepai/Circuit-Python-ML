from CPML import *
import ulab.numpy as np

#create neural network
net=Network(2)
net.add_layer(2)
net.add_layer(5)
net.add_layer(4)
net.show()

#set up logic gate data and expected outcome
X_data=[np.array([[0],[0]]),np.array([[0],[1]]),np.array([[1],[0]]),np.array([[1],[1]])]
y=np.array([[0],[0],[0],[1]])

net.forward(X_data[0])

#run training loop    
#net.train(X_data,y,10,activation,0.05)

print(net.network[0].matrix)