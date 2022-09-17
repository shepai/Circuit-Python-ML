from CPML import *
import ulab.numpy as np

def activation(nump):
    s=np.sum(nump.transpose()[0])
    if s>=0:
        return np.array([1])
    return np.array([0])
    
#create neural network
net=Network(2)
net.add_layer(2,vals=np.array([0,0,0,0]))
net.add_bias(vals=np.array([0,0]))
net.show()

#set up logic gate data and expected outcome
X_data=[np.array([[0],[0]]),np.array([[0],[1]]),np.array([[1],[0]]),np.array([[1],[1]])]
y=np.array([[0],[0],[0],[1]])

#run training loop    
net.train(X_data,y,10,activation,0.5)

print(net.network[0].matrix,net.network[0].bias)