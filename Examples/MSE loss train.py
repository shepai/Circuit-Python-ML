from CPML import *
import ulab.numpy as np

def activation(nump):
    s=np.sum(nump)
    if s>=0:
        return np.array([1])
    return np.array([0])
    
#create neural network
net=Network(2)
net.add_layer(2,vals=np.array([1,0,1,0]))
net.add_bias(vals=np.array([-1,-1]))
net.show()

#set up logic gate data and expected outcome
X_data=[np.array([[0],[0]]),np.array([[0],[1]]),np.array([[1],[0]]),np.array([[1],[1]])]
y=np.array([[0],[0],[0],[1]])


y_pred=np.zeros(y.shape)

for i in range(len(X_data)):
    y_pred[i]=activation(net.forward(X_data[i]).transpose()[0])
    
loss=MSE(y,y_pred)
print(loss)