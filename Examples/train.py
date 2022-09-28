from CPML import *
import ulab.numpy as np

#create neural network
net=Network(2)
net.add_layer(2)
net.add_layer(7)
net.add_layer(5)
net.show()

#set up logic gate data and expected outcome
X_data=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0,0],[1,1],[1,1],[0,0]])


#run training loop    
net.train(X_data,y,1,0.05)
