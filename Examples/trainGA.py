from CPML import *
import ulab.numpy as np

#create neural network
net=Network(2)
net.add_layer(2)
net.add_layer(5)
net.show()

#set up logic gate data and expected outcome
X_data=normal(size=(8,2))
y=normal(size=(8,2))


#run training loop    
net.trainGA(X_data,y,100,0.2)

