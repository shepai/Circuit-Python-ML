from CPML import *
import ulab.numpy as np

#create neural network
net=Network(2)
net.add_layer(2)
net.add_layer(5)
net.show()


#set up logic gate data and expected outcome
X_data=normal(mean=0,std=1,size=(8,2)) #N by input nodes
y=normal(size=(8,3)) #N by output nodes

def fitness(y,preds): #fitness function
    correct=0
    p=preds.transpose()
    for i in range(len(y)): #calculate how correct the prediction was
        if np.argmax(p[i])==np.argmax(y[i]):
            correct+=1
    return correct/len(y)

#run training loop    
net.trainGA(X_data,y,100,0.2,fitnessFunc=fitness)


