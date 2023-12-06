#train an auto encoder to recognise itself
#not yet finished

from CPML import *
import ulab.numpy as np

#set up data
data=normal(0,0.5,(100,5))
label=normal(0,0.5,(100,3))

#create network and train meaningful connections
net=AutoEncoder(5,3) #centre dim as 2
epochs=500
net.train(data,epochs,0.001,n_prints=100)

#get trained encoder
encoder=net.getEncoder()
#train
net2=Network()
net2.layers=encoder+[LinearLayer(3,10),sigmoid(),LinearLayer(10,3)] #add encoder layer to fnn layer
net2.compile()
for i in range(len(net2.layers)):
    if net2.layers[i].type=="layer":
        print(net2.layers[i].matrix.shape)
net2.train(data,label,epochs,0.001,n_prints=100)