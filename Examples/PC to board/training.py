"""
This code is to be run on the PC side,

It makes use of the neural network training
"""
import torch
import torch.nn as nn
from CPML_PC import *
import numpy as np
import random

output_nodes=5

X_data=torch.tensor(np.random.rand(50,10,1),dtype=torch.float32)
y_data=torch.tensor(np.zeros((50,output_nodes)),dtype=torch.float32)

#make classes
for i in range(100):
    y_data[random.randint(0,output_nodes-1)]=1
    
print(y_data.shape)


network=Network(output_nodes)
network.add_layer(10)
network.add_layer(6)

#train network
epochs=500
lr=0.01

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(network.parameters(), lr=lr)  # Let's try a different optimizer!
 
print(np.sum(network.network[1].matrix.detach().numpy()))

for epoch in range(epochs):
    acc=0
    l=0
    for dat,lab in zip(X_data,y_data):
        #calculate loss
        # Clear gradients 
        optimizer.zero_grad()
        # Predict outputs 
        #pass through network
        output=network.forward(dat)
        output=torch.sum(output, axis=0) #get nodes of output
        y_pred=torch.sigmoid(output)
        c=torch.argmax(y_pred)
        #get best

        if (c)==(torch.sum(lab)):
            acc+=1
        # Calculate loss
        loss = criterion(y_pred.requires_grad_(True),lab)
        # Calculate gradients
        loss.backward()
        if str(loss.item())=="nan":
            print(output,y_pred,"error")
            network.show()
            print(network.network[0].matrix)
            print(network.network[1].matrix)
            exit()
        # Update weights 
        # Backward and optimize
        optimizer.step()
        optimizer.zero_grad()
        l+=abs(loss.item())
        
    #reform(network,a) #try copy over incase a by-reference doesn't work
    if epoch%100==0: #sjow accuracy
        print("Epoch",epoch,"accuracy:",acc/len(X_data) *100,"loss:",l)
        print(np.sum(network.network[1].matrix.detach().numpy()),)
print("End accuracy:",acc/len(X_data) *100)
