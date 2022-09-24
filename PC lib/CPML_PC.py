"""
This code is to be run on the PC side,

It makes use of the neural network training
"""
import torch
import torch.nn as nn
from CPML_PC import *
import numpy as np

output_nodes=5

def reform(net,weights):
    #does not yet support biases
    for i,weight in enumerate(weights):
        net.network[i].matrix=weight.detach().numpy()

X_data=torch.tensor(np.random.rand(100,10,1))
y_data=torch.tensor(np.random.rand(100,output_nodes))
print(y_data.shape)


network=Network(output_nodes)
network.add_layer(10)
network.add_layer(6)

#train network
epochs=100
lr=0.05

criterion = nn.MSELoss()

a=network.parameters()
for i in range(len(network.network)): #convert to tensors
    network.network[i].matrix=nn.Parameter(torch.tensor(network.network[i].matrix))

a=[network.network[i].matrix for i in range(len(network.network))]

optimizer = torch.optim.Adam(a, lr=lr)  # Let's try a different optimizer!

print(np.sum(a[1].detach().numpy()))
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
        output=np.sum(output, axis=0) #get nodes of output
        y_pred=torch.sigmoid(torch.tensor(output))
        #get best
        #y_pred=np.argmax(output)
        if torch.sum(y_pred)==torch.sum(lab):
            acc+=1
        #t1=y_pred.reshape(1,output_nodes).requires_grad_(True)
        #t2=lab.reshape(1,output_nodes).requires_grad_(True)
        #print(t1.shape,t2.shape)
        # Calculate loss
        loss = criterion(y_pred.requires_grad_(True),lab)
        # Calculate gradients
        loss.backward()
        # Update weights
        # Backward and optimize
        optimizer.step()
        optimizer.zero_grad()

        l+=int(loss)

    #reform(network,a) #try copy over incase a by-reference doesn't work
    if epoch%10==0: #sjow accuracy
        print("Epoch",epoch,"accuracy:",acc/len(X_data) *100,"loss:",l)
        print(np.sum(a[1].detach().numpy()))
print("End accuracy:",acc/len(X_data) *100)
