"""
This code is to be run on the PC side,

It makes use of the neural network training
"""
import torch
import torch.nn as nn
from CPML_PC import *
import numpy as np

def reform(net,weights):
    #does not yet support biases
    for i,weight in enumerate(weights):
        net.network[i].matrix=weight.detach().numpy()
        
X_data=np.random.rand(100,10,1)
y_data=torch.tensor(np.random.randint(5,size=100))

network=Network(5)
network.add_layer(10)
network.add_layer(6)

#train network
epochs=100
lr=0.05

criterion = nn.MSELoss()
a=network.parameters()
a=[nn.Parameter(torch.tensor(a[i])) for i in range(len(a))]
optimizer = torch.optim.SGD(a, lr=lr)  # Let's try a different optimizer!

print(np.sum(a[1].detach().numpy()))
for epoch in range(epochs):
    acc=0
    l=0
    for dat,lab in zip(X_data,y_data):
        #calculate loss
        #print(torch.tensor(np.array([y_pred],dtype=np.float64), requires_grad=True).reshape(1,1),
        #torch.tensor(np.array([lab],dtype=np.float64), requires_grad=True).reshape(1,1))
        #print(torch.tensor(np.array([y_pred],dtype=np.float64), requires_grad=True).grad)
        # Clear gradients 
        optimizer.zero_grad()
        # Predict outputs 
        #pass through network
        output=network.forward(dat)
        output=np.sum(output, axis=1) #get nodes of output
        #get best
        y_pred=np.argmax(output)
        if y_pred==lab:
            acc+=1
        t1=torch.tensor(np.array([y_pred],dtype=np.float64), requires_grad=True).reshape(1,1)
        t2=torch.tensor(np.array([lab],dtype=np.float64)).reshape(1,1)
        # Calculate loss 
        loss = criterion(t1,t2)
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
