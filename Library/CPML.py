"""
Circuit Python machine learning tool kit

This library combines and provides functionality relevant to making neural networks on devices.
"""

import ulab.numpy as np
import random
import math as maths

"""
generate a normal distribution randomized
@param: Mean is the mean of the normal
@param: StdDev is the standard deviation of the normal
@param: size is the shape of the matrix
"""
def normal(Mean=0,StdDev=0.5,size=[5]):
    ar=np.zeros(size)
    shape=ar.shape
    ar=ar.flatten() #generate numpy
    for i in range(len(ar)): #generate random values
        X=random.randint(0,10)-Mean
        f_X=(1/maths.sqrt(2*maths.pi*StdDev**2))*maths.exp(-1*(X-Mean)**2/(2*StdDev**2))
        ar[i]=f_X #set random value
    return ar.reshape(shape)

class Layer:
    def __init__(self,nodes_in,nodes_out,vals=None):
        self.vals=vals
        if vals!=None:
            self.matrix=vals.reshape((nodes_in,nodes_out)) #generate random weights
        else:
            self.matrix=normal(size=(nodes_in,nodes_out)) #generate random weights
    def __mul__(self,other):
        return np.dot(other,self.matrix) #multiply the matrices together
    def getShape(self): #return the shape of the matrix
        return self.matrix.shape
class Network:
    def __init__(self,num_out):
        self.network=[]
        self.num_out=num_out
    def add_layer(self,nodes,vals=None):
        layer=Layer(nodes,self.num_out,vals=vals) #default x by y
        if len(self.network)>0: #there are previous nodes
            layer=self.network[-1][0]
            bias=self.network[-1][1]
            num=layer.getShape()
            val=layer.vals
            self.network[-1]=[Layer(num[0],nodes,vals=val),bias] #correct output of matrices before
            layer=Layer(nodes,self.num_out,vals=vals) #generate layer with correct matrices
        self.network.append([layer,None]) #add the layer to the network
    def add_bias(self,vals=None):
        assert len(self.network)>0, "Network is empty. Add layers"
        size=self.network[-1][0].getShape()[0] #get the end sizing to add on
        if vals==None:
            vals=normal(size=(size,1))
        self.network[-1]=[self.network[-1][0],vals]
    def forward(self,inp):
        assert len(self.network)>0, "Network is empty. Add layers"
        x=inp * self.network[0].matrix
        sub=self.network[1:-1]
        for layer in sub:
            x = x * layer
        return x
        


