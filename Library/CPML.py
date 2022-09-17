"""
Circuit Python machine learning tool kit

This library combines and provides functionality relevant to making neural networks on devices.

Library by Dexter R. Shepherd
University of Sussex PhD student

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

"""
generate a layer to hold information on network
@param: nodes_in is the number of inputs to this layer
@param: nodes_out is the number of nodes in the next layer
@param: vals is whether the user wishes to manually set the weights
"""
class Layer:
    def __init__(self,nodes_in,nodes_out,vals=None):
        if type(vals)==type(None):
            self.matrix=normal(size=(nodes_in,nodes_out)) #generate random weights
        else:
            self.matrix=vals.reshape((nodes_in,nodes_out)) #generate set weights
        self.vals=vals
        self.bias=None
    def __mul__(self,other):
        return np.dot(other,self.matrix) #multiply the matrices together
    def getShape(self): #return the shape of the matrix
        return self.matrix.shape
    def setBias(self,bias):
        self.bias=bias
"""
The network that combines all the layers together
@param: num_out is how many nodes in the output layer
"""
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
            layer=Layer(num[0],nodes,vals=val)
            layer.setBias(bias)
            self.network[-1]=layer #correct output of matrices before
            layer=Layer(nodes,self.num_out,vals=vals) #generate layer with correct matrices
        self.network.append(layer) #add the layer to the network
    def add_bias(self,vals=None):
        assert len(self.network)>0, "Network is empty. Add layers"
        size=self.network[-1].getShape() #get the end sizing to add on
        if type(vals)==type(None):
            vals=normal(size=(size,1))
        self.network[-1].setBias(vals) #set the bias in the current end layer
    def forward(self,inp):
        assert len(self.network)>0, "Network is empty. Add layers"
        x=inp * self.network[0].matrix
        
        sub=self.network
        for layer in sub:
            x=np.dot(x,layer.matrix) #perform multiplication
            if type(layer.bias)!=type(None):
                x += layer.bias #add the biases
        #a=[np.sum(i) for i in x.transpose()] #get output
        return x
    def show(self):
        #show all the network layers and biases
        for i in range(len(self.network)):
            print("Layer",i+1,", nodes:",self.network[i].getShape(),", biases:",self.network[i].bias)
    def train(self,inputData,y_data,epochs,activation,learning_rate):
        #update all the weights via the MSE
        sumError=0
        for i in range(epochs):
            correct=0
            error_updated=0
            for j in range(len(inputData)):
                #GET CURRENT DATA
                input_data=inputData[j]
                target=y_data[j]
                #get prediction
                preds=activation(self.forward(input_data))
                if (preds==target)[0]:
                    correct+=1
                error=preds-target
                sumError+=np.sum(error)
                # Calculate the slope: slope
                slope = 2 * input_data * error
                # Update the weights: weights_updated
                for z in range(len(self.network)):
                    self.network[z].matrix = self.network[z].matrix - learning_rate * slope
                    # Get updated predictions: preds_updated
                    preds_updated = activation(self.forward(input_data))
                    # Calculate updated error: error_updated
                    error_updated += np.sum(preds_updated - target)
            print("epoch",i+1,"Loss:",error_updated,"Accuracy:",(correct/len(y_data))*100,"%")
def MSE(y,y_pred):
    s=y - y_pred
    d=s**2
    mse = np.mean(d)
    return mse


