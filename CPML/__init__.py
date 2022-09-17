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
def normal(mean=0,std=0.5,size=[5]):
    num=1
    for i in size:
        num*=i
    ar=np.zeros(size)
    ar=ar.flatten() #generate numpy
    secondary=np.zeros(num*10)
    for i in range(num*10):
        X=maths.sqrt(abs(-2 * maths.log(i+1/num))) * maths.cos(2*maths.pi * i+1/num)
        #Y=maths.sqrt(abs(-2 * maths.log(i+1/num))) * maths.sin(2*maths.pi * i+1/num)
        X_ = mean + std * X
        #Y_ = mean + std * Y
        secondary[i]=X_
    for i in range(num): #select only from the random variables
        ar[i]=secondary[random.randint(0,num*10 -1)]
    return ar.reshape(size)

"""
get the mean squared error
@param: y the truth data
@param: y_pred the model predictions
"""
def MSE(y,y_pred):
    s=y - y_pred
    d=s**2
    mse = np.mean(d)
    return mse

"""
generate a layer to hold information on network
@param: nodes_in is the number of inputs to this layer
@param: nodes_out is the number of nodes in the next layer
@param: vals is whether the user wishes to manually set the weights
"""
class Layer:
    def __init__(self,nodes_in,nodes_out,vals=None,activ=None):
        if type(vals)==type(None):
            self.matrix=normal(size=(nodes_in,nodes_out)) #generate random weights
        else:
            self.matrix=vals.reshape((nodes_in,nodes_out)) #generate set weights
        self.vals=vals
        self.bias=None
        self.activation_func=activ
        if type(activ)==type(None):
            self.activation_func=self.activation_
    def __mul__(self,other):
        return np.dot(other,self.matrix) #multiply the matrices together
    def getShape(self): #return the shape of the matrix
        return self.matrix.shape
    def setBias(self,bias):
        self.bias=bias
    def activation_(self,inputs):
        #activation functions
        self.z=inputs
        self.a = 1/(1 + np.exp(-self.z))
        assert self.a.shape==inputs.shape,"Shape mismatch"
        return self.a

"""
The network that combines all the layers together
@param: num_out is how many nodes in the output layer
"""
class Network:
    def __init__(self,num_out):
        self.network=[]
        self.num_out=num_out
    def add_layer(self,nodes,vals=None,act=None):
        layer=Layer(nodes,self.num_out,vals=vals,activ=act) #default x by y
        if len(self.network)>0: #there are previous nodes
            layer=self.network[-1]
            bias=self.network[-1].bias
            activation=layer.activation_func
            num=layer.getShape()
            val=layer.vals
            layer=Layer(num[0],nodes,vals=val,activ=activation)
            layer.setBias(bias)
            self.network[-1]=layer #correct output of matrices before
            layer=Layer(nodes,self.num_out,vals=vals,activ=act) #generate layer with correct matrices
        self.network.append(layer) #add the layer to the network
    def add_bias(self,vals=None):
        assert len(self.network)>0, "Network is empty. Add layers"
        size=self.network[-1].getShape() #get the end sizing to add on
        if type(vals)==type(None):
            vals=normal(size=(size,1))
        self.network[-1].setBias(vals) #set the bias in the current end layer
    def forward(self,inp):
        #input layer
        assert len(self.network)>0, "Network is empty. Add layers"
        x=inp * self.network[0].matrix
        sub=self.network[1:-1]
        #hidden layers
        for layer in sub:
            x=np.dot(x,layer.matrix) #perform multiplication
            if type(layer.bias)!=type(None):
                x += layer.bias #add the biases
            x=layer.activation_func(x)
        #output layer
        layer=self.network[-1]
        x=np.dot(x,layer.matrix) #perform multiplication
        if type(layer.bias)!=type(None):
            x += layer.bias #add the biases
        x=layer.activation_func(x)
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
                delta_w, delta_b = self.backpropogate(input_data, y_bat)

            print("epoch",i+1,"Loss:",error_updated,"Accuracy:",(correct/len(y_data))*100,"%")

        def backpropogate(self, X, y):
            delta = list() #Empty list to store derivatives
            delta_w = [0 for _ in range(len(self.network))] #stores weight updates
            delta_b = [0 for _ in range(len(self.network))] #stores bias updates
            error_o = (self.layers[-1].z - y.T) #Calculate the the error at output layer.
            for i in reversed(range(len(self.network) - 1)):
                error_i = activation(self.forward(input_data)) # mutliply error with weights transpose to get gradients
                delta_w[i+1] = error_o.dot(self.network[i].matrix.a.T)/len(y) # store gradient for weights
                delta_b[i+1] = np.sum(error_o, axis=1, keepdims=True)/len(y) # store gradients for biases
                error_o = error_i # now make assign the previous layers error as current error and repeat the process.
            delta_w[0] = error_o.dot(X) # gradients for last layer
            delta_b[0] = np.sum(error_o, axis=1, keepdims=True)/len(y)
            return (delta_w, delta_b)
