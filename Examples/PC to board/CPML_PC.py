"""
Circuit Python machine learning tool kit for PC

This library combines and provides functionality to copy what you are doing on the device

Library by Dexter R. Shepherd
University of Sussex PhD student

"""

import numpy as np
import random
import math as maths
import torch

"""
generate a layer to hold information on network
@param: nodes_in is the number of inputs to this layer
@param: nodes_out is the number of nodes in the next layer
@param: vals is whether the user wishes to manually set the weights
"""
class Layer:
    def __init__(self,nodes_in,nodes_out,vals=None,activ=None):
        if type(vals)==type(None):
            self.matrix=np.random.normal(size=(nodes_in,nodes_out)) #generate random weights
        else:
            self.matrix=vals.reshape((nodes_in,nodes_out)) #generate set weights
        self.vals=vals
        self.bias=None
        self.activation_func=activ
        if type(activ)==type(None):
            self.activation_func=self.activation_
        self.a = 0 # defines the output of the layer after running through activation
        self.z = 0 # defines the input of layer to the activation function
    def __mul__(self,other):
        return np.dot(other,self.matrix) #multiply the matrices together
    def getShape(self): #return the shape of the matrix
        return self.matrix.shape
    def setBias(self,bias):
        self.bias=bias
    def activation_(self,inputs):
        #activation functions
        self.z=inputs
        self.a = 1/(1 + torch.exp(-self.z))
        return self.a
    def activation_grad(self):
        return self.a * (1 - self.a)   
        
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
        #x=self.network[0].activation_func(x)
        sub=self.network[1:-1]
        #hidden layers
        for i in range(len(sub)):
            x=torch.mm(x,self.network[i+1].matrix) #perform multiplication
            if type(self.network[i+1].bias)!=type(None):
                x += self.network[-1].bias #add the biases
            x=self.network[i+1].activation_func(x)
        #output layer
        x=torch.mm(x,self.network[-1].matrix) #perform multiplication
        if type(self.network[-1].bias)!=type(None):
            x += self.network[-1].bias #add the biases
        x=self.network[-1].activation_func(x)
        return x
    def show(self):
        #show all the network layers and biases
        for i in range(len(self.network)):
            print("Layer",i+1,", nodes:",self.network[i].getShape(),", biases:",self.network[i].bias)
    def get_weights(self):
        indicies=[0]
        s=0
        for i,layer in enumerate(self.network): #perform calculations
            indicies.append(len(layer.matrix.flatten())+indicies[-1]) #add each index
            indicies.append(len(layer.bias.flatten())+indicies[-1]) #add each index
            s+=len(layer.matrix.flatten())+layer.bias.flatten()
        wb=np.zeros((s))
        for i,layer in enumerate(self.network):
            #stretch out the array
            wb[ind[i]:ind[i+1]]=layer.matrix.flatten()
            wb[ind[i+1]:ind[i+2]]=layer.bias.flatten()
        return wb, indicies
    def reform_weights(self,wb,indicies):
        for i in range(0,indicies,3):
            #stretch out the array
            wb[ind[i]:ind[i+1]]=layer.matrix.flatten()
            wb[ind[i+1]:ind[i+2]]=layer.bias.flatten()
    def parameters(self):
        n=[]
        for i,layer in enumerate(self.network): #perform calculations
            n.append(self.network[i].matrix)
            if self.network[i].bias!=None:
                n.append(self.network[i].bias)
        return n



