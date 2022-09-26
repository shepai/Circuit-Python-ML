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
        self.a = 1/(1 + np.exp(-self.z))
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
    """
    Adds a layer to the network
    @param: nodes
    @param: vals
    @param: act
    """
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
    """
    adds bias to the network
    @param: vals
    """
    def add_bias(self,vals=None):
        assert len(self.network)>0, "Network is empty. Add layers"
        size=self.network[-1].getShape() #get the end sizing to add on
        if type(vals)==type(None):
            vals=normal(size=(size,1))
        self.network[-1].setBias(vals) #set the bias in the current end layer
    """
    
    @param: inp
    @return: x
    """
    def forward(self,inp):
        #input layer
        assert len(self.network)>0, "Network is empty. Add layers"
        x=inp * self.network[0].matrix
        #x=self.network[0].activation_func(x)
        sub=self.network[1:-1]
        #hidden layers
        for i in range(len(sub)):
            x=np.dot(x,self.network[i+1].matrix) #perform multiplication
            if type(self.network[i+1].bias)!=type(None):
                x += self.network[-1].bias #add the biases
            x=self.network[i+1].activation_func(x)
        #output layer
        x=np.dot(x,self.network[-1].matrix) #perform multiplication
        if type(self.network[-1].bias)!=type(None):
            x += self.network[-1].bias #add the biases
        x=self.network[-1].activation_func(x)
        return x
    """
    show all the network layers and biases
    """
    def show(self):
        #show all the network layers and biases
        for i in range(len(self.network)):
            print("Layer",i+1,", nodes:",self.network[i].getShape(),", biases:",self.network[i].bias)
    """
    @param: delta_w
    @param: delta_b
    @param: lr
    """
    def update_weights_bias(self, delta_w, delta_b, lr):
        #print(self.layers[0].bias.shape)
        print(delta_w[0].shape,delta_w[1].shape,delta_w[2].shape)
        for i in range(len(self.network)):
            layer = self.network[i]
            print(">>>",layer.matrix.shape, (lr*delta_w[i]).shape)
            layer.matrix = layer.matrix - (lr*delta_w[i])
            if type(layer.bias) != type(None):
                layer.bias = layer.bias - (lr*delta_b[i]) 
    """
    @param: X
    @param: y
    @param: targets
    """
    def backpropogate(self, X, y,targets):
        #backpropogation algorithm
        delta = list()
        delta_w = [0 for _ in range(len(self.network))]
        delta_b = [0 for _ in range(len(self.network))]
        error_o = (targets - y)
        for i in reversed(range(len(self.network)-1)):
            error_i = np.dot(self.network[i+1].matrix,error_o) * self.network[i].activation_grad()
            print(i+1,len(self.network),self.network[i].matrix.shape,self.network[i].a)
            delta_w[i+1] = (error_o * np.array([self.network[i].a.transpose()]))/len(y)
            delta_b[i+1] = np.sum(error_o, axis=1)/len(y)
            error_o = error_i
        delta_w[0] = np.dot(error_o,X)
        delta_b[0] = np.sum(error_o, axis=1)/len(y)
        return (delta_w, delta_b)
    """
    @param: inputData
    @param: y_data
    @param: epochs
    @param: learning_rate
    """
    def train(self,inputData,y_data,epochs,learning_rate):
        #update all the weights via the MSE
        sumError=0
        for i in range(epochs):
            correct=0
            error_updated=0
            err=np.zeros(len(inputData))
            for j in range(len(inputData)):
                #GET CURRENT DATA
                input_data=inputData[j]
                target=y_data[j]
                #get prediction
                preds=self.forward(input_data)
                err[j]=preds
                if (preds==target)[0]:
                    correct+=1
                delta_w, delta_b = self.backpropogate(input_data, preds, target)
                self.update_weights_bias(delta_w, delta_b, learning_rate)
                
            print("epoch",i+1,"Loss:",error_updated,"Accuracy:",(correct/len(y_data))*100,"%")




