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
            self.matrix=normal(size=(nodes_out,nodes_in)) #generate random weights
        else:
            self.matrix=vals.reshape((nodes_out,nodes_in)) #generate set weights
        self.vals=vals
        self.bias=None
        self.activation_func=activ
        if type(activ)==type(None):
            self.activation_func=self.activation_
        self.a = 0 # defines the output of the layer after running through activation
        self.z = 0 # defines the input of layer to the activation function
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
    def T(self):
        return self.matrix.transpose()

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
            layer=Layer(num[1],nodes,vals=val,activ=activation)
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
        x=inp
        #self.network[0].a=x.copy()
        for i in range(len(self.network)):
            x=np.dot(self.network[i].matrix,x)
            if type(self.network[i].bias)!=type(None):
                x+=self.network[i].bias
            x=self.network[i].activation_func(x)
            self.network[i].a=x.copy()
        #print(x.shape)
        #x = np.sum(x,axis=1)
        return x
    """
    show all the network layers and biases
    """
    def show(self):
        #show all the network layers and biases
        for i in range(len(self.network)):
            print("Layer",i+1,", nodes:",self.network[i].getShape(),", biases:",self.network[i].bias)

    def train(self,inputData,y_data,epochs,learning_rate):
        #update all the weights via the MSE
        sumError=0

        x,y=inputData.shape
        X_data=inputData.reshape((y,x))
        for i in range(epochs):
            preds=np.zeros(y_data.shape)
            correct=0
            #calculate loss
            preds=self.forward(X_data) #get forward pass
            loss = (preds-y_data.transpose())**2 #get loss
            loss= np.sum(np.sum(loss, axis=1)) #calculate overall loss
            print(loss)
            #calculate gradients
            grad_h = 2.*(preds-y_data.transpose())
            """
            print(self.network[2].matrix.shape,grad_h.shape,self.network[1].a.transpose().shape)
            grad_W3=np.dot(grad_h,self.network[1].a.transpose())
            print(grad_W3.shape,self.network[2].matrix.shape)
            self.network[2].matrix-=1e-4 * grad_W3
            """
            #print("epoch",i+1,"Loss:",loss,"Accuracy:",(correct/len(y_data))*100,"%")
            for i in reversed(range(len(self.network))):
                grad_W=np.dot(grad_h,self.network[i-1].a.transpose())
                grad_h=np.dot(self.network[i].matrix.transpose(),grad_h)
                assert grad_W.shape==self.network[i].matrix.shape, "matrix incorrect got "+str(grad_W.shape)+"but expected "+str(self.network[2].matrix.shape)
                self.network[i].matrix-=1e-4 * grad_W
