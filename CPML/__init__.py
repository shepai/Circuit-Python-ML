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
    for i in range(0,num*10 -1,2):
        X=maths.sqrt(abs(-2 * maths.log(i+1/num))) * maths.cos(2*maths.pi * i+1/num)
        #Y=maths.sqrt(abs(-2 * maths.log(i+1/num))) * maths.sin(2*maths.pi * i+1/num)
        X_ = mean + std * X
        X_n = mean - std * X
        #Y_ = mean + std * Y
        secondary[i]=X_
        secondary[i+1]=X_n
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
    def setWeight(self,val):
        val=np.array(val)
        val=val.reshape(self.getShape())
        self.matrix=val

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

        return x
    """
    show all the network layers and biases
    """
    def show(self):
        #show all the network layers and biases
        for i in range(len(self.network)):
            print("Layer",i+1,", nodes:",self.network[i].getShape(),", biases:",self.network[i].bias)
    """
    @param: inputData
    @param: y_data
    @param: epochs
    @param: learning_rate
    """
    def train(self,inputData,y_data,epochs,learning_rate):
        #update all the weights via the MSE
        correct=0
        x,y=inputData.shape
        X_data=inputData.reshape((y,x))
        for epoch in range(epochs):
            correct=0
            #calculate loss
            preds=self.forward(X_data) #get forward pass
            loss = (preds-y_data.transpose())**2 #get loss
            loss= np.sum(np.sum(loss, axis=0)) #calculate overall loss

            #calculate gradients
            grad_h = 2.*(preds-y_data.transpose())
            for i in reversed(range(len(self.network))):
                grad_W=np.dot(grad_h,self.network[i-1].a.transpose())
                grad_h=np.dot(self.network[i].matrix.transpose(),grad_h)
                assert grad_W.shape==self.network[i].matrix.shape, "matrix incorrect got "+str(grad_W.shape)+"but expected "+str(self.network[2].matrix.shape)
                self.network[i].matrix-=1e-2 * grad_W * learning_rate#learning rate
            #calculate accuracy and display
            p=preds.transpose()
            for i in range(len(y_data)):
                c=0
                for k in range(len(y_data[i])):
                    if round(p[i][k])==round(y_data[i][k]):
                        c+=1
                if c==len(y_data[i]):
                    correct+=1
            print("epoch",epoch+1,"Loss:",loss,"Accuracy:",(correct/len(y_data))*100,"%")
    def trainGA(self,inputData,y_data,epochs,learning_rate,fitnessFunc=None):
        x,y=inputData.shape
        X_data=inputData.reshape((y,x))
        def fitness(y,pred): #default fitness function
            correct=0
            p=preds.transpose()
            for i in range(len(y)): #calculate how correct the prediction was
                c=0
                for k in range(len(y[i])):
                    if round(p[i][k])==round(y[i][k]):
                        c+=1
                if c==len(y[i]):
                    correct+=1
            return correct/len(y)
        def mutate(matrix,rate): #mutate the matrix going in
            shap=matrix.shape
            flat1=matrix.flatten()
            new=normal(mean=0,std=1,size=flat1.shape) #add noise
            for i in range(len(flat1)):
                if random.random()<rate: #mutation rate enforcement
                    flat1[i]=new[i]
            return flat1.reshape(shap) #reshape the array
        if fitnessFunc==None:
            fitnessFunc=fitness
        #get initial fitness
        preds=self.forward(X_data)
        curfit=fitnessFunc(y_data,preds)
        for gen in range(epochs):
            safe=[]
            for i in range(len(self.network)): #go through network
                safe.append(self.network[i].matrix.copy()) #save current weights
                self.network[i].matrix+=mutate(self.network[i].matrix.copy(),learning_rate)
            preds=self.forward(X_data)
            fit=fitnessFunc(y_data,preds)
            if fit<=curfit: #if fitness is worse then reset the weights
                #print(np.sum(self.network[i].matrix))
                for i in range(len(self.network)):
                    self.network[i].matrix=safe[i].copy()
                #print(np.sum(self.network[i].matrix))
            else: curfit=fit
            print("Fitness at Gen",gen+1,":",curfit,fit)
    def reform_weights(self,wb,ind):
        back=ind[0]
        biases=[2+(2*i) for i in range(((len(ind)-1)//2))]
        layer=0
        for i in range(1,len(ind)):
            front=ind[i]
            if i in biases: #bias term
                if ind[i]!=-1:
                    self.network[layer].bias=wb[back:front]
                    back=ind[i]
                layer+=1
            else: #weight term
                self.network[layer].setWeight(wb[back:front])
                back=ind[i]


"""
Regression model
"""
class regression:
    def __init__(self):
        self.m = 0  # Initial slope
        self.b = 0  # Initial intercept
    def fit(self,X,y,learning_rate=0.01,epochs=1000):
        for epoch in range(epochs):
            # Compute predictions
            y_pred = self.m * X + self.b

            # Compute the loss (MSE)
            loss = np.mean((y_pred - y) ** 2)

            # Compute gradients
            dm = (2 / len(X)) * np.sum((y_pred - y) * X)
            db = (2 / len(X)) * np.sum(y_pred - y)

            # Update model parameters
            self.m = self.m - learning_rate * dm
            self.b = self.b - learning_rate * db

            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Loss = {loss:.4f}')
    def predit(self,x):
        return self.m * x + self.b
    
def hstack(arr1, arr2):
    return np.concatenate((arr1, arr2), axis=1)

# Manually implement np.identity
def identity(n):
    return np.zeros((n, n)) + np.eye(n)

class Ridge:
    def __init__(self):
        self.theta=0
        self.bias=0
    def fit(self, X, y, alpha=1):
        m, n = X.shape
        ones = np.ones((m, 1))
        X_bias = hstack(ones, X)  # Add a bias term (intercept)

        # Calculate the optimal parameters using the closed-form solution
        A = np.dot(X_bias.transpose(),X_bias)
        I = identity(n + 1)
        I[0, 0] = 0  # Don't regularize the intercept
        self.theta = np.dot(np.dot(np.linalg.inv(np.dot(X_bias.transpose(), X_bias) + alpha * I),X_bias.transpose()), y)
        self.bias=X_bias
        return self.theta
    def predict(self, X):
        X_test_bias = hstack(np.ones((X.shape[0], 1)), X)
        print(X_test_bias.shape,self.theta.reshape((self.theta.shape[0],1)).shape)
        return np.dot(X_test_bias, self.theta)