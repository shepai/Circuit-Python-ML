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
def MSE(y,y_pred):
    s=y - y_pred
    d=s**2
    mse = np.mean(d)
    return mse
"""
generate a layer to hold information on network
@param: nodes_in is the number of inputs to this layer
@param: nodes_out is the number of nodes in the next layer
"""
class LinearLayer:
    def __init__(self,in_size,out_size,bias=None,mean=0,std=0.5):
        self.matrix=normal(mean,std,size=(in_size,out_size))
        self.bias=np.array([0])
        self.name="linear"
        if bias==True:
            self.bias=normal(mean,std,size=(out_size,))
        self.type="layer"
    def __call__(self):
        return self.matrix
    def setLayer(self,matrix): #allow user to manually set a layer
        assert matrix.shape[0]==self.matrix.shape[0] and matrix.shape[1]==self.matrix.shape[1], "Shape mismatch"
        self.matrix=matrix.copy()
    def setBias(self,matrix): #allow user to manually set a layer
        assert matrix.shape[0]==self.bias.shape[0], "Shape mismatch"
        self.bias=matrix.copy()
        
class convNet:
    def __init__(self,in_size,out_size):
        self.type="layer"
        pass

"""
Drop out layer to be added in the self.layers
"""
class dropOutLayer:
    def __init__(self,probability):
        self.type="dropout"
        self.name=self.type
        self.prob=probability
    def __call__(self,x):
        return self.dropout(x)
    def dropout(self,x): #apply the drop out using probability
        shape=x.flatten().shape
        ar=np.array([random.randint(0,100) for i in range(shape[0])])/100
        ar[ar>self.prob]=1 #apply mask
        ar[ar<self.prob]=0
        ar=ar.reshape(x.shape)
        x=x*ar
        return x
        
"""
Sigmoid activation function
"""
class sigmoid:
    def __init__(self):
        self.type="activation"
        self.name="sigmoid"
    def __call__(self,x):
        return 1/(1 + np.exp(-x))
    def act(self,x):
        return self.__call__(x)
    def derivative(self,x):
        return x * (1 - x)
"""
ReLu activation function
"""
class ReLU:
    def __init__(self):
        self.type="activation"
        self.name="relu"
    def __call__(self,x):
        return np.maximum(0,x)
    def act(self,x):
        return self.__call__(x)
    def derivative(self,x):
        return np.where(x >= 0, 1, 0)
#derivative of sigmoid

"""
Main network class, the network that combines all the layers together
"""
class Network:
    def __init__(self):
        self.layers=[]
    def compile(self):
        self.activations = [np.zeros(layer.matrix.shape[1]) if layer.type=="layer" else 0 for j,layer in enumerate(self.layers)]
        self.layerInd=[]
        for i in range(len(self.layers)):
            if self.layers[i].type=="layer": self.layerInd.append(i)
    def __call__(self,x):
        return self.forward(x)
    def forward(self,x): #pass data through network and apply activation functions
        current_i=0
        for i in range(len(self.layers)):
            if self.layers[i].type=="layer": #if layer apply layer
                x=np.dot(x,self.layers[i]())+self.layers[i].bias.transpose()
                self.activations[i]=x
                current_i=i
            elif self.layers[i].type=="activation": #if activation apply activation
                x=self.layers[i].act(x)
                self.activations[current_i]=x.copy()
            elif self.layers[i].type=="dropout":
                x=self.layers[i].dropout(x)
                self.activations[current_i]=x.copy()
        return x
    def train(self,x,y,epochs,learning_rate,n_prints=10): #train the network
        for epoch in range(epochs):
            y_pred=self.forward(x) #foward pass
            error=y-y_pred
            self.backward_propagation(x,error,learning_rate) #update weights
            if epoch%n_prints == 0: #print progress n_print times
                print("Epoch",epoch,"Loss:",MSE(y,y_pred))
    def derivative(self,ind,x):
        if ind-1>0 and self.layers[ind-1].type=="activation":
            return self.layers[ind-1].derivative(x) #use specific
        if ind-2>0 and self.layers[ind-2].type=="activation":
            return self.layers[ind-2].derivative(x) #use specific
        return x * (1 - x) #default
    def backward_propagation(self, x, error, lr, threshold=10):
        for i in range(len(self.layerInd) - 1, -1, -1):  # Loop through layers in reverse order
            # Calculate the error term at each layer using the chain rule
            if self.layerInd[i] == len(self.layers) - 1:  # For the last layer
                layer_delta = error * self.derivative(self.layerInd[i], self.activations[self.layerInd[i]])
            else:  # For other layers
                layer_delta = np.dot(layer_delta, self.layers[self.layerInd[i + 1]].matrix.transpose()) * self.derivative(self.layerInd[i], self.activations[self.layerInd[i]])

            if self.layerInd[i] == 0:  # If it's the input layer
                layer_input = x
            else:  # For hidden layers and output layer
                layer_input = self.activations[self.layerInd[i - 1]]

            gradient_norm = np.linalg.norm(layer_delta)
            if gradient_norm == 0:
                gradient_norm = 0.1  # Prevent division by zero

            # Normalize gradients to prevent extremely large or small values
            scaled_gradients = layer_delta / gradient_norm
            layer_delta = np.clip(scaled_gradients, -threshold, threshold)  # Clip gradients within a certain range

            # Update weights and biases using the calculated gradients and learning rate
            self.layers[self.layerInd[i]].matrix -= np.dot(layer_input.transpose(), layer_delta) * lr
            if self.layers[self.layerInd[i]].bias.shape[0] == np.sum(layer_delta, axis=0).shape[0]:
                self.layers[self.layerInd[i]].bias -= np.sum(layer_delta, axis=0) * lr


        def save(self,pathname): #save as a json to be reconstructed
            dic={'layerInfo':[]}
            for i,layer in enumerate(layers):
                if layer.type=="layer":
                    dic[str(i)+"-"+layer.name]=layer.matrix.tolist()
                elif layer.type=="activation":
                    dic['layerInfo'].append(layer.name)
            #assume they have connected the sd card
            with open(pathname,"w") as file:
                file.write(str(dic))
        def load(self,pathname): #TODO
            pass
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
        #print(X_test_bias.shape,self.theta.reshape((self.theta.shape[0],1)).shape)
        return np.dot(X_test_bias, self.theta)
"""
Auto encoder neural network
"""
    
class AutoEncoder(Network):
    def __init__(self,input_dim,comp_dim):
        l1=LinearLayer(input_dim,comp_dim)
        l2=LinearLayer(comp_dim,input_dim)
        l3=LinearLayer(input_dim,input_dim)
        self.layers=[l1,ReLU(),l2,ReLU(),l3]
        self.compile()
    def train(self,x,epochs,learning_rate,n_prints=10):
        for epoch in range(epochs):
            y_pred=self.forward(x) #foward pass
            error=x-y_pred
            self.backward_propagation(x,error,learning_rate)
            if epoch%n_prints == 0:
                print("Epoch",epoch,"Loss:",MSE(x,y_pred))
    def getEncoder(self):
        return self.layers[0:-3]
