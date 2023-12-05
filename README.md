# Circuit-Python-ML
This project is to provide the tools for ML/AI on a circuit python device. It makes use of the ulab numpy function, but fills in some of the lacking functionality. We have worked on a library that allows creation of neural network architecture for circuitpython devices such as the Raspberry Pi Pico.

circuitpython devices are low cost, low power and small in physical size. They are great for small robotics and hardware projects. We have been working on a neural network library that makes AI easy to implement, with a low storage requirement and provide many examples of projects.

# Table of contents
1. [The library](#lib)
    1. [Gaussian randomized matrices](#Gaussian)
    2. [Creating networks](#Creating)
    3. [Activation functions](#Activation)
    4. [Forward pass](#forward)
2. [Training](#Training)
    1. [Backprop](#Backprop)
    2. [Genetic algorithm](#Genetic)
3. [Regresssion](#reg)
    1. [Linear](#reg1)
    2. [Ridge](#reg2)

## The Library <a name="lib"></a>
The library is called in using the imports:

```
from CPML import *
```

### Gaussian randomized matrices <a name="Gaussian"></a>
We can then proceed to use its features such as a normal distribution creation of an array. In standard Python this is done with the following:
```
#note this will not work in circuitpython
import numpy as np

np.random.normal(mu,std,(5,5))
```

The ulab library that provides numpy does not have this feature. You can call in the normal distribution using the following feature, where mean, std are specified and x and y are your matrix shapes.

```
>>> import CPML as cp
>>> array = cp.normal(0,1,size=(2,2))
>>> array
array([[1.29068, 0.517823],
       [0.987146, 1.1901]], dtype=float32)
```

### Creating a networks <a name="Creating"></a>
When creating an ANN we use create it as an object

```
from CPML import *

myNet=Network() #this network has
```
Following on from that, we can add layers and biases, with the specified number of nodes in each.

```
net=Network()
layer=LinearLayer(10,5,bias=True)

net.layers=[layer, sigmoid()] #add all outputs and activation functions
net.compile() #compile network
```
The network will automatically generate the weights and biases unless we specify otherwise using the setLayer and setBias methods. This parameter takes in a numpy array of the specified size. It needs to be the same matrix shape as you specified

```
myNet=Network() #this network has 2 outputs
layer=LinearLayer(2,2,bias=True)
layer.setLayer(np.array([[1,1],[1,1]]))
layer.setBias(np.array([[-2],[-2]]))
net.layers=[layer]
net.compile()
```
The best way to understand this is checking out the examples folder. Here you can find logic gate networks for AND and OR which shows you the network predicting the correct numbers.

### Activation functions <a name="Activation"></a>

The layers have an inbuilt default activation function, however you can specify your own as a parameter. For the AND gate example we can ignore two of the weights using . We can make out own activation functions as such. For example, if the sum of both these values is >=0 then the neuron is activated, otherwise the neuron is not. The 1 and 0 represent the true and false within the output of the logic gate.

```
class my_activation:
    def __init__(self):
        self.type="activation"
    def __call__(self,x):
        return self.act(x)
    def act(self,nump):
        #activation function for the and gate
        s=np.sum(nump.transpose()[0])
        if s>=0:
            return np.array([1])
        return np.array([0])
```
This activation function can be set into the layer. If you want to use the same activation function for all your layers you will have to specify it in each add_layer parameter.
```
#create the network
net=Network()
layer=LinearLayer(2,2,bias=True)
layer.setLayer(np.array([[1,1],[1,1]]))
net.layers=[layer,my_activation()]
net.compile()
```
An activation function must take in a single parameter that is an output matrix from the layer it is being entered in.

## Forward pass <a name="forward"></a>
Passing data through the network will take in an array of shape (N,m) where N is the size of the data set and m is the number of nodes in the proceeding layer. Our matrix of inputs represented by x is subjected to the dot product of the following layer. Lets say we have 2 input nodes and 3 hidden layer nodes, then a following of two output nodes, the following calculation would be made. The sample size of data is 3 items.

$$
\begin{bmatrix}
x_1 & x_2\\
x_3 & x_4\\
x_5 & x_6
\end{bmatrix} \cdot \begin{bmatrix}
w_1 & w_2 & w_3\\
w_4 & w_5 & w_6
\end{bmatrix} = \begin{bmatrix}
w_{1,1} & w_{1,2} & w_{1,3}\\
w_{1,4} & w_{1,5} & w_{1,6}\\
w_{1,7} & w_{1,8} & w_{1,9}
\end{bmatrix}
$$

Then this output is subjected to an activation function, represented as $\phi$ is proceeded into the next layer.

$$
\phi \begin{pmatrix}\begin{bmatrix}
w_{1,1} & w_{1,2} & w_{1,3}\\
w_{1,4} & w_{1,5} & w_{1,6}\\
w_{1,7} & w_{1,8} & w_{1,9}
\end{bmatrix}
\end{pmatrix}
\cdot
\begin{bmatrix}
w_{2,1} & w_{2,2}\\
w_{2,4} & w_{2,5}\\
w_{2,7} & w_{2,8}
\end{bmatrix} =\begin{bmatrix}
o_{1,1} & o_{1,2}\\
o_{2,1} & o_{2,2}\\
o_{3,1} & o_{3,2}
\end{bmatrix}
$$

Our output layer is of shape (N,o) where N was the initial sample size of data, and o is the number of output nodes.

This therefore gives the option of deciding upon output. You may wish to pick the highest value with the argmax function.
```
>>> X_data=normal(mean=0,std=1,size=(8,2)) #N by input nodes
>>> out=net.forward(X_data.transpose())
>>> out.shape
(2, 8)
>>> out_ind=np.argmax(out,axis=0)
>>> out_ind
array([1, 1, 1, 1, 1, 1, 1, 1], dtype=int16)
```


## Training <a name="Training"></a>
It is better recommended to train off of the device and transfer the weights and biases over. However, with smaller networks circuitpython is capable of performing backprop.

### Backprop <a name="Backprop"></a>
Using the training example, you can generate inputs and outputs, and train a network to learn them.
```
from CPML import *
import ulab.numpy as np

class myNet(Network):
    def __init__(self,i,h,o):
        self.layer1=LinearLayer(i,h)
        self.out=LinearLayer(h,o)
        self.layers=[self.layer1,sigmoid(),self.out]
        self.compile()

net=myNet(5,6,2)
data=normal(size=(100,5))
labels=normal(size=(100,2))
print(net.forward(normal(0,0.5,(10,5))).shape)
net.train(data,labels,100,0.1) #outputs loss over time

```

## Regression models <a name="reg"></a>


### Regression <a name="reg1"></a>

```
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])

reg=regression()
reg.fit(X,y)
# Make predictions
new_X = np.array([6, 7, 8])
predictions=reg.predit(new_X)
print("Predictions:", predictions)
```

### Ridge regression <a name="reg2"></a>

```
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,0,0,1])


X_test = np.array([[1,1],[0,1]])  # New data point to predict

ridge=Ridge()
theta_ridge=ridge.fit(X,y)
prediction=ridge.predict(X_test)

print("Ridge Regression Coefficients:", theta_ridge)
print("Predicted value:", prediction)

```