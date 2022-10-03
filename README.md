# Circuit-Python-ML
This project is to provide the tools for ML/AI on a circuit python device. It makes use of the ulab numpy function, but fills in some of the lacking functionality. We have worked on a library that allows creation of neural network architecture for circuitpython devices such as the Raspberry Pi Pico.

circuitpython devices are low cost, low power and small in physical size. They are great for small robotics and hardware projects. We have been working on a neural network library that makes AI easy to implement, with a low storage requirement and provide many examples of projects.

# Table of contents
1. [The library](#lib)
  1. [Gaussian randomized matrices](#Gaussian)
  2. [Creating networks](#Creating)
  3. [Activation functions](#Activation)
2. [Training](#Training)
    1. [Backprop](#Backprop)
    2. [Genetic algorithm](#Genetic)



## The Library <a name="lib"></a>
The library is called in using the imports:

```
from CPML import *
```

### Gaussian randomized matrices <a name="Gaussian"></a>
We can then proceed to use its features such as a normal distribution creation of an array. In standard Python this is done with the following:
```
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
When creating an ANN we use create it as an object and add the number of output nodes that the network will have.

```
from CPML import *

myNet=Network(2) #this network has 2 outputs
```
Following on from that, we can add layers and biases, with the specified number of nodes in each.

```
myNet.add_layer(5) #5 inputs
myNet.add_bias()
myNet.add_layer(2)
myNet.add_layer(4)
myNet.add_bias()
myNet.show() #display the network
```
The network will automatically generate the weights and biases unless we specify otherwise using the 'val' parameter. This parameter takes in a numpy array of the specified size. It does not need to be the same matrix shape, but does need to contain the same amount of elements.

```
myNet=Network(2) #this network has 2 outputs
myNet.add_layer(2,val=np.array([1,0,1,0])) #4 weights as the inputs joining to the next layer
myNet.add_bias(val=np.array([-1,-1]))
```
The best way to understand this is checking out the examples folder. Here you can find logic gate networks for AND and OR which shows you the network predicting the correct numbers.

### Activation functions <a name="Activation"></a>

The layers have an inbuilt default activation function, however you can specify your own as a parameter. For the AND gate example we can ignore two of the weights using .transpose()[0] and gather the inputs from the input nodes. If the sum of both these values is >=0 then the neuron is activated, otherwise the neuron is not. The 1 and 0 represent the true and false within the output of the logic gate.

```
def activation(nump):
    #activation function for the and gate
    s=np.sum(nump.transpose()[0])
    if s>=0:
        return np.array([1])
    return np.array([0])
```
This activation function can be set into the layer. If you want to use the same activation function for all your layers you will have to specify it in each add_layer parameter.
```
#create the network
net=Network(2)
net.add_layer(2,vals=np.array([1,0,1,0]),act=activation)
net.add_bias(vals=np.array([-1,-1]))
net.show()
```
An activation function must take in a single parameter that is an output matrix from the layer it is being entered in.

## Training <a name="Training"></a>
It is better recommended to train off of the device and transfer the weights and biases over. However, with smaller networks circuitpython is capable of performing backprop.

### Backprop <a name="Backprop"></a>
Using the training example, you can generate inputs and outputs, and train a network to learn them.
```
from CPML import *
import ulab.numpy as np

#create neural network
net=Network(2) #two outputs
net.add_layer(2) #two inputs
net.add_layer(5) #hidden layer of 5
net.show()

#set up logic gate data and expected outcome
X_data=np.array([[0,0],[0,1],[1,0],[1,1]]) #training data
y=np.array([[0,0],[1,1],[1,1],[0,0]]) #expected labels

#run training loop    
net.train(X_data,y,1000,0.05) #train
```

The train function has parameters of your input data (n,I) and labels (n,O), where n is the number of items, I is the number of inputs and O is the number of outputs. We then have epochs, which in the example above is 1000. This is how many iterations the training loop will execute. The final parameter is the learning rate.

### Genetic algorithm <a name="Genetic"></a>
You can also train using a genetic algorithm. Backprop algorithms can be computationally challenging for such small devices.

```
network.trainGA(inputData,y_data,epochs,learning_rate,fitnessFunc=None):
```

Within examples you can find some code that trains a neural network to predict classes using a genetic algorithm. The fitness is determined by the amount of correct predictions. One example of this is the following snippet where the y data, and predicted labels are counted.

```
def fitness(y,preds): #fitness function
    correct=0
    p=preds.transpose()
    for i in range(len(y)): #calculate how correct the prediction was
        if np.argmax(p[i])==np.argmax(y[i]):
            correct+=1
    return correct/len(y)
```
