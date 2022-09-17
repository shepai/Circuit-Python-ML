# Circuit-Python-ML
This project is to provide the tools for ML/AI on a circuit python device. It makes use of the ulab numpy function, but fills in some of the lacking functionality. We have worked on a library that allows creation of neural network architecture for circuitpython devices such as the Raspberry Pi Pico.

circuitpython devices are low cost, low power and small in physical size. They are great for small robotics and hardware projects. We have been working on a neural network library that makes AI easy to implement, with a low storage requirement and provide many examples of projects.

## The Library
The library is called in using the imports:

```
from CPML import *
```

### Gaussian randomized matrices
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

### Creating a networks
When creating an ANN we use create it as an object and add the number of output nodes that the network will have.

```
from CPML import *

myNet=Network(2) #this network has 2 outputs
```
Following on from that, we can add layers and biases, with the specified number of nodes in each.

```
myNet.add_layer(5)
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

### Activation functions

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
