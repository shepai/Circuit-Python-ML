"""
Circuit Python machine learning tool kit

This library combines and provides functionality relevant to making neural networks on devices.
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

class Layer:
    def __init__(self,nodes_in,nodes_out):
        self.matrix=
