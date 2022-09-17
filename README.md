# Circuit-Python-ML
This project is to provide the tools for ML/AI on a circuit python device. It makes use of the ulab numpy function, but fills in some of the lacking functionality. We have worked on a library that allows creation of neural network architecture for circuitpython devices such as the Raspberry Pi Pico.

circuitpython devices are low cost, low power and small in physical size. They are great for small robotics and hardware projects. We have been working on a neural network library that makes AI easy to implement, with a low storage requirement and provide many examples of projects.

## The Library
The library is called in using the imports:

```
from CPML import *
```

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


## Circuit Python neural networks
