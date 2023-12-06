#ridge regression model
from CPML import *
import ulab.numpy as np

#set up data
data=normal(0,0.5,(100,5))
label=normal(0,0.5,(100,3))

model=Ridge()

model.fit(data,label)
print("loss:",model.predict(data)-label)