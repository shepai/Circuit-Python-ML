from CPML import *
import ulab.numpy as np

#create data
data=normal(size=(100,5))
labels=normal(size=(100,2))

#train model
model=regression()
model.fit(data,labels,learning_rate=0.01)

y_pred=model.predict(data)

#calculate how far off predictions were
loss=labels-y_pred
print(loss)