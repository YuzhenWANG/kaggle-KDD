# this file is to train and predict volume
# prediction = f(day_in_week) + f(sesonnality) + f(t-1) + f(weather)  + error

import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot
import seaborn

# step 1 : find seasonal part and remove seasonnal part
trainset = np.loadtxt('volume_files/fold1/model_1_trainset.csv',dtype=int)
seasonal_part = np.empty((0,72*7))
for i in range(5):
    res = sm.tsa.seasonal_decompose(trainset[:,i],freq=72)
    seasonal_part = np.vstack((seasonal_part,res.seasonal[:72*7]))
    trainset[:,i] = trainset[:,i] - res.seasonal
print seasonal_part.shape
print trainset.shape

# step 2 : model for relative week

print ''''''