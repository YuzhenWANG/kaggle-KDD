# this file is to train and predict volume
# prediction = f(day_in_week) + f(sesonnality) + f(t-1) + f(weather)  + error

import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot
import seaborn

# pre-define function
def all_indices(value, qlist):
    indices = []
    idx = -1
    while True:
        try:
            idx = qlist.index(value, idx+1)
            indices.append(idx)
        except ValueError:
            break
    return indices

# step 0 : read fold info
# train set days and CV set days   &&    relative week day   &&  weather feature
print 'step 0: prepare features'
f = open('volume_files/fold1/train_CV_set.txt','r')
line = f.readline()
f.close()

line = line.replace('[','')
line = line.replace(']',' ')
line = line.split(' ')
line = filter(None,line)
trainset_days = []
CV_set_days = []

for i in range(16):
    trainset_days.append(int(line[i]))
for i in range(16,20):
    CV_set_days.append(int(line[i]))
print trainset_days
print CV_set_days

relative_weekday = [2,2,3,4,5,6,7,1,2,3,4,2,2,3,3,4,5,6,7,1]
relative_day_trainset = [relative_weekday[i] for i in trainset_days]
relative_day_CV_set = [relative_weekday[i] for i in CV_set_days]
relative_day_testset = [2,3,4,5,6,7,1]
print relative_day_trainset
print relative_day_CV_set
print relative_day_testset

# step 1 : find seasonal part and remove seasonnal part
print 'step 1: sesonal weekly model'
trainset = np.loadtxt('volume_files/fold1/model_1_trainset.csv',dtype=int)
seasonal_part = np.empty((0,72*7))
for i in range(5):
    res = sm.tsa.seasonal_decompose(trainset[:,i],freq=72)
    seasonal_part = np.vstack((seasonal_part,res.seasonal[:72*7]))
    trainset[:,i] = trainset[:,i] - res.seasonal
print seasonal_part.shape
print trainset.shape

# step 2 : model for relative week
print 'step 2: relative week day model'
alpha_model_value = np.zeros((7,5),dtype=float)
mean_value = np.mean(trainset,axis=0)
print mean_value

for i in range(1,8):
    index = all_indices(i,relative_day_trainset)
    print index
    selected_trainset = np.empty((0,5))
    for j in index:
        selected_trainset = np.vstack((selected_trainset,trainset[j*72:(j+1)*72,:]))
    alpha_model_value[i-1,:] = (np.mean(selected_trainset,axis=0) / mean_value) -1
print alpha_model_value

relative_weekday_part = np.ones((16*24*3,5),dtype=float)
for i in range(5):
    relative_weekday_part[:,i] = relative_weekday_part[:,i]*mean_value[i]
    for j,num in zip(relative_day_trainset,range(len(relative_day_trainset))):
        relative_weekday_part[num*72:(num+1)*72,i] = (1+alpha_model_value[j-1,i]) * relative_weekday_part[num*72:(num+1)*72,i]
print relative_weekday_part.shape

trainset = trainset - relative_weekday_part

pyplot.figure()
pyplot.plot(trainset)
pyplot.show()