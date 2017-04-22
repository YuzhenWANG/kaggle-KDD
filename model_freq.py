import numpy as np
import pandas as pd

model1=np.loadtxt('volume_files/volume_model1_trainset.csv')
model2=np.loadtxt('volume_files/volume_model2_trainset.csv')
model3=np.loadtxt('volume_files/volume_modelextra_trainset.csv')

relative_weekday = [2,2,3,4,5,6,7,1,2,3,4,2,2,3,3,4,5,6,7,1]


for i in range(20):
    sum_model=sum(model1[i*72:(i+1)*72]+model2[i*72:(i+1)*72]+model3[i*72:(i+1)*72])
    model1_prop= sum(model1[i*72:(i+1)*72])/sum_model
    model2_prop = sum(model2[i * 72:(i + 1) * 72] )/ sum_model
    model3_prop = sum(model3[i * 72:(i + 1) * 72] )/ sum_model
    print i,"st day model1 prop: ", model1_prop
    print "         model2 prop: ", model2_prop
    print "         model3 prop: ", model3_prop
    print

