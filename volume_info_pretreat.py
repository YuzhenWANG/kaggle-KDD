# rewrite volume info pretreat file
# save all volume info into a good matrix

import numpy as np
import pandas as pd
from datetime import datetime,timedelta
from matplotlib import pyplot
import seaborn
import statsmodels.api as sm

# index = pd.date_range('19/9/2016', periods=(4*7+1)*24*3, freq='20T')
# print index
#
# info = pd.read_csv('input/volume(table 6)_training.csv')
# vehicle_models = info['vehicle_model'].unique()
# print vehicle_models
#
# tollgate_id = info['tollgate_id'].unique()
# volume_info = np.zeros(((4*7+1)*24*3,len(vehicle_models)*5),dtype=int)
# print volume_info.shape
#
# start = '2016-09-19 00:00:00'
# FMT = "%Y-%m-%d %H:%M:%S"
# start = datetime.strptime(start, FMT)
#
# tollgate_dict = {(1,0):0,(1,1):1,(2,0):2,(3,0):3,(3,1):4}
#
# # read line by line and hash table into relative matrix
# fr = open('input/volume(table 6)_training.csv', 'r')
# fr.readline()
# txt = fr.readlines()  # skip the header
# fr.close()
# for str_line in txt:
#     str_line = str_line.replace('"', '').split(',')
#     tollgate_id = str_line[1]
#     direction = str_line[2]
#     vehile_model = str_line[3]
#     pass_time = str_line[0]
#
#     pass_time = datetime.strptime(pass_time, "%Y-%m-%d %H:%M:%S")
#
#     # calculating time delta
#     print str_line
#     delta = pass_time - start
#     day_diff = delta.days
#     hour_diff =  delta.seconds//3600
#     minute_diff = (delta.seconds % 3600) // 60
#     total_diff_minute = day_diff*24*60 + hour_diff*60 + minute_diff
#     row_num = total_diff_minute // 20
#
#     # calculating column num
#     column_num = int(vehile_model)*5 + tollgate_dict[(int(tollgate_id),int(direction))]
#
#     volume_info[row_num,column_num]+=1
# np.savetxt('volume.csv',volume_info,fmt='%d')

f = open('volume.csv','r')
info = np.loadtxt(f,dtype=int)
print info.shape

# i = 14
# for i in range(35):
#     print i
#     pyplot.figure()
#     # pyplot.plot(info[:,i]+info[:,i+10]+info[:,i+5]+info[:,i+15]+info[:,i+20])
#     pyplot.plot(info[:,i])
#     pyplot.show()


############################################################
#  add  relative day feature
# relative_day_in_week = []
# relative_day_in_week.extend([2]*72) # 19
# relative_day_in_week.extend([2]*72) # 20
# relative_day_in_week.extend([3]*72) # 21
# relative_day_in_week.extend([4]*72) # 22
# relative_day_in_week.extend([5]*72) # 23
# relative_day_in_week.extend([6]*72) # 24
# relative_day_in_week.extend([7]*72) # 25
# relative_day_in_week.extend([1]*72) # 26
# relative_day_in_week.extend([2]*72) # 27
# relative_day_in_week.extend([3]*72) # 28
# relative_day_in_week.extend([4]*72) # 29
# # delete 9 days
# relative_day_in_week.extend([0]*72*9)
#
# relative_day_in_week.extend([2]*72) # 9
# relative_day_in_week.extend([2]*72) # 10
# relative_day_in_week.extend([3]*72) # 11
# relative_day_in_week.extend([3]*72) # 12
# relative_day_in_week.extend([4]*72) # 13
# relative_day_in_week.extend([5]*72) # 14
# relative_day_in_week.extend([6]*72) # 15
# relative_day_in_week.extend([7]*72) # 16
# relative_day_in_week.extend([1]*72) # 17
#
# print len(relative_day_in_week)

model_1_volume = info[:,5:10]
model_2_volume = info[:,10:15]
model_extra_volume = info[:,0:5] + info[:,15:20] + info[:,20:25] + info[:,25:30] + info[:,30:35] + info[:,35:]
model_1_volume = np.vstack((model_1_volume[:792,:],model_1_volume[-648:,:]))
model_2_volume = np.vstack((model_2_volume[:792,:],model_2_volume[-648:,:]))
model_extra_volume = np.vstack((model_extra_volume[:792,:],model_extra_volume[-648:,:]))
print model_1_volume.shape
print model_2_volume.shape
print model_extra_volume.shape

# pyplot.figure()
# pyplot.plot(model_extra_volume)
# pyplot.show()

# save seasonal (weekly) feature
seasonal_matrix = np.empty((0,1440))
for i in range(5):
    res = sm.tsa.seasonal_decompose(model_1_volume[:,i],freq=72)
    seasonal_matrix = np.vstack((seasonal_matrix,res.seasonal))
for i in range(5):
    res = sm.tsa.seasonal_decompose(model_2_volume[:,i],freq=72)
    seasonal_matrix = np.vstack((seasonal_matrix,res.seasonal))
for i in range(5):
    res = sm.tsa.seasonal_decompose(model_extra_volume[:, i], freq=72)
    seasonal_matrix = np.vstack((seasonal_matrix, res.seasonal))

print seasonal_matrix
np.savetxt('volumn_seasonal_feature.csv',seasonal_matrix[:,:24*3],fmt='%f')