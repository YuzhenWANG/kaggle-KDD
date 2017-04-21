# rewrite volume info pretreat file
# save all volume info into a good matrix

import numpy as np
import pandas as pd
from datetime import datetime,timedelta
from matplotlib import pyplot
import seaborn
import statsmodels.api as sm
from sklearn.model_selection import KFold

# step 1 : change table 6 rural data into 20 minute time window and calculate them by vehicle model
# input: table 6 output: matrix 2088 * 40 (8 vehicle models and each 5 tollgate)
#########################################################################################################

print 'step1 : change table 6 into matrix'
index = pd.date_range('19/9/2016', periods=(4*7+1)*24*3, freq='20T')
print index

info = pd.read_csv('input/volume(table 6)_training.csv')
vehicle_models = info['vehicle_model'].unique()
print vehicle_models

tollgate_id = info['tollgate_id'].unique()
volume_info = np.zeros(((4*7+1)*24*3,len(vehicle_models)*5),dtype=int)
print volume_info.shape

start = '2016-09-19 00:00:00'
FMT = "%Y-%m-%d %H:%M:%S"
start = datetime.strptime(start, FMT)

tollgate_dict = {(1,0):0,(1,1):1,(2,0):2,(3,0):3,(3,1):4}

# read line by line and hash table into relative matrix
fr = open('input/volume(table 6)_training.csv', 'r')
fr.readline()
txt = fr.readlines()  # skip the header
fr.close()

for str_line in txt:
    str_line = str_line.replace('"', '').split(',')
    tollgate_id = str_line[1]
    direction = str_line[2]
    vehile_model = str_line[3]
    pass_time = str_line[0]

    pass_time = datetime.strptime(pass_time, "%Y-%m-%d %H:%M:%S")

    # calculating time delta
    delta = pass_time - start
    day_diff = delta.days
    hour_diff =  delta.seconds//3600
    minute_diff = (delta.seconds % 3600) // 60
    total_diff_minute = day_diff*24*60 + hour_diff*60 + minute_diff
    row_num = total_diff_minute // 20

    # calculating column num
    column_num = int(vehile_model)*5 + tollgate_dict[(int(tollgate_id),int(direction))]

    volume_info[row_num,column_num]+=1
np.savetxt('volume_files/volume_training_set.csv',volume_info,fmt='%d')
print volume_info.shape

# pyplot.figure()
# pyplot.plot(volume_info)
# pyplot.show()

# step2: delete national day 9 days
# separate total volume into 3 part (model1 , model2 , model extra)
#######################################################################################

print 'step 2 : delete national days '

model_1_volume = volume_info[:,5:10]
model_2_volume = volume_info[:,10:15]
model_extra_volume = volume_info[:,0:5] + volume_info[:,15:20] + volume_info[:,20:25] + volume_info[:,25:30] + volume_info[:,30:35] + volume_info[:,35:]
model_1_volume = np.vstack((model_1_volume[:792,:],model_1_volume[-648:,:]))
model_2_volume = np.vstack((model_2_volume[:792,:],model_2_volume[-648:,:]))
model_extra_volume = np.vstack((model_extra_volume[:792,:],model_extra_volume[-648:,:]))
print model_1_volume.shape
print model_2_volume.shape
print model_extra_volume.shape
np.savetxt('volume_files/volume_model1_trainset.csv',model_1_volume,fmt='%d')
np.savetxt('volume_files/volume_model2_trainset.csv',model_2_volume,fmt='%d')
np.savetxt('volume_files/volume_modelextra_trainset.csv',model_extra_volume,fmt='%d')

for i in range(5):
    pyplot.figure()
    pyplot.plot(model_1_volume[:,i])
    pyplot.plot(model_2_volume[:,i])
    pyplot.plot(model_extra_volume[:,i])
    pyplot.show()

# step3: prepare test set 1
# save testset1 into 3 (7 * 24 *3, 5) matrix
#################################################################################

print 'step 3 prepare test set 1 into 3 matrix '
volume_test_1 = np.zeros((7*24*3,len(vehicle_models)*5),dtype=int)
print volume_test_1.shape

start = '2016-10-18 00:00:00'
FMT = "%Y-%m-%d %H:%M:%S"
start = datetime.strptime(start, FMT)

fr = open('input/volume(table 6)_test1.csv', 'r')
fr.readline()
txt = fr.readlines()  # skip the header
fr.close()
for str_line in txt:
    str_line = str_line.replace('"', '').split(',')
    tollgate_id = str_line[1]
    direction = str_line[2]
    vehile_model = str_line[3]
    pass_time = str_line[0]

    pass_time = datetime.strptime(pass_time, "%Y-%m-%d %H:%M:%S")

    # calculating time delta
    # print str_line
    delta = pass_time - start
    day_diff = delta.days
    hour_diff =  delta.seconds//3600
    minute_diff = (delta.seconds % 3600) // 60
    total_diff_minute = day_diff*24*60 + hour_diff*60 + minute_diff
    row_num = total_diff_minute // 20

    # calculating column num
    column_num = int(vehile_model)*5 + tollgate_dict[(int(tollgate_id),int(direction))]
    volume_test_1[row_num,column_num]+=1

model_1_volume_testset_1 = volume_test_1[:,5:10]
model_2_volume_testset_1 = volume_test_1[:,10:15]
model_extra_volume_testset_1 = volume_test_1[:,0:5] + volume_test_1[:,15:20] + volume_test_1[:,20:25] + volume_test_1[:,25:30] + volume_test_1[:,30:35] + volume_test_1[:,35:]

np.savetxt('volume_files/volume_model1_test_set_1.csv',model_1_volume_testset_1,fmt='%d')
np.savetxt('volume_files/volume_model2_test_set_1.csv',model_2_volume_testset_1,fmt='%d')
np.savetxt('volume_files/volume_modelextra_test_set_1.csv',model_extra_volume_testset_1,fmt='%d')
print model_1_volume_testset_1.shape
print model_2_volume_testset_1.shape
print model_extra_volume_testset_1.shape

# pyplot.figure()
# pyplot.plot(volume_test_1)
# pyplot.show()

# step 4: create n fold fies:
# change 20 days data into 5 fold train and CV set and also test set
# generating train test files
#####################################################################

days = range(20)
relative_weed = [2,2,3,4,5,6,7,1,2,3,4,2,2,3,3,4,5,6,7,1]

kf = KFold(n_splits=5,random_state = 5)
fold = 1
for train_index,CV_index in kf.split(days):
    print 'fold:' +str(fold)
    print train_index,CV_index
    file_path = 'volume_files/fold'+str(fold)+'/'
    fold = fold + 1

    f = open(file_path+'train_CV_set.txt','w')
    f.write(str(train_index))
    f.write(str(CV_index))
    f.close()

    print 'save model 1'
    # save train total set
    train_set_fold = np.empty((0,5),dtype=int)
    for i in train_index:
        train_set_fold = np.vstack((train_set_fold,model_1_volume[72*i:72*(i+1),:]))
    print train_set_fold.shape
    np.savetxt(file_path+'model_1_trainset.csv',train_set_fold,'%d')

    # save CV total set
    CV_set_fold = np.empty((0,5),dtype=int)
    for i in CV_index:
        CV_set_fold = np.vstack((CV_set_fold,model_1_volume[72*i:72*(i+1),:]))
    print CV_set_fold.shape
    np.savetxt(file_path+'model_1_CV_set.csv',CV_set_fold,'%d')

    # save test set
    test_set_fold = model_1_volume_testset_1
    print test_set_fold.shape
    np.savetxt(file_path+'model_1_testset.csv',test_set_fold,'%d')

    print 'save model 2'
    # save train total set
    train_set_fold = np.empty((0,5),dtype=int)
    for i in train_index:
        train_set_fold = np.vstack((train_set_fold,model_2_volume[72*i:72*(i+1),:]))
    print train_set_fold.shape
    np.savetxt(file_path+'model_2_trainset.csv',train_set_fold,'%d')

    # save CV total set
    CV_set_fold = np.empty((0,5),dtype=int)
    for i in CV_index:
        CV_set_fold = np.vstack((CV_set_fold,model_2_volume[72*i:72*(i+1),:]))
    print CV_set_fold.shape
    np.savetxt(file_path+'model_2_CV_set.csv',CV_set_fold,'%d')

    # save test set
    test_set_fold = model_2_volume_testset_1
    print test_set_fold.shape
    np.savetxt(file_path+'model_2_testset.csv',test_set_fold,'%d')

    print 'save model extra'
    # save train total set
    train_set_fold = np.empty((0,5),dtype=int)
    for i in train_index:
        train_set_fold = np.vstack((train_set_fold,model_extra_volume[72*i:72*(i+1),:]))
    print train_set_fold.shape
    np.savetxt(file_path+'model_extra_trainset.csv',train_set_fold,'%d')

    # save CV total set
    CV_set_fold = np.empty((0,5),dtype=int)
    for i in CV_index:
        CV_set_fold = np.vstack((CV_set_fold,model_extra_volume[72*i:72*(i+1),:]))
    print CV_set_fold.shape
    np.savetxt(file_path+'model_extra_CV_set.csv',CV_set_fold,'%d')

    # save test set
    test_set_fold = model_extra_volume_testset_1
    print test_set_fold.shape
    np.savetxt(file_path+'model_extra_testset.csv',test_set_fold,'%d')