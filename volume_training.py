# this file is to train and predict volume
# prediction = f(day_in_week) + f(sesonnality) + f(t-1) + f(weather)  + error

import numpy as np
import statsmodels.api as sm
import pandas as pd
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

final_submission = np.zeros((7*72,5),dtype=float)

score_list = []
for fold in range(1,6):
    print '************************************************'
    print '    fold  ' + str(fold)
    print '************************************************'

    # step 0 : read fold info
    # train set days and CV set days   &&    relative week day   &&  weather feature
    print 'step 0: prepare features'
    f = open('volume_files/fold'+str(fold)+'/train_CV_set.txt','r')
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

    relative_weekday = [2,2,3,4,5,6,7,1,2,3,4,2,2,3,3,4,5,6,7,1]
    relative_day_trainset = [relative_weekday[i] for i in trainset_days]
    relative_day_CV_set = [relative_weekday[i] for i in CV_set_days]
    relative_day_testset = [2,3,4,5,6,7,1]
    print relative_day_trainset
    print relative_day_CV_set
    print relative_day_testset

    print 'start training .............................'
    prediction = np.zeros((7*72,5),dtype=float)
    total_set = np.zeros((16*72,5),dtype=int)
    CV_total_set = np.zeros((4*72,5),dtype=int)
    CV_total_set_prediction = np.zeros((4*72,5),dtype=float)

    for model in ['1','2','extra']:
        print 'model : '+ model

        # step 1 : find seasonal part and remove seasonnal part
        print 'step 1: sesonal weekly model'
        trainset = np.loadtxt('volume_files/fold'+str(fold)+'/model_'+model+'_trainset.csv',dtype=int)

        print np.mean(trainset,axis=0)

        total_set = total_set + trainset
        seasonal_part = np.empty((0,72*7))
        for i in range(5):
            res = sm.tsa.seasonal_decompose(trainset[:,i],freq=72)
            seasonal_part = np.vstack((seasonal_part,res.seasonal[:72*7]))
            trainset[:,i] = trainset[:,i] - res.seasonal
        seasonal_part = np.transpose(seasonal_part)

        # step 2 : model for relative week
        print 'step 2: relative week day model'
        alpha_model_value = np.zeros((7,5),dtype=float)
        mean_value = np.mean(trainset,axis=0)

        for i in range(1,8):
            index = all_indices(i,relative_day_trainset)
            selected_trainset = np.empty((0,5))
            for j in index:
                selected_trainset = np.vstack((selected_trainset,trainset[j*72:(j+1)*72,:]))
            alpha_model_value[i-1,:] = (np.mean(selected_trainset,axis=0) / mean_value) -1

        relative_weekday_part = np.ones((16*24*3,5),dtype=float)
        for i in range(5):
            relative_weekday_part[:,i] = relative_weekday_part[:,i]*mean_value[i]
            for j,num in zip(relative_day_trainset,range(len(relative_day_trainset))):
                relative_weekday_part[num*72:(num+1)*72,i] = (1+alpha_model_value[j-1,i]) * relative_weekday_part[num*72:(num+1)*72,i]

        trainset = trainset - relative_weekday_part

        # step 3 : weather influence


        # CV prediction
        print 'predict CV set '
        CV_seasonal_part = seasonal_part[:4*72,:]
        CV_relative_weedday_part = np.ones((4*72,5),dtype=float)
        for i in range(5):
            CV_relative_weedday_part[:,i] = CV_relative_weedday_part[:,i]*mean_value[i]
            for j,num in zip(relative_day_CV_set,range(len(relative_day_CV_set))):
                CV_relative_weedday_part[num*72:(num+1)*72,i] = (1+alpha_model_value[j-1,i]) * CV_relative_weedday_part[num*72:(num+1)*72,i]

        # CV test
        CV_set = np.loadtxt('volume_files/fold'+str(fold)+'/model_'+model+'_CV_set.csv',dtype=int)
        CV_total_set = CV_total_set + CV_set

        CV_set_prediction = CV_seasonal_part + CV_relative_weedday_part
        CV_total_set_prediction = CV_total_set_prediction + CV_set_prediction

        # for i in range(5):
        #     pyplot.figure()
        #     pyplot.plot(CV_set[:,i])
        #     pyplot.plot(CV_set_prediction[:,i])
        #     pyplot.show()

        # prediction for test set
        print 'predict test set '
        testset_seasonal_part = seasonal_part[:7*72,:]
        testset_relative_weedday_part = np.ones((7*72,5),dtype=float)
        for i in range(5):
            testset_relative_weedday_part[:,i] = testset_relative_weedday_part[:,i]*mean_value[i]
            for j,num in zip(relative_day_testset,range(len(relative_day_testset))):
                testset_relative_weedday_part[num*72:(num+1)*72,i] = (1+alpha_model_value[j-1,i]) * testset_relative_weedday_part[num*72:(num+1)*72,i]

        testset_prediction = testset_seasonal_part + testset_relative_weedday_part
        prediction = prediction + testset_prediction

    # visualizing CV set prediction
    CV_total_set_prediction[CV_total_set_prediction<0] = 0
    CV_total_set_prediction = np.round(CV_total_set_prediction)

    # for i in range(5):
    #     pyplot.figure()
    #     pyplot.plot(CV_total_set_prediction[:,i])
    #     pyplot.plot(CV_total_set[:,i])
    #     pyplot.show()

    CV_diff = CV_total_set - CV_total_set_prediction
    CV_diff = np.abs(CV_diff)

    print CV_diff.shape

    # evaluate function
    CV_prediction_up = np.zeros(120,dtype=float)
    CV_denum_up = np.zeros(120,dtype=float)
    for i in range(5):
        for k in range(6):
            for j in range(4):
                CV_prediction_up[i*4*6+k*4+j] = CV_diff[72*j+24+k,i]
                CV_denum_up[i*4*6+k*4+j] = CV_total_set[72*j+24+k,i] + 1

    CV_prediction_down = np.zeros(120,dtype=float)
    CV_denum_down = np.zeros(120,dtype=float)
    for i in range(5):
        for k in range(6):
            for j in range(4):
                CV_prediction_down[i*4*6+k*4+j] = CV_diff[72*j+51+k,i]
                CV_denum_down[i*4*6+k*4+j] = CV_total_set[72*j+51+k,i] + 1
    print 'score ..........................'
    #
    pyplot.figure()
    pyplot.plot(np.hstack((CV_prediction_up,CV_prediction_down)))
    pyplot.plot(np.hstack((CV_denum_up,CV_denum_down)))
    pyplot.show()
    #
    pyplot.figure()
    pyplot.plot(np.hstack((CV_prediction_up,CV_prediction_down))/np.hstack((CV_denum_up,CV_denum_down)))
    pyplot.show()

    CV_results = np.hstack((CV_prediction_up,CV_prediction_down))/np.hstack((CV_denum_up,CV_denum_down))
    CV_results[CV_results>10] = 1

    print np.mean(CV_results)
    score_list.append(np.mean(np.hstack((CV_prediction_up,CV_prediction_down))/np.hstack((CV_denum_up,CV_denum_down))))

    # visualizing final prediction

    prediction[prediction<0] = 0
    prediction = np.round(prediction)
    print prediction.shape

    final_submission = final_submission + prediction

    # pyplot.figure()
    # pyplot.plot(total_set)
    # pyplot.plot(np.transpose(np.vstack((np.arange(16*72,23*72),np.arange(16*72,23*72),np.arange(16*72,23*72),np.arange(16*72,23*72),np.arange(16*72,23*72)))),prediction)
    # pyplot.show()

# CV score predict
print 'final CV score '
print score_list
print np.mean(score_list)

# generating submission files
final_submission = np.round(final_submission/5)+1

submission_up = np.zeros(210,dtype=int)
for i in range(5):
    for k in range(6):
        for j in range(7):
            submission_up[i*7*6+k*7+j] = final_submission[72*j+24+k,i]

submission_down = np.zeros(210,dtype=int)
for i in range(5):
    for k in range(6):
        for j in range(7):
            submission_down[i*7*6+k*7+j] = final_submission[72*j+51+k,i]

submission = np.hstack((submission_up,submission_down))
submission_table = pd.read_csv('input/submission_sample_volume.csv')
print submission.shape

pyplot.figure()
pyplot.plot(submission)
pyplot.show()

submission_table['volume'] = submission
submission_table.to_csv('volume_submission.csv',index=False)

