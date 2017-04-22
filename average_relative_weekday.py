import pandas as pd
import numpy as np

volume1=np.loadtxt('volume_files/volume_model1_trainset.csv')
volume2=np.loadtxt('volume_files/volume_model2_trainset.csv')
volume3=np.loadtxt('volume_files/volume_modelextra_trainset.csv')
submission_table = pd.read_csv('submission_sample_volume.csv')
prediction=np.zeros((7*72,5),dtype=float)
relative_weekday = [2,2,3,4,5,6,7,1,2,3,4,2,2,3,3,4,5,6,7,1]
predict_weekday=[3,4,5,6,7,1,2]

# i day correspond index= [i*72:(i+1)*72]
for volume in [volume1,volume2,volume3]:
    prediction[:72]+=(volume[2*72:(2+1)*72]+volume[9*72:(9+1)*72]+volume[13*72:(13+1)*72]+volume[14*72:(14+1)*72])/4 #day 3
    prediction[72:2*72]+=(volume[3*72:(3+1)*72]+volume[10*72:(10+1)*72]+volume[15*72:(15+1)*72])/3 #day 4
    prediction[2*72:3*72]+=(volume[4*72:(4+1)*72]+volume[16*72:(16+1)*72])/2  #day 5
    prediction[3*72:4*72] += (volume[5 * 72:(5 + 1) * 72] + volume[17 * 72:(17 + 1) * 72]) / 2  # day 6
    prediction[4 * 72:5 * 72] += (volume[6 * 72:(6 + 1) * 72] + volume[18 * 72:(18 + 1) * 72]) / 2  # day 7
    prediction[5 * 72:6 * 72] += (volume[7 * 72:(7 + 1) * 72] + volume[19 * 72:(19 + 1) * 72]) / 2  # day 1
    prediction[6 * 72:7 * 72]+=(volume[0*72:(0+1)*72]+volume[1*72:(1+1)*72]+volume[8*72:(8+1)*72]+volume[11*72:(11+1)*72]+volume[12*72:(12+1)*72])/5  # day2


# generating submission files
submission_up = np.zeros(210, dtype=int)
for i in range(5): # 5 direction
    for k in range(6): # 2 hours = 6 * 20mins
        for j in range(7): # 7 days
            submission_up[i * 7 * 6 + k * 7 + j] = prediction[72 * j + 21 + k, i]


submission_down = np.zeros(210, dtype=int)
for i in range(5):
    for k in range(6):
        for j in range(7):
            submission_down[i * 7 * 6 + k * 7 + j] = prediction[72 * j + 51 + k, i]

submission = np.hstack((submission_up,submission_down))

print submission.shape

submission_table['volume'] = submission

print submission_table[:15]
submission_table.to_csv('volume_submission_relativeday_2.csv',index=False)

