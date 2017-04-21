import pandas as pd
import numpy as np

volume1=np.loadtxt('volume_files/volume_model1_trainset.csv')
submission_table = pd.read_csv('submission_sample_volume.csv')
prediction=np.zeros((7*72,5),dtype=float)
relative_weekday = [2,2,3,4,5,6,7,1,2,3,4,2,2,3,3,4,5,6,7,1]
predict_weekday=[2,3,4,5,6,7,1]

print type(volume1)
print volume1.shape


for model in range(5):
    volume=volume1[:,model]
    # day = 3
    for i in range(72):
        prediction[i,model]=(volume[2*72+i]+volume[9*72+i]+volume[13*72+i]+volume[14*72+i])/4


    # day = 4
    for i in range(72):
        prediction[72+i,model]=(volume[3*72+i]+volume[10*72+i]+volume[15*72+i])/3
    # day = 5
    for i in range(72):
        prediction[2*72+i,model]=(volume[4*72+i]+volume[16*72+i])/2
    # day = 6
    for i in range(72):
        prediction[3*72+i,model]=(volume[5*72+i]+volume[17*72+i])/2
    # day = 7
    for i in range(72):
        prediction[4*72+i,model]=(volume[6*72+i]+volume[18*72+i])/2
    # day = 1
    for i in range(72):
        prediction[5*72+i,model]=(volume[7*72+i]+volume[19*72+i])/2
    # day = 2
    for i in range(72):
        prediction[6*72+i,model]=(volume[0*72+i]+volume[1*72+i]+volume[8*72+i]+volume[11*72+i]+volume[12*72+i])/5

print prediction
print prediction.shape


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
submission_table.to_csv('volume_submission_relativeday.csv',index=False)