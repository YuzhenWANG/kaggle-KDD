import pandas as pd
import numpy as np

from matplotlib import pyplot
import seaborn

volume1=np.loadtxt('volume_files/volume_model1_trainset.csv')
volume2=np.loadtxt('volume_files/volume_model2_trainset.csv')
volume3=np.loadtxt('volume_files/volume_modelextra_trainset.csv')
submission_table = pd.read_csv('input/submission_sample_volume.csv')
prediction=np.zeros((7*72,5),dtype=float)
relative_weekday = [2,2,3,4,5,6,7,1,2,3,4,2,2,3,3,4,5,6,7,1]
predict_weekday=[2,3,4,5,6,7,1]

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

# i day correspond index= [i*72:(i+1)*72]
volume = volume1 + volume2 + volume3

pyplot.figure()
pyplot.plot(volume)
pyplot.show()

for day in predict_weekday:
    indice = all_indices(day,relative_weekday)
    for i in indice:
        prediction[(day-1)*72:day*72,:] += volume[i*72:(i+1)*72,:]
    prediction[(day-1)*72:day*72,:] = prediction[(day-1)*72:day*72,:]/len(indice)


pyplot.figure()
pyplot.plot(prediction)
pyplot.show()

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

pyplot.figure()
pyplot.plot(submission)
pyplot.show()

print submission.shape

submission_table['volume'] = submission

print submission_table[:15]
submission_table.to_csv('volume_submission_relativeday_2.csv',index=False)

