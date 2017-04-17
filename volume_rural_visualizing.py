import pandas as pd
from matplotlib import pyplot
import seaborn

info = pd.read_csv('input/training_20min_avg_volume.csv')
print info

info = info[info['tollgate_id']==1]
info = info[info['direction']==0]

pyplot.figure()
pyplot.plot(info.values[:,3])
pyplot.show()