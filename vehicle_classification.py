# do vehicle classification

import numpy as np
import pandas as pd

path = 'input/'
info = pd.read_csv(path+'volume(table 6)_training.csv',delimiter=',')
print info
print info.groupby(by=['vehicle_model','has_etc'],sort=False).count()

