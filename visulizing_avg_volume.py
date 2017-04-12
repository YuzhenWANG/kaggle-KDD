# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Objective:
Visualize the average travel time for each 20-minute time window.

"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn

info = pd.read_csv("tollgate_avg_volume/tollgate_1_1.csv",delimiter=',')
volume = info["volume"].values
plt.figure()
plt.plot(volume)
plt.show()