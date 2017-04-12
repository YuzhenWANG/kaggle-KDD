# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Objective:
Visualize the average travel time for each 20-minute time window.

"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn

info = pd.read_csv("route_avg_time/route_C_3.csv",delimiter=',')
avg_time = info["avg_time"].values
plt.figure()
plt.plot(avg_time[:])
plt.show()

weather = pd.read_csv()