# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Objective:
Visualize the average travel time for each 20-minute time window.

"""

import csv
path = 'input/'
tollgate_1_0 = open("tollgate_avg_volume/tollgate_1_0.csv",'wb')
tollgate_1_1 = open("tollgate_avg_volume/tollgate_1_1.csv",'wb')
tollgate_3_0 = open("tollgate_avg_volume/tollgate_3_0.csv",'wb')
tollgate_3_1 = open("tollgate_avg_volume/tollgate_3_1.csv",'wb')
tollgate_2_0 = open("tollgate_avg_volume/tollgate_2_0.csv",'wb')
tollgate_1_0.write("time_start,time_end,volume"+"\n")
tollgate_1_1.write("time_start,time_end,volume"+"\n")
tollgate_3_0.write("time_start,time_end,volume"+"\n")
tollgate_3_1.write("time_start,time_end,volume"+"\n")
tollgate_2_0.write("time_start,time_end,volume"+"\n")

def split_time_window(str):
    return str[1:20],str[-20:-1]

def write_row(start_time,end_time,volume):
    return start_time+","+end_time+","+volume

f_tollgate = open(path+"training_20min_avg_volume.csv",'rb')
reader = csv.reader(f_tollgate,delimiter=',')
for row in reader:
    if row[0]=="1":
        if row[2]=="0":
            tollgate_1_0.write(write_row(split_time_window(row[1])[0],split_time_window(row[1])[1],row[3])+"\n")
        else:
            tollgate_1_1.write(write_row(split_time_window(row[1])[0], split_time_window(row[1])[1], row[3])+"\n")
    elif row[0]=="3":
        if row[2] == "0":
            tollgate_3_0.write(write_row(split_time_window(row[1])[0], split_time_window(row[1])[1], row[3])+"\n")
        else:
            tollgate_3_1.write(write_row(split_time_window(row[1])[0], split_time_window(row[1])[1], row[3])+"\n")
    else:
        tollgate_2_0.write(write_row(split_time_window(row[1])[0], split_time_window(row[1])[1], row[3])+"\n")

f_tollgate.close()
tollgate_1_0.close()
tollgate_1_1.close()
tollgate_2_0.close()
tollgate_3_0.close()
tollgate_3_1.close()

