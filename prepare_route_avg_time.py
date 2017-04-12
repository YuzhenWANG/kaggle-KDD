# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Objective:
Visualize the average travel time for each 20-minute time window.

"""

import csv
path = 'input/'
route_A_2 = open("route_avg_time/route_A_2.csv",'wb')
route_A_3 = open("route_avg_time/route_A_3.csv",'wb')
route_B_1 = open("route_avg_time/route_B_1.csv",'wb')
route_B_3 = open("route_avg_time/route_B_3.csv",'wb')
route_C_1 = open("route_avg_time/route_C_1.csv",'wb')
route_C_3 = open("route_avg_time/route_C_3.csv",'wb')

route_A_2.write("time_start,time_end,avg_time"+"\n")
route_A_3.write("time_start,time_end,avg_time"+"\n")
route_B_1.write("time_start,time_end,avg_time"+"\n")
route_B_3.write("time_start,time_end,avg_time"+"\n")
route_C_1.write("time_start,time_end,avg_time"+"\n")
route_C_3.write("time_start,time_end,avg_time"+"\n")

def split_time_window(str):
    return str[1:20],str[-20:-1]

def write_row(start_time,end_time,avg_time):
    return start_time+","+end_time+","+avg_time

f_route = open(path+"training_20min_avg_travel_time.csv",'rb')
reader = csv.reader(f_route,delimiter=',')
for row in reader:
    if row[0]=="A":
        if row[1]=="2":
            route_A_2.write(write_row(split_time_window(row[2])[0],split_time_window(row[2])[1],row[3])+"\n")
        else:
            route_A_3.write(write_row(split_time_window(row[2])[0], split_time_window(row[2])[1], row[3])+"\n")
    elif row[0]=="B":
        if row[1] == "1":
            route_B_1.write(write_row(split_time_window(row[2])[0], split_time_window(row[2])[1], row[3])+"\n")
        else:
            route_B_3.write(write_row(split_time_window(row[2])[0], split_time_window(row[2])[1], row[3])+"\n")
    else:
        if row[1] == "1":
            route_C_1.write(write_row(split_time_window(row[2])[0], split_time_window(row[2])[1], row[3])+"\n")
        else:
            route_C_3.write(write_row(split_time_window(row[2])[0], split_time_window(row[2])[1], row[3])+"\n")

f_route.close()
route_A_2.close()
route_A_3.close()
route_B_1.close()
route_B_3.close()
route_C_1.close()
route_C_3.close()