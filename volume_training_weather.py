import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy.stats.stats import pearsonr

warnings.filterwarnings("ignore")

weather=pd.read_csv('input/weather (table 7)_training.csv')
weather=weather.drop(['pressure','sea_pressure','wind_direction','wind_speed','rel_humidity'],axis=1)
#fill in 10.10 data
weather_1010=weather[718:726]

for i in range(0,8):
    weather_1010.iloc[i,0]='2016-10-10'
    weather_1010.iloc[i,1]=3*i
    weather_1010.iloc[i,2]=(weather.iloc[718+3*i,2]+weather.iloc[726+3*i,2])/2
    weather_1010.iloc[i,3] = (weather.iloc[725, 3] + weather.iloc[726, 3]) / 2
weather.iloc[647,0]='2016-09-29'
weather.iloc[647,1]=21
weather=np.vstack((weather[560:648],weather[718:726],weather_1010,weather[726:]) )  #weather[560:648] : 9.19-9.29  weather[718:]     #10.09-10.17
print weather.shape  #20 days' weather info in every 3 hours

#convert the weather information in every 20 mins
weather_min= [[0 for col in range(4)] for row in range(1)]
for i in range(weather.shape[0]):
    k=0
    while(k<9):
        weather_min=np.row_stack((weather_min,weather[i]))
        k+=1
weather_min=weather_min[1:]


# explore the relation between volume and weather
for i in ['1','2','extra']: #model
    plt.figure()
    model=np.loadtxt('volume_files/volume_model'+i+'_trainset.csv',dtype=int)
    for j in range(5):  #tollgate and direction
        model_j=model[:,j] #1d array tollgate 10
        plt.plot(model_j)
        plt.plot(weather_min[:,2])  #temperature
        plt.plot(weather_min[:,3]+5)  #rain
        plt.legend(['volume','temperature','rain'])
        plt.title('model'+i+'   tollgate'+str(j))
        plt.show()


for i in ['1','2','extra']: #model
    model = np.loadtxt('volume_files/volume_model' + i + '_trainset.csv', dtype=int)
    for j in range(5):
        model_j = model[:, j]
        plt.scatter(weather_min[:,2],model_j,c='b')
        plt.scatter(weather_min[:,3],model_j,c='r')
        plt.legend(['temperature', 'rain'])
        plt.title('model'+i+'   tollgate'+str(j))
        plt.show()
        print "correlation between volume and temperature for model ",i,"at tollgate ",j,":  "
        print pearsonr(weather_min[:,2],model_j)

        print "correlation between volume and rain for model ", i, "at tollgate ", j, ":  "
        print pearsonr(weather_min[:,3],model_j)
        print

        # if(pearsonr(weather_min[:,3],model_j)[1]>0.05):
        #     print"**************** when p>0.05 rain *****************"
        #     print "correlation between volume and rain for model ", i, "at tollgate ", j, ":  "
        #     print pearsonr(weather_min[:, 3], model_j)



