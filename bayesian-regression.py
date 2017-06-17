import math
import numpy as np
import matplotlib.pyplot as plt

beta = 25
alpha = 2.0
data_size = 20
function_size = 5
dataX = np.zeros(data_size)
dataY = np.zeros(data_size)

for i in range(0, data_size):
    dataX[i] = (-1*math.pi + i*2*math.pi/data_size)
    dataY[i] = (math.sin(-1*math.pi + i*2*math.pi/data_size) + np.random.normal(0,0.2,1)) 
plt.plot(dataX, dataY, 'ro')
plt.axis([-math.pi, math.pi, -2.5, 2.5])


PHY = np.empty((data_size, function_size))

for i in range(0, data_size):
    for j in range(0, function_size):
        PHY[i][j] = dataX[i]**j

PHYT = PHY.transpose()
Sinv = alpha* np.identity(function_size)+ beta*np.dot(PHYT,PHY)
S = np.linalg.inv(Sinv)
m = beta*np.dot(S,np.dot(PHYT,np.array(dataY)))


sample_number = 30
plot_density = 200
coefficient_sample = np.zeros(function_size)
to_plotX = np.zeros(plot_density)
to_plotY = np.zeros(plot_density)
for i in range(0, sample_number):
    coefficient_sample = np.random.multivariate_normal(np.array(m), S, 1)
    for j in range(0,plot_density):
        to_plotX[j] = (-1*math.pi + j*2*math.pi/plot_density)
        to_plotY[j] = 0
        for k in range(0, function_size):
            to_plotY[j] = to_plotY[j] + coefficient_sample[0][k] * to_plotX[j]**k
    plt.plot(to_plotX, to_plotY)

""" MAXIMUM LIKELIHOOD ESTIMATION
mML = np.dot(np.linalg.inv(np.dot(PHYT,PHY)),np.dot(PHYT,dataY))
print(mML)
for j in range(0,plot_density):
 to_plotX[j] = (-1*math.pi + j*2*math.pi/plot_density)
 to_plotY[j] = 0
 for k in range(0, function_size):
  to_plotY[j] = to_plotY[j] + mML[k] * to_plotX[j]**k
plt.plot(to_plotX, to_plotY)"""

plt.show()