import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

#files
filepath1 = "data/memory_access.txt"
data_memory_access = np.genfromtxt(filepath1, dtype=float, delimiter=' ')
# filepath1a = "data/basic_cuda_a2.txt"
# data_a2 = np.genfromtxt(filepath1a, dtype=float, delimiter=' ')
# filepath2 = "data/basic_cuda_b2.txt"
# data_b = np.genfromtxt(filepath2, dtype=float, delimiter=' ')
# filepath3 = "data/basic_cuda_c.txt"
# data_c= np.genfromtxt(filepath3, dtype=float, delimiter=' ')
# filepath4 = "data/basic_cuda_e.txt"
# data_e= np.genfromtxt(filepath4, dtype=float, delimiter=' ')
# filepath5 = "data/cuda_dot.txt"
# data_dot= np.genfromtxt(filepath5, dtype=float, delimiter=' ')


#plot1
k=[]
time_kth=[]
time_skip=[]
for i in range(len(data_memory_access)):
    k.append(data_memory_access[i][0])
    time_kth.append(data_memory_access[i][1])
    time_skip.append(data_memory_access[i][2])

plt.figure(figsize=(10,5))
plt.plot(k, time_kth , label = "time to sum up every kth element")
plt.plot(k, time_skip , label = "time to sum up starting with the kth element")
# plt.plot(k, data , label = "time")
#plt.xscale('log', base=10)
#plt.yscale('log', base=2)
plt.xlabel("k")
plt.ylabel("time [s]")
plt.legend()
plt.title("Memory access example for vector[N=10^8] summation")
plt.grid()
plt.savefig("plots/memory_access.jpg", bbox_inches='tight')
