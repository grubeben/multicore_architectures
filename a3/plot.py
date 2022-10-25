import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

#files
filepath1 = "data/memory_access_k40.txt"
data_memory_access_k40 = np.genfromtxt(filepath1, dtype=float, delimiter=' ')
filepath1a = "data/memory_access_rtx3060.txt"
data_memory_access_rtx3060 = np.genfromtxt(filepath1a, dtype=float, delimiter=' ')
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
for i in range(len(data_memory_access_k40)):
    k.append(data_memory_access_k40[i][0])
    time_kth.append(data_memory_access_k40[i][1])
    time_skip.append(data_memory_access_k40[i][2])

plt.figure(figsize=(10,5))
plt.plot(k, time_kth , label = "bandwidth for a k-stride vector summation")
plt.plot(k, time_skip , label = "bandwidth for a vector summation with offset k")
# plt.plot(k, data , label = "time")
#plt.xscale('log', base=10)
#plt.yscale('log', base=2)
plt.xlabel("k")
plt.ylabel("bandwidth [GB/s]")
plt.legend()
plt.title("Stride and offset patterns for vector [N=10^8] summation on the K40-machine")
plt.grid()
plt.savefig("plots/memory_access_k.jpg", bbox_inches='tight')

# #plot1a
# k=[]
# time_kth=[]
# time_skip=[]
# for i in range(len(data_memory_access_rtx3060)):
#     k.append(data_memory_access_rtx3060[i][0])
#     time_kth.append(data_memory_access_rtx3060[i][1])
#     time_skip.append(data_memory_access_rtx3060[i][2])

# plt.figure(figsize=(10,5))
# plt.plot(k, time_kth , label = "bandwidth for a k-stride vector summation")
# plt.plot(k, time_skip , label = "bandwidth for a vector summation with offset k")
# # plt.plot(k, data , label = "time")
# #plt.xscale('log', base=10)
# #plt.yscale('log', base=2)
# plt.xlabel("k")
# plt.ylabel("bandwidth [GB/s]")
# plt.legend()
# plt.title("Memory access example for vector [N=10^8] summation on the RTX3060-machine")
# plt.grid()
# plt.savefig("plots/memory_access_rtx.jpg", bbox_inches='tight')
