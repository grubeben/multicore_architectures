import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

# files
filepath1 = "data/memory_access_k40.txt"
data_memory_access_k40 = np.genfromtxt(filepath1, dtype=float, delimiter=' ')
filepath1a = "data/memory_access_rtx3060.txt"
data_memory_access_rtx3060 = np.genfromtxt(
    filepath1a, dtype=float, delimiter=' ')

filepath2 = "data/dense_matrix_GB_k40.txt"
data_dense_matrix_GB_k40 = np.genfromtxt(filepath2, dtype=float, delimiter=' ')
filepath2a = "data/dense_matrix_GB_rtx3060.txt"
data_dense_matrix_GB_rtx3060 = np.genfromtxt(
    filepath2a, dtype=float, delimiter=' ')

# filepath3 = "data/dense_matrix_GB_k40.txt"
# data_dense= np.genfromtxt(filepath3, dtype=float, delimiter=' ')
# filepath3a = "data/basic_cuda_e.txt"
# data_e= np.genfromtxt(filepath3a, dtype=float, delimiter=' ')


# plot1 - memorry access
k = []
time_kth = []
time_skip = []
for i in range(len(data_memory_access_k40)):
    k.append(data_memory_access_k40[i][0])
    time_kth.append(data_memory_access_k40[i][1])
    time_skip.append(data_memory_access_k40[i][2])

plt.figure(figsize=(10, 5))
plt.plot(k, time_kth, label="bandwidth for a k-stride vector summation")
plt.plot(k, time_skip, label="bandwidth for a vector summation with offset k")
#plt.plot(k, data , label = "time")
#plt.xscale('log', base=10)
#plt.yscale('log', base=2)
plt.xlabel("k")
plt.ylabel("bandwidth [GB/s]")
plt.legend()
plt.title(
    "Stride and offset patterns for vector [N=10^8] summation on the K40-machine")
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

# plot2 - dense matrix
N = []
bw_vanilla = []
bw_vanilla_in_place = []
bw_read_write_optimal = []
bw_in_place = []

for i in range(len(data_dense_matrix_GB_k40)):
    N.append(data_dense_matrix_GB_k40[i][0])
    bw_vanilla.append(data_dense_matrix_GB_k40[i][1])
    bw_vanilla_in_place.append(data_dense_matrix_GB_k40[i][2])
    bw_read_write_optimal.append(data_dense_matrix_GB_k40[i][3])
    bw_in_place.append(data_dense_matrix_GB_k40[i][4])

plt.figure(figsize=(10, 5))
plt.plot(N, bw_vanilla, label="bandwidth for my vanilla transformation", color='g')
plt.plot(N, bw_read_write_optimal,
         label="bandwidth for the read-write-optimal transformation", color='g', linestyle='dotted')
plt.plot(N, bw_vanilla_in_place,
         label="bandwidth for my vanilla in-place-transformation", color='c')
plt.plot(N, bw_in_place, label="bandwidth for read-write-optimal in-place-transformation",
         color='c', linestyle='dotted')
#plt.plot(k, data , label = "time")
#plt.xscale('log', base=10)
#plt.yscale('log', base=2)
plt.xlabel("N")
plt.ylabel("bandwidth [GB/s]")
plt.legend()
plt.title(
    "Performance of different algorithms for matrix transformation on the K40-machine")
plt.grid()
plt.savefig("plots/dense_matrix_k.jpg", bbox_inches='tight')

# # plot2b - dense matrix
# N = []
# bw_vanilla = []
# bw_vanilla_in_place = []
# bw_read_write_optimal = []
# bw_in_place = []

# for i in range(len(data_dense_matrix_GB_rtx3060)):
#     N.append(data_dense_matrix_GB_rtx3060[i][0])
#     bw_vanilla.append(data_dense_matrix_GB_rtx3060[i][1])
#     bw_vanilla_in_place.append(data_dense_matrix_GB_rtx3060[i][2])
#     bw_read_write_optimal.append(data_dense_matrix_GB_rtx3060[i][3])
#     bw_in_place.append(data_dense_matrix_GB_rtx3060[i][4])

# plt.figure(figsize=(10, 5))
# plt.plot(N, bw_vanilla, label="bandwidth for my vanilla transformation", color='g')
# plt.plot(N, bw_read_write_optimal,
#          label="bandwidth for the read-write-optimal transformation", color='g', linestyle='dotted')
# plt.plot(N, bw_vanilla_in_place,
#          label="bandwidth for my vanilla in-place-transformation", color='c')
# plt.plot(N, bw_in_place, label="bandwidth for read-write-optimal in-place-transformation",
#          color='c', linestyle='dotted')
# #plt.plot(k, data , label = "time")
# #plt.xscale('log', base=10)
# #plt.yscale('log', base=2)
# plt.xlabel("N")
# plt.ylabel("bandwidth [GB/s]")
# plt.legend()
# plt.title(
#     "Performance of different algorithms for matrix transformation on the RTX3060-machine")
# plt.grid()
# plt.savefig("plots/dense_matrix_rtx.jpg", bbox_inches='tight')

