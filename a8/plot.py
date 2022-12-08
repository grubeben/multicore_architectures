import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from subprocess import Popen, PIPE
import subprocess
import seaborn as sns



# import sys

# files
filepath1 = "data/dp_rtx.txt"
data_1 = np.genfromtxt(filepath1, dtype=float, delimiter=' ')
filepath2 = "data/dp_k40.txt"
data_2 = np.genfromtxt(filepath2, dtype=float, delimiter=' ')
filepath3 = "data/dp_rtx_cuda.txt"
data_3= np.genfromtxt(filepath3, dtype=float, delimiter=' ')
filepath4 = "data/dp_k40_cuda.txt"
data_4= np.genfromtxt(filepath4, dtype=float, delimiter=' ')


# filepath6 = "data/ex2cg.txt"
# data_6= np.genfromtxt(filepath6, dtype=float, delimiter=' ')


# filepath5 = "data/ex2bonus.txt"
# data_ex5 = np.genfromtxt(filepath5, dtype=float, delimiter=' ')



# plot - ex1 dotp
N = []
rtx_cuda= []
rtx_ocl= []
k40_cuda=[]
k40_ocl=[]

for i in range(len(data_1)):
    N.append(data_1
             [i][0])
    rtx_cuda.append(data_3
             [i][1])
    rtx_ocl.append(data_1
             [i][1])
    k40_cuda.append(data_4
             [i][1])
    k40_ocl.append(data_2
             [i][1])


plt.figure(figsize=(10, 5))

plt.plot(N, rtx_cuda,
         label="RTX CUDA", color='y')
plt.plot(N, rtx_ocl,
         label="RTX OCL", color='y', linestyle='dotted')
plt.plot(N, k40_ocl,
         label="K40 OCL", color='r',linestyle='dotted')
plt.plot(N, k40_cuda,
         label="K40 CUDA", color='r')

plt.xscale('log', base=10)
plt.yscale('log', base=10)
plt.xlabel("vector size N")
plt.ylabel("computation time[ms]")
plt.legend()
plt.title(
    "Dot-Product Performance for OCL and CUDA implementations")
plt.grid()
plt.savefig("plots/dotp.jpg", bbox_inches='tight')




# # plot - ex2k40
# N = []
# rtx_cuda= []
# rtx_ocl= []
# k40_cuda=[]

# for i in range(len(data_4)):
#     N.append(data_4
#              [i][0])
#     rtx_cuda.append(data_4
#              [i][1])
#     rtx_ocl.append(data_4
#              [i][2])
#     k40_cuda.append(data_6
#              [i][1])


# plt.figure(figsize=(10, 5))
# plt.plot(N, rtx_cuda,
#          label="assembly via gpu", color='y', linestyle='dotted', linewidth=10)
# plt.plot(N, rtx_ocl,
#          label="assembly via cpu", color='r')
# plt.plot(N, k40_cuda,
#          label="cg kernel", color='b')
# #plt.plot(k, data , label = "time_custom")
# plt.xscale('log', base=10)
# plt.yscale('log', base=10)
# plt.xlabel("grid size N (=points in one direction)")
# plt.ylabel("computation time[ms]")
# plt.legend()
# plt.title(
#     "Performance of GPU- vs CPU-powered matrix assembly on K40 compared to GPU-CG on RTX3060")
# plt.grid()
# plt.savefig("plots/ex2k40.jpg", bbox_inches='tight')








# data_ex5=np.reshape(data_ex5,(10,10))
# #data_ex5.reshape((10,10))

# ax=sns.heatmap(data_ex5,linewidth=0.5)

# plt.savefig("plots/ex2bonus.jpg", bbox_inches='tight')

# # plot1 - ex1rtx
# N = []
# t_ex= []
# t_ib= []
# t_ic= []

# for i in range(len(data_ex1)):
#     N.append(data_ex1
#              [i][0])
#     t_ex.append(data_ex1
#              [i][1])
#     t_ib.append(data_ex1
#              [i][2])
#     t_ic.append(data_ex1
#              [i][3])


# plt.figure(figsize=(10, 5))
# plt.plot(N, t_ex,
#          label="original exclusive scan", color='y', linestyle='dotted', linewidth=10)
# plt.plot(N, t_ib,
#          label="inclusive kernel as wrap around exclusive scan", color='r')
# plt.plot(N, t_ic,
#          label="inclusive scan by adapting exclusive-kernel", color='b')
# #plt.plot(k, data , label = "time_custom")
# plt.xscale('log', base=10)
# plt.yscale('log', base=10)
# plt.xlabel("N")
# plt.ylabel("computation time[ms]")
# plt.legend()
# plt.title(
#     "Performance of exclusive and inclusive scan implementation on RTX3060")
# plt.grid()
# plt.savefig("plots/ex1rtx.jpg", bbox_inches='tight')

# # plot1 - ex1k40
# N = []
# t_ex= []
# t_ib= []
# t_ic= []

# for i in range(len(data_ex2)):
#     N.append(data_ex2
#              [i][0])
#     t_ex.append(data_ex2
#              [i][1])
#     t_ib.append(data_ex2
#              [i][2])
#     t_ic.append(data_ex2
#              [i][3])


# plt.figure(figsize=(10, 5))
# plt.plot(N, t_ex,
#          label="original exclusive scan", color='y', linestyle='dotted', linewidth=10)
# plt.plot(N, t_ib,
#          label="inclusive kernel as wrap around exclusive scan", color='r')
# plt.plot(N, t_ic,
#          label="inclusive scan by adapting exclusive-kernel", color='b')
# #plt.plot(k, data , label = "time_custom")
# plt.xscale('log', base=10)
# plt.yscale('log', base=10)
# plt.xlabel("N")
# plt.ylabel("computation time[ms]")
# plt.legend()
# plt.title(
#     "Performance of exclusive and inclusive scan implementation on K40")
# plt.grid()
# plt.savefig("plots/ex1k40.jpg", bbox_inches='tight')
