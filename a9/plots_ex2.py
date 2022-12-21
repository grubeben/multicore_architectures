import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from subprocess import Popen, PIPE
import subprocess
import seaborn as sns



# import sys

# files
filepath1 = "data/cg/hip_cg_k40.txt"
data_1 = np.genfromtxt(filepath1, dtype=float, delimiter=' ')
filepath2 = "data/cg/cuda_cg_k40.txt"
data_2 = np.genfromtxt(filepath2, dtype=float, delimiter=' ')

filepath3 = "data/cg/hip_cg_rtx.txt"
data_3= np.genfromtxt(filepath3, dtype=float, delimiter=' ')
filepath4 = "data/cg/cuda_cg_rtx.txt"
data_4= np.genfromtxt(filepath4, dtype=float, delimiter=' ')


# filepath6 = "data/ex2cg.txt"
# data_6= np.genfromtxt(filepath6, dtype=float, delimiter=' ')

# filepath5 = "data/ex2bonus.txt"
# data_ex5 = np.genfromtxt(filepath5, dtype=float, delimiter=' ')



# plot - ex2k40
N = []
t_hip_k40= []
t_cuda_k40= []
t_hip_rtx=[]
t_cuda_rtx= []

for i in range(len(data_1)):
    N.append(data_1
             [i][0])
    t_hip_k40.append(data_1
             [i][1])
    t_cuda_k40.append(data_2
             [i][1])
    t_hip_rtx.append(data_3
             [i][1])
    t_cuda_rtx.append(data_4
             [i][1])


plt.figure(figsize=(10, 5))
plt.plot(N, t_hip_k40,
         label="k40: cg hip implementation", color='y')
plt.plot(N, t_cuda_k40,
         label="k40: cg cuda implementation", color='r')
plt.plot(N, t_hip_rtx,
         label="rtx: cg hip implementation", color='y', linestyle='dotted', linewidth=3)
plt.plot(N, t_cuda_rtx,
         label="rtx: cg cuda implementation", color='r', linestyle='dotted')

plt.xscale('log', base=10)
plt.yscale('log', base=10)
plt.xlabel("number of unknowns")
plt.ylabel("computation time[ms]")
plt.legend()
plt.title(
    "Performance of HIP vs CUDA CG-implementations on K40")
plt.grid()
plt.savefig("plots/cg.jpg", bbox_inches='tight')
