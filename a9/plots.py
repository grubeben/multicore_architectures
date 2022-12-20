import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from subprocess import Popen, PIPE
import subprocess
import seaborn as sns



# import sys

# files
filepath1 = "data/hip_cg_k40.txt"
data_1 = np.genfromtxt(filepath1, dtype=float, delimiter=' ')
filepath2 = "data/cuda_cg_k40.txt"
data_2 = np.genfromtxt(filepath2, dtype=float, delimiter=' ')

# filepath3 = "data/ex2rtx.txt"
# data_3= np.genfromtxt(filepath3, dtype=float, delimiter=' ')
# filepath4 = "data/ex2k40.txt"
# data_4= np.genfromtxt(filepath4, dtype=float, delimiter=' ')
# filepath6 = "data/ex2cg.txt"
# data_6= np.genfromtxt(filepath6, dtype=float, delimiter=' ')

# filepath5 = "data/ex2bonus.txt"
# data_ex5 = np.genfromtxt(filepath5, dtype=float, delimiter=' ')



# plot - ex2k40
N = []
t_hip= []
t_cuda= []
t_cg=[]

for i in range(len(data_1)):
    N.append(data_1
             [i][0])
    t_hip.append(data_1
             [i][1])
    t_cuda.append(data_2
             [i][1])


plt.figure(figsize=(10, 5))
plt.plot(N, t_hip,
         label="cg hip implementation", color='y')
plt.plot(N, t_cuda,
         label="cg cuda implementation", color='r')

plt.xscale('log', base=10)
plt.yscale('log', base=10)
plt.xlabel("number of unknowns")
plt.ylabel("computation time[ms]")
plt.legend()
plt.title(
    "Performance of HIP vs CUDA CG-implementations on K40")
plt.grid()
plt.savefig("plots/k40.jpg", bbox_inches='tight')
