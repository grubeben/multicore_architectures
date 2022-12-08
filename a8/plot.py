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
filepath5 = "data/dp_cpu.txt"
data_5= np.genfromtxt(filepath5, dtype=float, delimiter=' ')


filepath6 = "data/build_cash_rtx.txt"
data_6= np.genfromtxt(filepath6, dtype=float, delimiter=' ')
filepath7 = "data/build_cash_k40txt"
data_7= np.genfromtxt(filepath7, dtype=float, delimiter=' ')


# plot - ex1 dotp
N = []
rtx_cuda= []
rtx_ocl= []
k40_cuda=[]
k40_ocl=[]
cpu=[]

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
    cpu.append(data_5
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
plt.plot(N, cpu,
         label="CPU (pthread-Intel(R) Core(TM) i7 CPU)", color='b')

plt.xscale('log', base=10)
plt.yscale('log', base=10)
plt.xlabel("vector size N")
plt.ylabel("computation time[ms]")
plt.legend()
plt.title(
    "Dot-Product Performance for OCL and CUDA implementations")
plt.grid()
plt.savefig("plots/dotp.jpg", bbox_inches='tight')

# plot - ex2 dotp
N = []
rtx_cuda= []
rtx_ocl= []
k40_cuda=[]
k40_ocl=[]
cpu=[]

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
    cpu.append(data_5
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
plt.plot(N, cpu,
         label="CPU (pthread-Intel(R) Core(TM) i7 CPU)", color='b')

plt.xscale('log', base=10)
plt.yscale('log', base=10)
plt.xlabel("vector size N")
plt.ylabel("computation time[ms]")
plt.legend()
plt.title(
    "Dot-Product Performance for OCL and CUDA implementations")
plt.grid()
plt.savefig("plots/dotp.jpg", bbox_inches='tight')


