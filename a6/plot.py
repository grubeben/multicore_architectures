import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from subprocess import Popen, PIPE
import subprocess
import seaborn as sns

# import sys

# files
filepath1 = "data/ex1rtx.txt"
data_ex1 = np.genfromtxt(filepath1, dtype=float, delimiter=' ')
filepath2 = "data/ex1k40.txt"
data_ex2 = np.genfromtxt(filepath2, dtype=float, delimiter=' ')


filepath5 = "data/ex2bonus.txt"
data_ex5 = np.genfromtxt(filepath5, dtype=float, delimiter=' ')


# filepath3 = "data/dense_matrix_GB_k40.txt"
# data_dense= np.genfromtxt(filepath3, dtype=float, delimiter=' ')



data_ex5=np.reshape(data_ex5,(10,10))
#data_ex5.reshape((10,10))

ax=sns.heatmap(data_ex5,linewidth=0.5)

plt.savefig("plots/ex2bonus.jpg", bbox_inches='tight')







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
