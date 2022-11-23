import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from subprocess import Popen, PIPE
import subprocess
# import sys

# files
filepath1 = "data/ex1rtx.txt"
data_ex1 = np.genfromtxt(filepath1, dtype=float, delimiter=' ')
filepath2 = "data/ex1k40.txt"
data_ex2 = np.genfromtxt(filepath2, dtype=float, delimiter=' ')


# filepath3 = "data/dense_matrix_GB_k40.txt"
# data_dense= np.genfromtxt(filepath3, dtype=float, delimiter=' ')


# plot1 - ex1rtx
N = []
t_ex= []
t_ib= []
t_ic= []

for i in range(len(data_ex1)):
    N.append(data_ex1
             [i][0])
    t_ex.append(data_ex1
             [i][1])
    t_ib.append(data_ex1
             [i][2])
    t_ic.append(data_ex1
             [i][3])


plt.figure(figsize=(10, 5))
plt.plot(N, t_ex,
         label="original exclusive scan", color='y', linestyle='dotted', linewidth=10)
plt.plot(N, t_ib,
         label="inclusive kernel as wrap around exclusive scan", color='r')
plt.plot(N, t_ic,
         label="inclusive scan by adapting exclusive-kernel", color='b')
#plt.plot(k, data , label = "time_custom")
plt.xscale('log', base=10)
plt.yscale('log', base=10)
plt.xlabel("N")
plt.ylabel("computation time[ms]")
plt.legend()
plt.title(
    "Performance of exclusive and inclusive scan implementation on RTX3060")
plt.grid()
plt.savefig("plots/ex1rtx.jpg", bbox_inches='tight')

# plot1 - ex1k40
N = []
t_ex= []
t_ib= []
t_ic= []

for i in range(len(data_ex2)):
    N.append(data_ex2
             [i][0])
    t_ex.append(data_ex2
             [i][1])
    t_ib.append(data_ex2
             [i][2])
    t_ic.append(data_ex2
             [i][3])


plt.figure(figsize=(10, 5))
plt.plot(N, t_ex,
         label="original exclusive scan", color='y', linestyle='dotted', linewidth=10)
plt.plot(N, t_ib,
         label="inclusive kernel as wrap around exclusive scan", color='r')
plt.plot(N, t_ic,
         label="inclusive scan by adapting exclusive-kernel", color='b')
#plt.plot(k, data , label = "time_custom")
plt.xscale('log', base=10)
plt.yscale('log', base=10)
plt.xlabel("N")
plt.ylabel("computation time[ms]")
plt.legend()
plt.title(
    "Performance of exclusive and inclusive scan implementation on K40")
plt.grid()
plt.savefig("plots/ex1k40.jpg", bbox_inches='tight')

# filepath2 = "data/ex2.txt"
# data_ex2 = np.genfromtxt(filepath2, dtype=float, delimiter=' ')

# # plots ex2
# n = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
# K = []
# time_custom = []
# time_cublas = []
# for i in range(len(data_ex2)):
#     K.append(data_ex2[i][1])
#     time_custom.append(data_ex2[i][2])
#     time_cublas.append(data_ex2[i][3])


# # custom plot
# dx = 1*np.ones(len(data_ex2))
# dy = 8*np.ones(len(data_ex2))
# dz = time_custom

# cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
# max_height = np.max(dz)   # get range of colorbars so we can normalize
# min_height = np.min(dz)
# # scale each z to [0,1], and get their rgb values
# rgba = [cmap((k-min_height)/max_height) for k in dz]

# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111, projection="3d")
# ax.bar3d(n, K, np.zeros(len(data_ex2)), dx, dy, dz, color=rgba)

# ax.set_zlabel("time_custom")
# ax.set_ylabel("K")
# ax.set_xlabel("log(N) basis=10")
# #ax.set_xscale('symlog')
# # plt.legend()
# plt.title(
#     "custom implementation for K dot products <x,v_i> for i{0,..,K} and len(x)=len(v_i) = N")
# # plt.grid()
# plt.savefig("plots/ex2_custom.jpg", bbox_inches='tight')



# filepath2 = "data/ex2_cub.txt"
# data_ex2 = np.genfromtxt(filepath2, dtype=float, delimiter=' ')

# # plots ex2
# n = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
# K = []
# time_custom = []
# time_cublas = []
# for i in range(len(data_ex2)):
#     K.append(data_ex2[i][1])
#     time_custom.append(data_ex2[i][2])
#     time_cublas.append(data_ex2[i][3])


# # custom plot
# dx = 1*np.ones(len(data_ex2))
# dy = 8*np.ones(len(data_ex2))
# dz = time_cublas

# cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
# max_height = np.max(dz)   # get range of colorbars so we can normalize
# min_height = np.min(dz)
# # scale each z to [0,1], and get their rgb values
# rgba = [cmap((k-min_height)/max_height) for k in dz]

# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111, projection="3d")
# ax.bar3d(n, K, np.zeros(len(data_ex2)), dx, dy, dz, color=rgba)

# ax.set_zlabel("time_cublas")
# ax.set_ylabel("K")
# ax.set_xlabel("log(N) basis=10")
# #ax.set_xscale('symlog')
# # plt.legend()
# plt.title(
#     "cublas implementation for K dot products <x,v_i> for i{0,..,K} and len(x)=len(v_i) = N on K40")
# # plt.grid()
# plt.savefig("plots/ex2_cublas.jpg", bbox_inches='tight')