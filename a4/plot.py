import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm


# files
filepath1 = "data/ex1.txt"
data_ex1 = np.genfromtxt(filepath1, dtype=float, delimiter=' ')


# filepath3 = "data/dense_matrix_GB_k40.txt"
# data_dense= np.genfromtxt(filepath3, dtype=float, delimiter=' ')


# # plot1 - dot product
# N = []
# t_dot_product = []
# t_4dot_product = []
# t_shared_mem = []
# t_ws_fixed = []
# t_ws_elastix = []

# for i in range(len(data_ex1)):
#     N.append(data_ex1
#              [i][0])
#     t_dot_product.append(data_ex1
#                          [i][1])
#     t_shared_mem.append(data_ex1
#                         [i][2])
#     t_ws_fixed.append(data_ex1
#                       [i][3])
#     t_ws_elastix.append(data_ex1
#                         [i][4])
# for i in range(len(t_dot_product)):
#     t_4dot_product.append(4*t_dot_product[i])

# plt.figure(figsize=(10, 5))
# plt.plot(N, t_4dot_product,
#          label="execution of 4 dot-products as reference", color='y')
# plt.plot(N, t_dot_product,
#          label="execution of 1 dot-product as reference", color='y', linestyle='dotted')
# plt.plot(N, t_ws_fixed,
#          label="shared-memory implementation", color='g')
# plt.plot(N, t_shared_mem,
#          label="warf-shuffle implementation with fixed #threads", color='c')
# plt.plot(N, t_ws_elastix, label="warf-shuffle implementation with #threads(N)",
#          color='c', linestyle='dashdot')
# #plt.plot(k, data , label = "time_custom")
# plt.xscale('log', base=10)
# plt.yscale('log', base=10)
# plt.xlabel("vector length N")
# plt.ylabel("computation time_custom [s]")
# plt.legend()
# plt.title(
#     "Performance of different norm-computing-kernels on the RTX3060-machine")
# plt.grid()
# plt.savefig("plots/ex1.jpg", bbox_inches='tight')


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



filepath2 = "data/ex2_cub.txt"
data_ex2 = np.genfromtxt(filepath2, dtype=float, delimiter=' ')

# plots ex2
n = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
K = []
time_custom = []
time_cublas = []
for i in range(len(data_ex2)):
    K.append(data_ex2[i][1])
    time_custom.append(data_ex2[i][2])
    time_cublas.append(data_ex2[i][3])


# custom plot
dx = 1*np.ones(len(data_ex2))
dy = 8*np.ones(len(data_ex2))
dz = time_cublas

cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
max_height = np.max(dz)   # get range of colorbars so we can normalize
min_height = np.min(dz)
# scale each z to [0,1], and get their rgb values
rgba = [cmap((k-min_height)/max_height) for k in dz]

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection="3d")
ax.bar3d(n, K, np.zeros(len(data_ex2)), dx, dy, dz, color=rgba)

ax.set_zlabel("time_cublas")
ax.set_ylabel("K")
ax.set_xlabel("log(N) basis=10")
#ax.set_xscale('symlog')
# plt.legend()
plt.title(
    "cublas implementation for K dot products <x,v_i> for i{0,..,K} and len(x)=len(v_i) = N on K40")
# plt.grid()
plt.savefig("plots/ex2_cublas.jpg", bbox_inches='tight')


# # cublas plot
# dx = 100*np.ones(len(data_ex2))
# dy = 100*np.ones(len(data_ex2))
# dz = time_cublas

# cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
# max_height = np.max(dz)   # get range of colorbars so we can normalize
# min_height = np.min(dz)
# # scale each z to [0,1], and get their rgb values
# rgba = [cmap((k-min_height)/max_height) for k in dz]

# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111, projection="3d")
# ax.bar3d(N, K, np.zeros(len(data_ex2)), dx, dy, dz, color=rgba)

# ax.set_zlabel("time_cublas")
# ax.set_ylabel("N")
# ax.set_xlabel("K")
# # plt.legend()
# plt.title(
#     "cublas implementation for K dot products <x,v_i> for i{0,..,K} and len(x)=len(v_i) = N")
# # plt.grid()
# plt.savefig("plots/ex2_cublas.jpg", bbox_inches='tight')
