import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

# files
filepath1 = "data/ex1.txt"
data_ex1 = np.genfromtxt(filepath1, dtype=float, delimiter=' ')

# filepath2 = "data/ex2.txt"
# data_ex1 = np.genfromtxt(filepath2, dtype=float, delimiter=' ')
# filepath3 = "data/dense_matrix_GB_k40.txt"
# data_dense= np.genfromtxt(filepath3, dtype=float, delimiter=' ')


# plot1 - dot product
N = []
t_dot_product = []
t_shared_mem = []
t_ws_fixed = []
t_ws_elastix = []

for i in range(len(data_ex1)):
    N.append(data_ex1
             [i][0])
    t_dot_product.append(data_ex1
                         [i][1])
    t_shared_mem.append(data_ex1
                        [i][2])
    t_ws_fixed.append(data_ex1
                      [i][3])
    t_ws_elastix.append(data_ex1
                        [i][4])

plt.figure(figsize=(10, 5))
plt.plot(N, t_dot_product, label="dot-product reference", color='y')
plt.plot(N, t_ws_fixed,
         label="shared-memory implementation", color='g')
plt.plot(N, t_shared_mem,
         label="warf-shuffle implementation with fixed #threads", color='c')
plt.plot(N, t_ws_elastix, label="warf-shuffle implementation with #threads(N)",
         color='c', linestyle='dashdot')
#plt.plot(k, data , label = "time")
plt.xscale('log', base=10)
plt.yscale('log', base=10)
plt.xlabel("vector length N")
plt.ylabel("computation time [s]")
plt.legend()
plt.title(
    "Performance of different norm-computing-kernelson on the RTX3060-machine")
plt.grid()
plt.savefig("plots/ex1.jpg", bbox_inches='tight')

# # plot2b - dense matrix
# N = []
# t_dot_product = []
# t_shared_mem = []
# t_ws_fixed   = []
# t_ws_elastix = []

# for i in range(len(data_dense_matrix_GB_rtx3060)):
#     N.append(data_dense_matrix_GB_rtx3060[i][0])
#     t_dot_product.append(data_dense_matrix_GB_rtx3060[i][1])
#     t_shared_mem.append(data_dense_matrix_GB_rtx3060[i][2])
#     t_ws_fixed  .append(data_dense_matrix_GB_rtx3060[i][3])
#     t_ws_elastix.append(data_dense_matrix_GB_rtx3060[i][4])

# plt.figure(figsize=(10, 5))
# plt.plot(N, t_dot_product, label="bandwidth for my vanilla transformation", color='g')
# plt.plot(N, t_ws_fixed  ,
#          label="bandwidth for the read-write-optimal transformation", color='g', linestyle='dotted')
# plt.plot(N, t_shared_mem,
#          label="bandwidth for my vanilla in-place-transformation", color='c')
# plt.plot(N, t_ws_elastix, label="bandwidth for read-write-optimal in-place-transformation",
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
