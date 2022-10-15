import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

#files
filepath1 = "data/basic_cuda_a.txt"
data_a = np.genfromtxt(filepath1, dtype=float, delimiter=' ')
filepath1a = "data/basic_cuda_a2.txt"
data_a2 = np.genfromtxt(filepath1a, dtype=float, delimiter=' ')
filepath2 = "data/basic_cuda_b.txt"
data_b = np.genfromtxt(filepath2, dtype=float, delimiter=' ')
filepath3 = "data/basic_cuda_c.txt"
data_c= np.genfromtxt(filepath3, dtype=float, delimiter=' ')
filepath4 = "data/basic_cuda_e.txt"
data_e= np.genfromtxt(filepath4, dtype=float, delimiter=' ')
filepath5 = "data/cuda_dot.txt"
data_dot= np.genfromtxt(filepath5, dtype=float, delimiter=' ')


#plota
vectorlength=[]
time_al=[]
time_fr=[]
for i in range(len(data_a)):
    vectorlength.append(data_a[i][0])
    time_al.append(data_a[i][1])
    time_fr.append(data_a[i][2])

plt.figure(figsize=(10,5))
plt.plot(vectorlength, time_al , label = "allocation_time")
plt.plot(vectorlength, time_fr , label = "freeing_time")
# plt.plot(vectorlength, data , label = "time")
plt.xscale('log', base=10)
#plt.yscale('log', base=2)
plt.xlabel("vectorlength N")
plt.ylabel("time [s]")
plt.legend()
plt.title("BASIC CUDA memory allocation")
plt.grid()
plt.savefig("plots/basic_cuda_a.jpg", bbox_inches='tight')

#plota
vectorlength=[]
time_al=[]
time_fr=[]
for i in range(len(data_a2)):
    vectorlength.append(data_a2[i][0])
    time_al.append(data_a2[i][1])
    time_fr.append(data_a2[i][2])

plt.figure(figsize=(10,5))
plt.plot(vectorlength, time_al , label = "allocation_time")
plt.plot(vectorlength, time_fr , label = "freeing_time")
# plt.plot(vectorlength, data , label = "time")
plt.xscale('log', base=10)
#plt.yscale('log', base=2)
plt.xlabel("vectorlength N")
plt.ylabel("time [s]")
plt.legend()
plt.title("BASIC CUDA memory allocation")
plt.grid()
plt.savefig("plots/basic_cuda_a2.jpg", bbox_inches='tight')

#plotb
vectorlength=[]
time_cpy=[]
time_ker=[]
for i in range(len(data_b)):
    vectorlength.append(data_b[i][0])
    time_cpy.append(data_b[i][1])
    time_ker.append(data_b[i][2])

plt.figure(figsize=(10,5))
plt.plot(vectorlength, time_cpy , label = "init_time_cpy")
plt.plot(vectorlength, time_ker , label = "init_time_kernel")
# plt.plot(vectorlength, data , label = "time")
plt.xscale('log', base=10)
plt.yscale('log', base=2)
plt.xlabel("vectorlength N")
plt.ylabel("time [s]")
plt.legend()
plt.title("BASIC CUDA 2 options for initialization")
plt.grid()
plt.savefig("plots/basic_cuda_b.jpg", bbox_inches='tight')

#plotcd
vectorlength=[]
time_add=[]
# time_ker=[]
for i in range(len(data_c)):
    vectorlength.append(data_c[i][0])
    time_add.append(data_c[i][1])
    #time_ker.append(data_b[i][2])

plt.figure(figsize=(10,5))
plt.plot(vectorlength, time_add , label = "+operation time")
#plt.plot(vectorlength, time_ker , label = "init_time_kernel")
# plt.plot(vectorlength, data , label = "time")
plt.xscale('log', base=10)
plt.yscale('log', base=2)
plt.xlabel("vectorlength N")
plt.ylabel("time [s]")
plt.legend()
plt.title("BASIC CUDA addition")
plt.grid()
plt.savefig("plots/basic_cuda_c.jpg", bbox_inches='tight')

#plote
n_blocks=[]
n_threads=[]
time=[]
for i in range(len(data_e)):
    n_blocks.append(data_e[i][0])
    n_threads.append(data_e[i][1])
    time.append(data_e[i][2])

dx = 100*np.ones(len(data_e))
dy = 100*np.ones(len(data_e))
dz = time

cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
max_height = np.max(dz)   # get range of colorbars so we can normalize
min_height = np.min(dz)   
# scale each z to [0,1], and get their rgb values
rgba = [cmap((k-min_height)/max_height) for k in dz] 

fig=plt.figure(figsize=(10,5))
ax = fig.add_subplot(111,projection="3d")
ax.bar3d(n_blocks,n_threads, np.zeros(len(data_e)), dx,dy,dz, color=rgba)

ax.set_zlabel("time")
ax.set_ylabel("#threads")
ax.set_xlabel("#blocks")
# plt.legend()
plt.title("BASIC CUDA <<< #blocks, #threads/block>>> configurations")
# plt.grid()
plt.savefig("plots/basic_cuda_e.jpg", bbox_inches='tight')

"""
#plot_dot
vectorlength=[]
time_coop=[]
time_2ker=[]
for i in range(len(data_dot)):
    vectorlength.append(data_b[i][0])
    time_coop.append(data_b[i][2])
    time_2ker.append(data_b[i][1])

plt.figure(figsize=(10,5))
plt.plot(vectorlength, time_coop , label = "time GPU/CPU cooperation")
plt.plot(vectorlength, time_2ker , label = "time two-kernel operation")
# plt.plot(vectorlength, data , label = "time")
plt.xscale('log', base=10)
plt.yscale('log', base=2)
plt.xlabel("vectorlength N")
plt.ylabel("time [s]")
plt.legend()
plt.title("CUDA dot product 2 options")
plt.grid()
plt.savefig("plots/cuda_dot.jpg", bbox_inches='tight')
"""