import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#files
filepath1 = "data/basic_cuda_a.txt"
data_a = np.genfromtxt(filepath1, dtype=float, delimiter=' ')
filepath2 = "data/basic_cuda_b.txt"
data_b = np.genfromtxt(filepath2, dtype=float, delimiter=' ')
filepath3 = "data/basic_cuda_c.txt"
data_c= np.genfromtxt(filepath3, dtype=float, delimiter=' ')
# data = np.loadtxt(filepath,dtype = str, delimiter=',')

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
plt.yscale('log', base=2)
plt.xlabel("vectorlength N")
plt.ylabel("time [s]")
plt.legend()
plt.title("BASIC CUDA memory allocation")
plt.grid()
plt.savefig("plots/basic_cuda_a.pdf", bbox_inches='tight')

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
plt.savefig("plots/basic_cuda_b.pdf", bbox_inches='tight')

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
plt.savefig("plots/basic_cuda_c.pdf", bbox_inches='tight')