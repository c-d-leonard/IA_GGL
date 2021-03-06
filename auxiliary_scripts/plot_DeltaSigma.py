import numpy as np
import matplotlib.pyplot as plt

#(r11, DS_11) = np.loadtxt('../txtfiles/DeltaSigma_M1e11_newexp.txt', unpack=True)
(r12, DS_12) = np.loadtxt('../txtfiles/DeltaSigma_M1e12_newexp.txt', unpack=True)
(r13, DS_13) = np.loadtxt('../txtfiles/DeltaSigma_M1e13_newexp.txt', unpack=True)
(r14, DS_14) = np.loadtxt('../txtfiles/DeltaSigma_M1e14_newexp.txt', unpack=True)
#(r15, DS_15) = np.loadtxt('../txtfiles/DeltaSigma_M1e15_newexp.txt', unpack=True)

plt.figure()
#plt.loglog(r11, DS_11,label='M=1e11')
#plt.hold(True)
plt.loglog(r12, DS_12, label='M=1e12')
plt.hold(True)
plt.loglog(r13, DS_13, label='M=1e13')
plt.hold(True)
plt.loglog(r14, DS_14, label='M=1e14')
plt.hold(True)
#plt.loglog(r15, DS_15, label='M=1e15')
#plt.hold(True)
plt.xlim(0.01, 30.)
plt.ylim(0.1,5000)
plt.xlabel('$r_p$')
plt.ylabel('$\Delta \Sigma$')
plt.legend()
plt.savefig('../plots/DeltaSigma_various_masses_newexp.png')
plt.close()
