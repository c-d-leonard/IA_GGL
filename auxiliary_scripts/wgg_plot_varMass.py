import numpy as np
import matplotlib.pyplot as plt

#N = 7

r1e12, wgg1e12 = np.loadtxt('../txtfiles/wgg_1000pts_newexp1h_M1e12.txt', unpack=True)
r4e12, wgg4e12 = np.loadtxt('../txtfiles/wgg_1000pts_newexp1h_M4e12.txt', unpack=True)
rLOWZ, wggLOWZ = np.loadtxt('../txtfiles/wgg_1000pts_newexp1h_MLOWZ.txt', unpack=True)
r2e13, wgg2e13 = np.loadtxt('../txtfiles/wgg_1000pts_newexp1h_M2e13.txt', unpack=True)

plt.figure()
plt.semilogx(r1e12, r1e12*wgg1e12, 'b', label='M=1e12')
plt.hold(True)
plt.semilogx(r4e12, r4e12*wgg4e12, 'k', label='M=4e12')
plt.hold(True)
plt.semilogx(rLOWZ, rLOWZ*wggLOWZ, 'm', label='M=LOWZ')
plt.hold(True)
plt.semilogx(r2e13, r2e13*wgg2e13, 'r', label ='M=2e13')
plt.xlim(0.1, 10.)
plt.ylim(0., 300.)
plt.legend()
plt.savefig('../plots/wgg_various_masses_newexp.png')
plt.close()



















