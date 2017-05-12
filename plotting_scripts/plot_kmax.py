import matplotlib.pyplot as plt
import numpy as np

N=5

Rp=[0]*N
projcorr = [0]*N
Rproj = [0]*N

kmax = [2000, 4000, 6000, 8000, 10000]

for i in range(0,N):
	Rp[i] = np.loadtxt('../txtfiles/corr_rp_z=0.32_kmax='+str(kmax[i])+'.txt')
	projcorr[i] = np.loadtxt('../txtfiles/proj_corr_k='+str(kmax[i])+'.txt')
	Rproj[i] = np.loadtxt('../txtfiles/Rint_proj_k='+str(kmax[i])+'.txt')

plt.figure()
plt.loglog(Rp[0], projcorr[0], 'b+')
plt.hold(True)
plt.loglog(Rp[1], projcorr[1], 'r+')
plt.hold(True)
plt.loglog(Rp[2], projcorr[2], 'g+')
plt.hold(True)
plt.loglog(Rp[3], projcorr[3], 'm+')
plt.hold(True)
plt.loglog(Rp[4], projcorr[4], 'k+')
plt.hold(True)
plt.savefig('../plots/projcorr_kmax.png')

plt.figure()
plt.loglog(Rp[0], Rproj[0], 'b+')
plt.hold(True)
plt.loglog(Rp[1], Rproj[1], 'r+')
plt.hold(True)
plt.loglog(Rp[2], Rproj[2], 'g+')
plt.hold(True)
plt.loglog(Rp[3], Rproj[3], 'm+')
plt.hold(True)
plt.loglog(Rp[4], Rproj[4], 'k+')
plt.hold(True)
plt.savefig('../plots/Rproj_kmax.png')

plt.figure()
plt.loglog(Rp[0], Rproj[0]- projcorr[0], 'b+')
plt.hold(True)
plt.loglog(Rp[1], Rproj[1]-projcorr[1], 'r+')
plt.hold(True)
plt.loglog(Rp[2], Rproj[2]-projcorr[2], 'g+')
plt.hold(True)
plt.loglog(Rp[3], Rproj[3]-projcorr[3], 'm+')
plt.hold(True)
plt.loglog(Rp[4], Rproj[4]-projcorr[4], 'k+')
plt.hold(True)
plt.savefig('../plots/diff_kmax.png')
