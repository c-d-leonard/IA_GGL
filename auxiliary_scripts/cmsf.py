import numpy as np
import matplotlib.pyplot as plt

def CSMF(Mh, mstar):
	""" Compute the cumulative stellar mass function with the form used in 1601.06791 """
	
	b0 = -0.14
	b1 = 1.03
	alpha_s = -1.00
	Ms0 = 10**11.22
	Mh1 = 10**11.70
	beta_1 = 4.5
	beta_2 = 0.05
	
	M13 = Mh / 10**13 # In units Msol / h
	
	phis = 10** (b0 + b1 * np.log10(M13))
	
	Mc = Ms0 * (Mh / Mh1)**beta_1 / (1. + (Mh /Mh1)** (beta_1 - beta_2))
	
	Ms = Mc * 0.56
	
	Phi = phis / Ms * (mstar / Ms)**alpha_s * np.exp( - (mstar / Ms)**2)
	
	return Phi


mstar = np.logspace(-10, 12, 1000)

Mh = 10.**13

csmf = CSMF(Mh, mstar)

plt.figure()
plt.semilogx(mstar, csmf)
plt.savefig('./plots/csmf_test.pdf')
plt.close()

