import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

def SM_NofZ(Nstar_file):
	""" Loads from file and normalizes the stellar mass number density, then returns it. """
	
	(mstar, Ngal) = np.loadtxt(Nstar_file, unpack=True)
	
	norm = scipy.integrate.simps(Ngal, 10**mstar)
	
	Ngal_normed = Ngal / norm
	
	return (10**mstar, Ngal_normed)
	
ms, Ns = SM_NofZ('./txtfiles/stellar_masses_boss.txt')

plt.figure()
plt.plot(np.log10(ms), Ns, 'mo')
plt.savefig('./plots/stellarmass_fromfile.pdf')
plt.close()


