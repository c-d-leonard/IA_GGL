import matplotlib.pyplot as plt
import numpy as np

(sigz, Ncorr) = np.loadtxt('./txtfiles/Ncorr_of_sigz_Dec19.txt', unpack=True)

plt.figure()
plt.semilogx(sigz, Ncorr, 'ro')
plt.xlabel('$\sigma_z$')
plt.ylabel('$N$')
plt.savefig('./Ncorr_Dec19.png')
