import numpy as np
import matplotlib.pyplot as plt

rp = np.asarray([0.63, 1, 1.58, 2.5, 4, 6.3, 10, 15.8, 25., 40.])
from_M13 = np.asarray([500., 280., 170., 105., 80., 50., 32., 20., 10., 5.5])
our_code = np.asarray([143., 91., 64., 47., 35., 26.,18.5, 12., 6.8, 3.2])

plt.figure()
plt.loglog(rp, from_M13, 'go')
plt.hold(True)
plt.loglog(rp, our_code * (2.07 / 1.77)**2, 'mo')
plt.xlim(0.4, 80.)
plt.ylabel('$w_{gg}$, Mpc/h, com')
plt.xlabel('$r_p$, Mpc/h, com')
plt.savefig('../plots/wgg_compare_M13.png')
plt.close()

plt.figure()
plt.loglog(rp, np.abs((from_M13 - our_code) / our_code), 'bo')
plt.xlim(0.4, 80.)
plt.ylabel('$w_{gg}$, Mpc/h, com')
plt.xlabel('$r_p$, Mpc/h, com')
plt.savefig('../plots/wgg_fracdiff_compare_M13.png')
plt.close()
