import numpy as np
import matplotlib.pyplot as plt

(rp_c, var_blazek) = np.loadtxt('../txtfiles/frac_StatError_Blazek_LRG-shapes_updateSYS.txt', unpack=True)

(rp_c, var_shapes_a1) = np.loadtxt('../txtfiles/fractional_staterror_shapemethod_LRG-shapes_covperc=0.6_a=0.8_7bins_updateSYS.txt', unpack=True)

(rp_c, var_shapes_a2) = np.loadtxt('../txtfiles/fractional_staterror_shapemethod_LRG-shapes_covperc=0.6_a=0.666666666667_7bins_updateSYS.txt', unpack=True)

(rp_c, var_shapes_a3) = np.loadtxt('../txtfiles/fractional_staterror_shapemethod_LRG-shapes_covperc=0.6_a=0.571428571429_7bins_updateSYS.txt', unpack=True)


plt.figure()
plt.loglog(rp_c,var_blazek, 'go', label='Blazek et al. method')
plt.hold(True)
#plt.loglog(rp_c,var_shapes_0pt5, 'bo', label='Shapes method, cov 0.5')
#plt.hold(True)
plt.loglog(rp_c,var_shapes_a1, 'yo', label='Shapes, a=0.8')
plt.hold(True)
plt.loglog(rp_c,var_shapes_a2, 'mo', label='Shapes, a=2/3')
plt.hold(True)
plt.loglog(rp_c,var_shapes_a3, 'bo', label='Shapes, a=0.57')
plt.legend()
plt.xlabel('$r_p$, Mpc/h')
plt.ylabel('Fractional error')
#plt.xlim(0.07, 60)
#plt.ylim(0.09, 10)
plt.xlim(0.04, 35)
plt.title('Stat error, shape method cov = 60%')
plt.savefig('../plots/frac_staterr_f(a)_7bins_updateSYS.pdf')
plt.close()

(rp_c, var_shapes_cov1) = np.loadtxt('../txtfiles/fractional_staterror_shapemethod_LRG-shapes_covperc=0.4_a=0.666666666667_7bins_updateSYS.txt', unpack=True)

(rp_c, var_shapes_cov2) = np.loadtxt('../txtfiles/fractional_staterror_shapemethod_LRG-shapes_covperc=0.6_a=0.666666666667_7bins_updateSYS.txt', unpack=True)

(rp_c, var_shapes_cov3) = np.loadtxt('../txtfiles/fractional_staterror_shapemethod_LRG-shapes_covperc=0.8_a=0.666666666667_7bins_updateSYS.txt', unpack=True)

plt.figure()
plt.loglog(rp_c,var_blazek, 'go', label='Blazek et al. method')
plt.hold(True)
plt.loglog(rp_c,var_shapes_cov1, 'yo', label='Shapes method, cov = 0.4')
plt.hold(True)
plt.loglog(rp_c,var_shapes_cov2, 'mo', label='Shapes method, cov = 0.6')
plt.hold(True)
plt.loglog(rp_c,var_shapes_cov3, 'bo', label='Shapes method, cov = 0.8')
plt.legend()
plt.xlabel('$r_p$, Mpc/h')
plt.ylabel('Fractional error')
#plt.xlim(0.07, 60)
#plt.ylim(0.09, 3)
plt.xlim(0.04, 35)
plt.title('stat error, a = 2/3')
plt.savefig('../plots/frac_staterr_f(cov_methods)_7bins_updateSYS.pdf')
plt.close()
