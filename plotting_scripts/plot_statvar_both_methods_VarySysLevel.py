import numpy as np
import matplotlib.pyplot as plt

(rp_c, var_blazek_10) = np.loadtxt('../txtfiles/frac_StatError_Blazek_LRG-shapes_mod_dNdz_pz_pars.txt', unpack=True)

(rp_c, var_blazek_20) = np.loadtxt('../txtfiles/frac_StatError_Blazek_LRG-shapes_mod_dNdz_pz_pars.txt', unpack=True)

(rp_c, var_blazek_30) = np.loadtxt('../txtfiles/frac_StatError_Blazek_LRG-shapes_mod_dNdz_pz_pars.txt', unpack=True)

(rp_c, var_shapes_10_cov60) = np.loadtxt('../txtfiles/fractional_staterror_shapemethod_LRG-shapes_covperc=0.6_a=0.714285714286_mod_dNdz_pz_pars.txt', unpack=True)

(rp_c, var_shapes_20_cov60) = np.loadtxt('../txtfiles/fractional_staterror_shapemethod_LRG-shapes_covperc=0.6_a=0.714285714286_mod_dNdz_pz_pars.txt', unpack=True)

(rp_c, var_shapes_30_cov60) = np.loadtxt('../txtfiles/fractional_staterror_shapemethod_LRG-shapes_covperc=0.6_a=0.714285714286_mod_dNdz_pz_pars.txt', unpack=True)

plt.figure()
plt.loglog(rp_c,var_blazek_10, 'go')
plt.hold(True)
plt.loglog(rp_c,var_blazek_20, 'mo')
plt.hold(True)
plt.loglog(rp_c,var_blazek_30, 'bo')
plt.hold(True)
plt.loglog(rp_c,var_shapes_10_cov60, 'g+', label='10%')
plt.hold(True)
plt.loglog(rp_c,var_shapes_20_cov60, 'm+', label='20%')
plt.hold(True)
plt.loglog(rp_c,var_shapes_30_cov60, 'b+', label='30%')
plt.legend()
plt.xlabel('$r_p$, Mpc/h')
plt.ylabel('Fractional error')
plt.xlim(0.04, 35)
#plt.ylim(0.02, 3)
plt.title('Stat error, cov = 60%, + shapes, o Blazek')
plt.savefig('../plots/frac_staterr_VarySys_cov60_mod_dNdz_pz_pars.pdf')
plt.close()

(rp_c, var_shapes_10_cov80) = np.loadtxt('../txtfiles/fractional_staterror_shapemethod_LRG-shapes_covperc=0.8_a=0.714285714286_mod_dNdz_pz_pars.txt', unpack=True)

(rp_c, var_shapes_20_cov80) = np.loadtxt('../txtfiles/fractional_staterror_shapemethod_LRG-shapes_covperc=0.8_a=0.714285714286_mod_dNdz_pz_pars.txt', unpack=True)

(rp_c, var_shapes_30_cov80) = np.loadtxt('../txtfiles/fractional_staterror_shapemethod_LRG-shapes_covperc=0.8_a=0.714285714286_mod_dNdz_pz_pars.txt', unpack=True)

plt.figure()
plt.loglog(rp_c,var_blazek_10, 'go')
plt.hold(True)
plt.loglog(rp_c,var_blazek_20, 'mo')
plt.hold(True)
plt.loglog(rp_c,var_blazek_30, 'bo')
plt.hold(True)
plt.loglog(rp_c,var_shapes_10_cov80, 'g+', label='10%')
plt.hold(True)
plt.loglog(rp_c,var_shapes_20_cov80, 'm+', label='20%')
plt.hold(True)
plt.loglog(rp_c,var_shapes_30_cov80, 'b+', label='30%')
plt.legend()
plt.xlabel('$r_p$, Mpc/h')
plt.ylabel('Fractional error')
plt.xlim(0.04, 35)
#plt.ylim(0.02, 3)
plt.title('Stat error, cov = 80%, + shapes, o Blazek')
plt.savefig('../plots/frac_staterr_VarySys_cov80_mod_dNdz_pz_pars.pdf')
plt.close()


