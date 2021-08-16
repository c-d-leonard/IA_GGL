import numpy as np
import shared_functions_wlp_wls as shared

survey = 'DESY1'
endfile = 'with1halo'

if (survey == 'SDSS'):
	import params as pa
elif (survey == 'LSST_DESI'):
	import params_LSST_DESI as pa
elif (survey == 'DESY1'):
	import params_DESY1_testpz as pa
else:
	print("We don't have support for that survey yet; exiting.")
	exit()

print("Run 1 halo terms")

# Get y(k, M) for calculating various power spectra from the halo model
logkmin = -6; kpts =40000; logkmax = 5; Mmax = 15;
kvec_FT = np.logspace(logkmin, logkmax, kpts)
kvec_short = np.logspace(np.log10(kvec_FT[0]), np.log10(kvec_FT[-1]), 40)
Mhalo = np.logspace(10., Mmax, 30)
y_ls = shared.gety_ls(Mhalo, kvec_short, survey)
y_ldm = shared.gety_ldm(Mhalo, kvec_short, survey)
print("y computed")

# Get M* low for the Zu & Mandelbaum halo model.
tot_nsrc= shared.vol_dens(pa.fsky, pa.N_shapes, survey)
Mstarlow = shared.get_Mstar_low(survey, tot_nsrc)
print("Mstar low computed")

# Get the 1halo term for lxs averaged over z (save to file)
#shared.get_Pkgg_1halo(kvec_FT, pa.fsky, Mhalo, kvec_short, y_ls, y_ldm, Mstarlow, endfile, survey)
#print("Pkgg 1halo computed")

# Get the 1halo term for lxs at individual z's for the Boost (save to file)
shared.get_Pkgg_1halo_multiz(kvec_FT, pa.fsky, Mhalo, kvec_short, y_ls, y_ldm, Mstarlow, endfile, survey)
print("Pkgg 1halo multiz computed")

# Get the 1halo term for lxmatter integrated over z to get Delta Sigma theory (save to file)
shared.get_Pkgm_1halo(kvec_FT, Mhalo, kvec_short, y_ldm, endfile, survey)
print("Pkgm 1halo computed")

print("Run 2 halo terms")
#shared.get_Pkgg_2h_multiz(kvec_FT, endfile, survey)
print("Pkgg 2halo multiz and z averaged computed")









