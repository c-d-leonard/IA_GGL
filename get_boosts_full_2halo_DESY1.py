# Script to get boosts using only the 2halo term for the correlation function and in the case where we have an input weighted vector dNdz for sources and input vector dNdz for lenses.

import numpy as np
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import os.path

import shared_functions_setup as setup
import shared_functions_wlp_wls as shared
import pyccl as ccl

SURVEY = 'DESY1'
endfile = 'narrow_lens'

if (SURVEY=='SDSS'):
    import params_SDSS_testpz as pa
elif (SURVEY=='LSST_DESI'):
    import params_LSST_DESI as pa
elif (SURVEY=='DESY1'):
    import params_DESY1_testpz as pa
else:
    print("We don't have support for that survey yet; exiting.")
    exit()
    
# Set up rp vector

cosmo = ccl.Cosmology(Omega_c = pa.OmC_t, Omega_b = pa.OmB, h = (pa.HH0_t/100.), sigma8 = pa.sigma8, n_s=pa.n_s)

# Option to provide theta min and theta max and convert to rp for a given effective lens redshift:
theta_min = 0.1
theta_max = 200
rp_min = setup.arcmin_to_rp(theta_min, pa.zeff,cosmo)
rp_max = setup.arcmin_to_rp(theta_max, pa.zeff,cosmo)
print("rp_min=", rp_min, "rp_max=", rp_max)

rp_edges = 	setup.setup_rp_bins(rp_min, rp_max, pa.N_bins)
rpvec	=	setup.rp_bins_mid(rp_edges)

np.savetxt('./txtfiles/boosts/rpvec.txt', rpvec)
	
# First check if we need to do this:
Boost_file_a = './txtfiles/boosts/Boost_A_survey='+SURVEY+'_'+endfile+'.txt'
Boost_file_b = './txtfiles/boosts/Boost_B_survey='+SURVEY+'_'+endfile+'.txt'

"""if (os.path.isfile(Boost_file_a) and os.path.isfile(Boost_file_b)):
    print("The boost files have previously been calculated for this endfile.")
    Boost_a = np.loadtxt('./txtfiles/boosts/Boost_A_survey='+SURVEY+'_'+endfile+'.txt')
    Boost_b = np.loadtxt('./txtfiles/boosts/Boost_B_survey='+SURVEY+'_'+endfile+'.txt')
    interp_B_a = scipy.interpolate.interp1d(np.log(rpvec), np.log(Boost_a))
    interp_B_b = scipy.interpolate.interp1d(np.log(rpvec), np.log(Boost_b))

    print("B_a 1 Mpc/h=", np.exp(interp_B_a(np.log(1.))))
    print("B_b 1 Mpc/h=", np.exp(interp_B_b(np.log(1.))))
	
    exit()"""

zLvec = np.loadtxt('./z_list_DESY1.txt')
#print("zLvec=", zLvec)

# Import the correlation function, for the 2halo term.
xi_2h_mm = np.zeros((40000, len(zLvec)))
for zi in range(0,len(zLvec)):
    stringz = '{:1.12f}'.format(zLvec[zi])
    #(r, xi_2h_mm[:, zi]) = np.loadtxt('./txtfiles/halofit_xi/xi2h_z='+stringz+'_'+endfile+'.txt', unpack=True)
    (r, xi_2h_mm[:, zi]) = np.loadtxt('./txtfiles/halofit_xi/xi2h_z='+stringz+'_test.txt', unpack=True)
xi = pa.bd* pa.bs * xi_2h_mm

# Get the comoving distance associated to the lens redshift
chi_vec = np.zeros(len(zLvec))
for zi in range(0,len(zLvec)):
    chi_vec[zi] = ccl.comoving_radial_distance(cosmo, 1./(1.+zLvec[zi])) * (pa.HH0_t/100.) # CCL returns in Mpc but we want Mpc/h

# Figure out the min and max value of the * positive * part of the vector of projected distances

minPiPos = 10**(-8)
max_int = 200. # We integrate the numerator out to 200 Mpc/h away from the lenses because the correlation is ~zero outside of this.
maxPiPos = max_int
Pi_pos = np.logspace(np.log10(minPiPos), np.log10(maxPiPos), 3000)
    
# Pi can be positive or negative, so now flip this and include the negative values, but only down to z=0
# And avoid including multiple of the same values - this messes up some integration routines.
print("Get Pi")
Pi = [0]*len(zLvec)
chismin = ccl.comoving_radial_distance(cosmo, 1./(1.+3.393976853817444635e-03)) * (pa.HH0_t / 100.) # CCL returns in Mpc but we want Mpc/h
for zi in range(0, len(zLvec)):
    print("zi=", zi)
    Pi_pos_vec= list(Pi_pos)[1:]
    Pi_pos_vec.reverse()
    index_cut = next(j[0] for j in enumerate(Pi_pos_vec) if j[1]<=(chi_vec[zi]-chismin))
    Pi[zi] = np.append(-np.asarray(Pi_pos_vec[index_cut:]), np.append([0],Pi_pos))
    #print("Pi=", Pi[zi])

# Get the correlation function in terms of Pi and rp 
xi_interp_r = [0] * len(zLvec)
xi_ofPi = [0] * len(zLvec)
print("get 2D xi")
for zi in range(0,len(zLvec)):
    print("zi=", zi)
    xi_interp_r[zi] = scipy.interpolate.interp1d(np.log(r), xi[:, zi])
    xi_ofPi[zi] = np.zeros((len(rpvec), len(Pi[zi])))
    for ri in range(0,len(rpvec)):
        for pi in range(0,len(Pi[zi])):
            xi_ofPi[zi][ri, pi] = xi_interp_r[zi](np.log(np.sqrt(rpvec[ri]**2 + Pi[zi][pi]**2)))
    
# Get the vector of com dist values associated to Pi values:
com_Pi = [0]*len(zLvec)
z_Pi = [0]*len(zLvec)
print("Get z and com for Pi")
for zi in range(0,len(zLvec)):
    print("zi=", zi)
    com_Pi[zi] = chi_vec[zi] + Pi[zi]
    z_Pi[zi] = (1./ccl.scale_factor_of_chi(cosmo, com_Pi[zi] / (pa.HH0_t/100.))) - 1.  # CCL wants distance in Mpc but we are working in Mpc/h

# Now we have xi_{ls}(rp, Pi(z_s); z_L)

# Okay, now we do the required integrals:
# Load the weighted dNdz_mc for the two source samples:
# Interpolate to get the weighted dNdz in terms of the required z_Pi vectors at each z_l.
z_a = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin0_zmc_centres.dat')
dNdz_a_weighted = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin0_zmc_weighted')
interp_dndza = scipy.interpolate.interp1d(z_a, dNdz_a_weighted) 

#plt.figure()
#plt.plot(z_a, dNdz_a_weighted)
#plt.savefig('./dNdz_weighted_a_boosts.png')
#plt.close()

z_b = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_centres.dat')
dNdz_b_weighted = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_weighted')
interp_dndzb = scipy.interpolate.interp1d(z_b, dNdz_b_weighted) 

#plt.figure()
#plt.plot(z_b, dNdz_b_weighted)
#plt.savefig('./dNdz_weighted_b_boosts.png')
#plt.close()

# Get dNdz
dNdz_a = [0] * len(zLvec)
dNdz_b = [0] * len(zLvec)
for zi in range(0, len(zLvec)):
    dNdz_a[zi] = interp_dndza(z_Pi[zi])
    dNdz_b[zi] = interp_dndzb(z_Pi[zi])
    #plt.figure()
    #plt.plot(z_Pi[zi], dNdz_a[zi])
    #plt.plot(z_Pi[zi], dNdz_b[zi])
    #plt.savefig('./dNdz_z='+str(z_Pi[zi])+'.png')
    
# Now do the integrals in zmc
B_1_a = np.zeros((len(zLvec), len(rpvec)))
B_1_b = np.zeros((len(zLvec), len(rpvec)))
for zi in range(0,len(zLvec)):
    print("zi=", zi)
    for ri in range(len(rpvec)):
        # Instead of this go straight to dN_w/dz_mc * xi
        # Norm is over full dNdz because z_Pi is just cut short because we don't expect correlation outside this extent because xi drops off, should not affect norm
        B_1_a[zi,ri] = scipy.integrate.simps(dNdz_a[zi]*xi_ofPi[zi][ri,:], z_Pi[zi]) / scipy.integrate.simps(dNdz_a_weighted, z_a)
        B_1_b[zi,ri] = scipy.integrate.simps(dNdz_b[zi]*xi_ofPi[zi][ri,:], z_Pi[zi]) / scipy.integrate.simps(dNdz_b_weighted, z_b)

# Now integrate these over zl

# Load lens redshifts from file
#zL, dndzl = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/z_dNdz_lenses_subbin.dat', unpack=True)
#zL, dndzl = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/z_dNdz_lenses.dat', unpack=True)
zL, dndzl = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/z_dNdz_lenses_narrow_test.dat', unpack=True)
norm_L = scipy.integrate.simps(dndzl, zL)

Boost_a = np.zeros(len(rpvec))
Boost_b = np.zeros(len(rpvec))

for ri in range(0,len(rpvec)):
    Boost_a[ri] = scipy.integrate.simps(B_1_a[:, ri] * dndzl, zL) / norm_L
    Boost_b[ri] = scipy.integrate.simps(B_1_b[:,ri]* dndzl, zL) / norm_L

np.savetxt(Boost_file_a, Boost_a)
np.savetxt(Boost_file_b, Boost_b)

print("Boosts computed")

interp_B_a = scipy.interpolate.interp1d(rpvec, Boost_a)
interp_B_b = scipy.interpolate.interp1d(rpvec, Boost_b)

print("B_a 1 Mpc/h=", interp_B_a(1.))
print("B_b 1 Mpc/h=", interp_B_b(1.))




