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
endfile = 'test'

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

cosmo = ccl.Cosmology(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), sigma8 = pa.sigma8, n_s=pa.n_s)

# Option to provide theta min and theta max and convert to rp for a given effective lens redshift:
theta_min = 0.1
theta_max = 200
rp_min = setup.arcmin_to_rp(theta_min, pa.zeff,cosmo)
rp_max = setup.arcmin_to_rp(theta_max, pa.zeff,cosmo)
print("rp_min=", rp_min, "rp_max=", rp_max)

rp_edges = 	setup.setup_rp_bins(rp_min, rp_max, pa.N_bins)
rpvec	=	setup.rp_bins_mid(rp_edges)

#rp_edges = setup.setup_rp_bins(pa.rp_min, pa.rp_max, pa.N_bins)
#rpvec = setup.rp_bins_mid(rp_edges)
	
# First check if we need to do this:
Boost_file_a = './txtfiles/boosts/Boost_full_A_survey='+SURVEY+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt'
Boost_file_b = './txtfiles/boosts/Boost_full_B_survey='+SURVEY+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt'

if (os.path.isfile(Boost_file_a) and os.path.isfile(Boost_file_b) and os.path.isfile(Boost_file_asc) and os.path.isfile(Boost_file_ascBl)):
    print("The boost files have previously been calculated for this endfile.")
    Boost_a = np.loadtxt('./txtfiles/boosts/Boost_full_A_survey='+SURVEY+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')
    Boost_b = np.loadtxt('./txtfiles/boosts/Boost_full_B_survey='+SURVEY+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')
    interp_B_a = scipy.interpolate.interp1d(np.log(rpvec), np.log(Boost_a))
    interp_B_b = scipy.interpolate.interp1d(np.log(rpvec), np.log(Boost_b))

    print("B_a 1 Mpc/h=", np.exp(interp_B_a(np.log(1.))))
    print("B_b 1 Mpc/h=", np.exp(interp_B_b(np.log(1.))))
	
    exit()

zLvec = np.loadtxt('./z_list_DESY1.txt')

# Import the correlation function, for the 2halo term.
xi_2h_mm = np.zeros((40000, len(zLvec)))
for zi in range(0,len(zLvec)):
    stringz = '{:1.12f}'.format(zLvec[zi])
    (r, xi_2h_mm[:, zi]) = np.loadtxt('./txtfiles/halofit_xi/xi2h_z='+stringz+'_'+endfile+'.txt', unpack=True)
xi = pa.bd* pa.bs * xi_2h_mm

# Get the comoving distance associated to the lens redshift
chi_vec = np.zeros(len(zLvec))
for zi in range(0,len(zLvec)):
    chi_vec[zi] = ccl.comoving_radial_distance(cosmo, 1./(1.+zLvec[zi])) * (pa.HH0/100.) # CCL returns in Mpc but we want Mpc/h

# Figure out the min and max value of the * positive * part of the vector of projected distances

minPiPos = 10**(-8)
max_int = 200. # We integrate the numerator out to 200 Mpc/h away from the lenses because the correlation is ~zero outside of this.
maxPiPos = max_int
Pi_pos = scipy.logspace(np.log10(minPiPos), np.log10(maxPiPos), 3000)
    
# Pi can be positive or negative, so now flip this and include the negative values, but only down to z=0
# And avoid including multiple of the same values - this messes up some integration routines.
Pi = [0]*len(zLvec)
chismin = ccl.comoving_radial_distance(cosmo, 1./(1.+pa.zsmin)) * (pa.HH0 / 100.) # CCL returns in Mpc but we want Mpc/h
for zi in range(0, len(zLvec)):
    Pi_pos_vec= list(Pi_pos)[1:]
    Pi_pos_vec.reverse()
    index_cut = next(j[0] for j in enumerate(Pi_pos_vec) if j[1]<=(chi_vec[zi]-chismin))
    Pi[zi] = np.append(-np.asarray(Pi_pos_vec[index_cut:]), np.append([0],Pi_pos))


# Get the correlation function in terms of Pi and rp 
xi_interp_r = [0] * len(zLvec)
xi_ofPi = [0] * len(zLvec)
for zi in range(0,len(zLvec)):
    xi_interp_r[zi] = scipy.interpolate.interp1d(np.log(r), xi[:, zi])
    xi_ofPi[zi] = np.zeros((len(rpvec), len(Pi[zi])))
    for ri in range(0,len(rpvec)):
        for pi in range(0,len(Pi[zi])):
            xi_ofPi[zi][ri, pi] = xi_interp_r[zi](np.log(np.sqrt(rpvec[ri]**2 + Pi[zi][pi]**2)))

    
# Get the vector of com dist values associated to Pi values:
com_Pi = [0]*len(zLvec)
z_Pi = [0]*len(zLvec)
for zi in range(0,len(zLvec)):
    com_Pi[zi] = chi_vec[zi] + Pi[zi]
    z_Pi[zi] = (1./ccl.scale_factor_of_chi(cosmo, com_Pi[zi] / (pa.HH0/100.))) - 1.  # CCL wants distance in Mpc but we are working in Mpc/h

# Now we have xi_{ls}(rp, Pi(z_s); z_L)

# Okay, now we do the required integrals:
# Load the weighted dNdz_mc for the two source samples:
z_a_full = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin0_zmc_centres.dat') #??
dNdz_a_weighted_fulll = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin0_zmc_weighted') #??
#and for b

    
#z_Pi_norm = scipy.linspace(pa.zsmin, pa.zsmax, 1000)   

# Get dNdz
#dNdz = [0] * len(zLvec)
#for zi in range(0, len(zLvec)):
#    dNdz[zi] = get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, z_Pi[zi])
    
#dNdz_norm = get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, z_Pi_norm)

# Interpolate to get the weighted dNdz in terms of the required z_Pi vectors at each z_l.
# Plot the weighted dNdz vs bin cents first to make sure they make sense!

"""# Do the integrals in spec-z
specint_num_a = np.zeros(((lenzph), len(zLvec), len(rpvec))); specint_num_b = np.zeros(((lenzph), len(zLvec), len(rpvec))); 
specint_denom_a = np.zeros(((lenzph), len(zLvec))); specint_denom_b = np.zeros(((lenzph), len(zLvec))); 
for j in range(0,len(zLvec)):
    print("zlj=", j)
    for i in range(0, lenzph):
        for ri in range(len(rpvec)):
            specint_num_a[i,j, ri] = scipy.integrate.simps(dNdz[j] * setup.p_z(z_a[j][i], z_Pi[j], pa.pzpar_fid, pa.pztype) * xi_ofPi[j][ri, :], z_Pi[j])
            specint_num_b[i,j, ri] = scipy.integrate.simps(dNdz[j] * setup.p_z(z_b[j][i], z_Pi[j], pa.pzpar_fid, pa.pztype) * xi_ofPi[j][ri, :], z_Pi[j])
            specint_num_asc[i,j, ri] = scipy.integrate.simps(dNdz[j] * setup.p_z(z_asc[j][i], z_Pi[j], pa.pzpar_fid, pa.pztype) * xi_ofPi[j][ri, :], z_Pi[j])
            specint_num_ascBl[i,j, ri] = scipy.integrate.simps(dNdz[j] * setup.p_z(z_ascBl[j][i], z_Pi[j], pa.pzpar_fid, pa.pztype) * xi_ofPi[j][ri, :], z_Pi[j])

            
for j in range(0,len(zLvec)):
    for i in range(0,lenzph):
        specint_denom_a[i,j] = scipy.integrate.simps(dNdz_norm * setup.p_z(z_a[j][i], z_Pi_norm, pa.pzpar_fid, pa.pztype), z_Pi_norm)
        specint_denom_b[i,j] = scipy.integrate.simps(dNdz_norm * setup.p_z(z_b[j][i], z_Pi_norm, pa.pzpar_fid, pa.pztype), z_Pi_norm)
        specint_denom_asc[i,j] = scipy.integrate.simps(dNdz_norm * setup.p_z(z_asc[j][i], z_Pi_norm, pa.pzpar_fid, pa.pztype), z_Pi_norm)
        specint_denom_ascBl[i,j] = scipy.integrate.simps(dNdz_norm * setup.p_z(z_ascBl[j][i], z_Pi_norm, pa.pzpar_fid, pa.pztype), z_Pi_norm)"""
    
# Now do the integrals in zmc
"""w_a = [0]*len(zLvec); w_b = [0]*len(zLvec); w_asc = [0]*len(zLvec); w_ascBl = [0]*len(zLvec)
for zi in range(0,len(zLvec)):
    w_a[zi] = weights(pa.e_rms_Bl_a,z_a[zi], zLvec)
    w_b[zi] = weights(pa.e_rms_Bl_b,z_b[zi], zLvec)
    w_asc[zi] = weights_shapes(pa.e_rms_a,z_asc[zi])
    w_ascBl[zi] = weights_shapes(pa.e_rms_a,z_ascBl[zi])"""

B_1_a = np.zeros((len(zLvec), len(rpvec)))
B_1_b = np.zeros((len(zLvec), len(rpvec)))
for zi in range(0,len(zLvec)):
    print("zi=", zi)
    for ri in range(len(rpvec)):
        # Instead of this go straight to dN_w/dz_mc * xi
        B_1_a[zi, ri]= scipy.integrate.simps(w_a[zi][:, zi] * specint_num_a[:, zi, ri], z_a[zi]) / scipy.integrate.simps(w_a[zi][:, zi]* specint_denom_a[:, zi], z_a[zi])
        B_1_b[zi, ri] = scipy.integrate.simps(w_b[zi][:, zi] * specint_num_b[:, zi, ri], z_b[zi]) / scipy.integrate.simps(w_b[zi][:, zi]* specint_denom_b[:, zi], z_b[zi])

# Now integrate these over zl
#dndzl = setup.get_dNdzL(zLvec, SURVEY)
# Load lens redshifts from file

Boost_a = np.zeros(len(rpvec))
Boost_b = np.zeros(len(rpvec))

for ri in range(0,len(rpvec)):
    Boost_a[ri] = scipy.integrate.simps(B_1_a[:, ri] * dndzl, zLvec)
    Boost_b[ri] = scipy.integrate.simps(B_1_b[:,ri]* dndzl, zLvec)

np.savetxt(Boost_file_a, Boost_a)
np.savetxt(Boost_file_b, Boost_b)

print("Boosts computed")

interp_B_a = scipy.interpolate.interp1d(rpvec, Boost_a)
interp_B_b = scipy.interpolate.interp1d(rpvec, Boost_b)

print("B_a 1 Mpc/h=", interp_B_a(1.))
print("B_b 1 Mpc/h=", interp_B_b(1.))




