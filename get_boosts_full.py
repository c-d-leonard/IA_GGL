# Script to get boosts automatically (rather than in an ipynb notebook)

import numpy as np
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import os.path

import shared_functions_setup as setup
import shared_functions_wlp_wls as shared
import pyccl as ccl

SURVEY = 'DESY1'
endfile = 'DESHoD'

if (SURVEY=='SDSS'):
    print("RUNNING WITH TEST VALUE FOR SPEC Z")
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

#rp_edges = setup.setup_rp_bins(pa.rp_min, pa.rp_max, pa.N_bins)
#rpvec = setup.rp_bins_mid(rp_edges)
	
# First check if we need to do this:
Boost_file_a = './txtfiles/boosts/Boost_full_A_survey='+SURVEY+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt'
Boost_file_b = './txtfiles/boosts/Boost_full_B_survey='+SURVEY+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt'
Boost_file_asc = './txtfiles/boosts/Boost_full_close_survey='+SURVEY+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt'
Boost_file_ascBl = './txtfiles/boosts/Boost_full_assocBl_survey='+SURVEY+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt'

if (os.path.isfile(Boost_file_a) and os.path.isfile(Boost_file_b) and os.path.isfile(Boost_file_asc) and os.path.isfile(Boost_file_ascBl)):
    print("The boost files have previously been calculated for this endfile.")
    Boost_a = np.loadtxt('./txtfiles/boosts/Boost_full_A_survey='+SURVEY+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')
    Boost_b = np.loadtxt('./txtfiles/boosts/Boost_full_B_survey='+SURVEY+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')
    Boost_asc = np.loadtxt('./txtfiles/boosts/Boost_full_close_survey='+SURVEY+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')
    interp_B_a = scipy.interpolate.interp1d(np.log(rpvec), np.log(Boost_a))
    interp_B_b = scipy.interpolate.interp1d(np.log(rpvec), np.log(Boost_b))
    interp_B_close = scipy.interpolate.interp1d(np.log(rpvec), np.log(Boost_asc))

    print("B_a 1 Mpc/h=", np.exp(interp_B_a(np.log(1.))))
    print("B_b 1 Mpc/h=", np.exp(interp_B_b(np.log(1.))))
    print("B_close 1 Mpc/h=", np.exp(interp_B_close(np.log(1.))))
	
    exit()
    
    
def sigma_e(z_s_):
    """ Returns a value for the model for the per-galaxy noise as a function of source redshift"""

    if hasattr(z_s_, "__len__"):
        sig_e = 2. / pa.S_to_N * np.ones(len(z_s_))
    else:
        sig_e = 2. / pa.S_to_N

    return sig_e

def get_SigmaC_inv(z_s_, z_l_):
    """ Returns the theoretical value of 1/Sigma_c, (Sigma_c = the critcial surface mass density).
    z_s_ and z_l_ can be 1d arrays, so the returned value will in general be a 2d array. """

    #com_s = chi_of_z(z_s_) 
    #com_l = chi_of_z(z_l_) 
    com_s = ccl.comoving_radial_distance(cosmo, 1./(1.+z_s_)) * (pa.HH0 / 100.) # CCL returns in Mpc but we want Mpc/h
    com_l = ccl.comoving_radial_distance(cosmo, 1./(1.+z_l_)) * (pa.HH0 / 100.) # CCL returns in Mpc but we want Mpc/h

    if ((hasattr(z_s_, "__len__")==True) and (hasattr(z_l_, "__len__")==True)):
        Sigma_c_inv = np.zeros((len(z_s_), len(z_l_)))
        for zsi in range(0,len(z_s_)):
            for zli in range(0,len(z_l_)):
                # Units are pc^2 / (h Msun), comoving
                if((com_s[zsi] - com_l[zli])<0.):
                    Sigma_c_inv[zsi, zli] = 0.
                else:
                    Sigma_c_inv[zsi, zli] = 4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc *   com_l[zli] * (com_s[zsi] - com_l[zli]) * (1 + z_l_[zli]) / com_s[zsi]
    else: 
        # Units are pc^2 / (h Msun), comoving
        if hasattr(z_s_, "__len__"):
            Sigma_c_inv = np.zeros(len(z_s_))
            for zsi in range(0, len(z_s_)):
                if(com_s[zsi]<=com_l):
                    Sigma_c_inv[zsi] = 0.
                else:
                    Sigma_c_inv[zsi] = 4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc *   com_l * (com_s[zsi] - com_l)* (1 + z_l_)/ com_s[zsi]
        elif hasattr(z_l_, "__len__"): 
            Sigma_c_inv = np.zeros(len(z_l_))
            for zli in range(0,len(z_l_)):
                if(com_s<=com_l[zli]):
                    Sigma_c_inv[zli] = 0.
                else:
                    Sigma_c_inv[zli] = 4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc *   com_l[zli] * (com_s - com_l[zli])* (1 + z_l_[zli]) / com_s
        else:
            if (com_s < com_l):
                Sigma_c_inv=0.
            else:
                 Sigma_c_inv= 4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc *   com_l * (com_s - com_l) * (1 + z_l_) / com_s
                    
    return Sigma_c_inv


def weights(e_rms, z_, z_l_):

    """ Returns the inverse variance weights as a function of redshift. """
        
    SigC_t_inv = get_SigmaC_inv(z_, z_l_)

    if ((hasattr(z_, "__len__")==True) and (hasattr(z_l_, "__len__")==True)):
        weights = np.zeros((len(z_), len(z_l_)))
        for zsi in range(0,len(z_)):
            for zli in range(0,len(z_l_)):
                weights[zsi, zli] = SigC_t_inv[zsi, zli]**2/(sigma_e(z_)[zsi]**2 + e_rms**2)
    else:
        if (hasattr(z_, "__len__")):
            weights = SigC_t_inv**2/(sigma_e(z_)**2 + e_rms**2 * np.ones(len(z_)))
        else:
            weights = SigC_t_inv**2/(sigma_e(z_)**2 + e_rms**2 )

    return weights

def weights_shapes(e_rms, z_):
    """ Returns the inverse variance weights as a function of redshift for tangential shear (not DS). """

    weights = 1./(sigma_e(z_)**2 + e_rms**2 * np.ones(len(z_)))

    return weights

def get_NofZ_unnormed(dNdzpar, dNdztype, z):
    """ Returns the dNdz of the sources as a function of photometric redshift, as well as the z points at which it is evaluated."""

    if (dNdztype == 'Nakajima'):
        # dNdz takes form like in Nakajima et al. 2011 equation 3
        a = dNdzpar[0]
        zs = dNdzpar[1]

        nofz_ = (z / zs)**(a-1.) * np.exp( -0.5 * (z / zs)**2)
    elif (dNdztype == 'Smail'):
        # dNdz take form like in Smail et al. 1994
        alpha = dNdzpar[0]
        z0 = dNdzpar[1]
        beta = dNdzpar[2]
        nofz_ = z**alpha * np.exp( - (z / z0)**beta)
    else:
        print("dNdz type "+str(dNdztype)+" not yet supported; exiting.")
        exit()

    return  nofz_

# UPDATE - use CCL functions for this instead    
# Set up interpolating functions for z(chi) and chi(z)
#(z_of_chi, chi_of_z) = setup.z_interpof_com(SURVEY)

zLvec = np.loadtxt('./z_list_DESY1.txt')
#zLvec = np.linspace(pa.zLmin, pa.zLmax, 100)

# Import the correlation function, from CLASS w/ halofit + FFTlog. This is for the 2halo term.
# Import also the 1-halo term for the correlation function.
# This is computed using the HOD's of the lens and source samples and FFTlog.
xi_2h_mm = np.zeros((40000, len(zLvec)))
xi_1h = np.zeros((40000, len(zLvec)))
for zi in range(0,len(zLvec)):
    stringz = '{:1.12f}'.format(zLvec[zi])
    #print("IMPORTING SPECIFIC XI TERMS FOR TEST")
    (r, xi_2h_mm[:, zi]) = np.loadtxt('./txtfiles/halofit_xi/xi2h_z='+stringz+'_'+endfile+'.txt', unpack=True)
    (r, xi_1h[:, zi]) = np.loadtxt('./txtfiles/xi_1h_terms/xi1h_ls_z='+stringz+'_'+endfile+'.txt', unpack=True)
    for ri in range(0,len(r)):
        if r[ri]>3:
            xi_1h[ri,zi] = 0.
xi_2h = pa.bd* pa.bs * xi_2h_mm

xi = xi_1h + xi_2h

# Get the comoving distance associated to the lens redshift
#chi_vec = chi_of_z(zLvec)
#cosmo = ccl.Cosmology(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), sigma8 = pa.sigma8, n_s=pa.n_s)
chi_vec = np.zeros(len(zLvec))
for zi in range(0,len(zLvec)):
    chi_vec[zi] = ccl.comoving_radial_distance(cosmo, 1./(1.+zLvec[zi])) * (pa.HH0/100.) # CCL returns in Mpc but we want Mpc/h

# Figure out the min and max value of the * positive * part of the vector of projected distances

minPiPos = 10**(-8)
max_int = 200. # We integrate the numerator out to 200 Mpc/h away from the lenses because the correlation is zero outside of this.
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


# Get the correlation function in terms of Pi at rp where we want it for using the power law
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
    #z_Pi[zi] = z_of_chi(com_Pi[zi])
    z_Pi[zi] = (1./ccl.scale_factor_of_chi(cosmo, com_Pi[zi] / (pa.HH0/100.))) - 1.  # CCL wants distance in Mpc but we are working in Mpc/h

# Now we have xi_{ls}(rp, Pi(z_s); z_L)

# Okay, now we do the required integrals:
# Define the z_ph vectors for the four subsamples we care about:
lenzph = 500
z_a = [0]*len(zLvec); z_b = [0]* len(zLvec); z_asc = [0] * len(zLvec); z_ascBl = [0] * len(zLvec)
for zi in range(0,len(zLvec)):
    z_a[zi] = scipy.linspace(zLvec[zi], zLvec[zi] +pa.delta_z, lenzph)
    z_b[zi] = scipy.linspace(zLvec[zi]+pa.delta_z, pa.zphmax, lenzph)
    # For the "assoc" sample we need to get the z-edges
    if (pa.close_cut<chi_vec[zi]):
        #zasc_min = z_of_chi(chi_vec[zi] - pa.close_cut)
        #zasc_max = z_of_chi(chi_vec[zi] + pa.close_cut)
        zasc_min = (1./ccl.scale_factor_of_chi(cosmo, (chi_vec[zi] - pa.close_cut) / (pa.HH0/100.) )) - 1.
        zasc_max = (1./ccl.scale_factor_of_chi(cosmo, (chi_vec[zi] + pa.close_cut) / (pa.HH0/100.) )) - 1.
    else:
        zasc_min = 0.
        #zasc_max = z_of_chi(chi_vec[zi] + pa.close_cut)
        zasc_max = (1./ccl.scale_factor_of_chi(cosmo, (chi_vec[zi] + pa.close_cut) / (pa.HH0/100.) )) - 1.
    z_asc[zi] = scipy.linspace(zasc_min, zasc_max, lenzph)
    
    if (pa.delta_z<zLvec[zi]):
        #zasc_minBl = z_of_chi(chi_vec[zi]) - pa.delta_z
        #zasc_maxBl = z_of_chi(chi_vec[zi]) + pa.delta_z
        zasc_minBl = ((1./ccl.scale_factor_of_chi(cosmo, chi_vec[zi] / (pa.HH0/100.) )) - 1.) - pa.delta_z
        zasc_maxBl = ((1./ccl.scale_factor_of_chi(cosmo, chi_vec[zi] / (pa.HH0/100.) )) - 1.) + pa.delta_z
    else:
        zasc_minBl = 0.
        zasc_maxBl = ((1./ccl.scale_factor_of_chi(cosmo, chi_vec[zi] / (pa.HH0/100.) )) - 1.) + pa.delta_z
    z_ascBl[zi] = scipy.linspace(zasc_minBl, zasc_maxBl, lenzph)
    print("zasc=", z_ascBl[zi])
    
z_Pi_norm = scipy.linspace(pa.zsmin, pa.zsmax, 1000)       

# Get dNdz
dNdz = [0] * len(zLvec)
for zi in range(0, len(zLvec)):
    dNdz[zi] = get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, z_Pi[zi])
    
dNdz_norm = get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, z_Pi_norm)

# Do the integrals in spec-z
specint_num_a = np.zeros(((lenzph), len(zLvec), len(rpvec))); specint_num_b = np.zeros(((lenzph), len(zLvec), len(rpvec))); specint_num_asc = np.zeros(((lenzph), len(zLvec), len(rpvec)))
specint_num_ascBl = np.zeros(((lenzph), len(zLvec), len(rpvec)))
specint_denom_a = np.zeros(((lenzph), len(zLvec))); specint_denom_b = np.zeros(((lenzph), len(zLvec))); specint_denom_asc = np.zeros(((lenzph), len(zLvec))); specint_denom_ascBl = np.zeros(((lenzph), len(zLvec)))
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
        specint_denom_ascBl[i,j] = scipy.integrate.simps(dNdz_norm * setup.p_z(z_ascBl[j][i], z_Pi_norm, pa.pzpar_fid, pa.pztype), z_Pi_norm)
    
# Now do the integrals in photo-z
w_a = [0]*len(zLvec); w_b = [0]*len(zLvec); w_asc = [0]*len(zLvec); w_ascBl = [0]*len(zLvec)
for zi in range(0,len(zLvec)):
    w_a[zi] = weights(pa.e_rms_Bl_a,z_a[zi], zLvec)
    w_b[zi] = weights(pa.e_rms_Bl_b,z_b[zi], zLvec)
    w_asc[zi] = weights_shapes(pa.e_rms_a,z_asc[zi])
    w_ascBl[zi] = weights_shapes(pa.e_rms_a,z_ascBl[zi])

B_1_a = np.zeros((len(zLvec), len(rpvec)))
B_1_b = np.zeros((len(zLvec), len(rpvec)))
B_1_asc = np.zeros((len(zLvec), len(rpvec)))
B_1_ascBl = np.zeros((len(zLvec), len(rpvec)))
for zi in range(0,len(zLvec)):
    print("zi=", zi)
    for ri in range(len(rpvec)):
        B_1_a[zi, ri]= scipy.integrate.simps(w_a[zi][:, zi] * specint_num_a[:, zi, ri], z_a[zi]) / scipy.integrate.simps(w_a[zi][:, zi]* specint_denom_a[:, zi], z_a[zi])
        B_1_b[zi, ri] = scipy.integrate.simps(w_b[zi][:, zi] * specint_num_b[:, zi, ri], z_b[zi]) / scipy.integrate.simps(w_b[zi][:, zi]* specint_denom_b[:, zi], z_b[zi])
        B_1_asc[zi, ri] = scipy.integrate.simps(w_asc[zi]*specint_num_asc[:, zi, ri], z_asc[zi]) / scipy.integrate.simps(w_asc[zi]* specint_denom_asc[:,zi], z_asc[zi])
        B_1_ascBl[zi, ri] = scipy.integrate.simps(w_ascBl[zi]*specint_num_ascBl[:, zi, ri], z_ascBl[zi]) / scipy.integrate.simps(w_ascBl[zi]* specint_denom_ascBl[:,zi], z_ascBl[zi])

# Now integrate these over zl
dndzl = setup.get_dNdzL(zLvec, SURVEY)

Boost_a = np.zeros(len(rpvec))
Boost_b = np.zeros(len(rpvec))
Boost_asc = np.zeros(len(rpvec))
Boost_ascBl = np.zeros(len(rpvec))

for ri in range(0,len(rpvec)):
    Boost_a[ri] = scipy.integrate.simps(B_1_a[:, ri] * dndzl, zLvec)
    Boost_b[ri] = scipy.integrate.simps(B_1_b[:,ri]* dndzl, zLvec)
    Boost_asc[ri]= scipy.integrate.simps(B_1_asc[:, ri]* dndzl, zLvec)
    Boost_ascBl[ri]= scipy.integrate.simps(B_1_ascBl[:, ri]* dndzl, zLvec)

np.savetxt(Boost_file_a, Boost_a)
np.savetxt(Boost_file_b, Boost_b)
np.savetxt(Boost_file_asc, Boost_asc)
np.savetxt(Boost_file_ascBl, Boost_ascBl)

print("Boosts computed")

interp_B_a = scipy.interpolate.interp1d(rpvec, Boost_a)
interp_B_b = scipy.interpolate.interp1d(rpvec, Boost_b)
interp_B_close = scipy.interpolate.interp1d(rpvec, Boost_asc)

print("B_a 1 Mpc/h=", interp_B_a(1.))
print("B_b 1 Mpc/h=", interp_B_b(1.))
print("B_close 1 Mpc/h=", interp_B_close(1.))




