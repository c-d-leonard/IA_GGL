# This is a script which predicts constraints on intrinsic alignments, using an updated version of the method of Blazek et al 2012 in which it is not assumed that only excess galaxies contribute to IA (rather it is assumed that source galaxies which are close to the lens along the line-of-sight can contribute.)

SURVEY = 'SDSS'
print("SURVEY=", SURVEY)

import numpy as np
import scipy
import scipy.integrate
import matplotlib.pyplot as plt
import scipy.interpolate
import shared_functions_setup as setup
import shared_functions_wlp_wls as ws
import pyccl as ccl
import os.path

np.set_printoptions(linewidth=240)
	
############## GENERIC FUNCTIONS ###############
	
def N_of_zph(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, dNdzpar, pzpar, dNdztype, pztype):
    """ Returns dNdz_ph, the number density in terms of photometric redshift, defined over photo-z range (z_a_def_ph, z_b_def_ph), normalized over the photo-z range (z_a_norm_ph, z_b_norm_ph), normalized over the spec-z range (z_a_norm, z_b_norm), and defined on the spec-z range (z_a_def, z_b_def)"""
	
    #print("z_a_def_s=", z_a_def_s, "z_b_def_s=", z_b_def_s)
    #print("z_a_norm_s=", z_a_norm_s, "z_b_norm_s=", z_b_norm_s)
	
    (z, dNdZ) = setup.get_NofZ_unnormed(dNdzpar, dNdztype, z_a_def_s, z_b_def_s, 1000, SURVEY)
	
    (z_norm, dNdZ_norm) = setup.get_NofZ_unnormed(dNdzpar, dNdztype, z_a_norm_s, z_b_norm_s, 1000, SURVEY)
	
    #print("z_a_def_ph=", z_a_def_ph, "z_b_def_ph=", z_b_def_ph)
    #print("z_a_norm_ph=", z_a_norm_ph, "z_b_norm_ph=", z_b_norm_ph)
	
    #save_dNdzs = np.column_stack((z, dNdZ))
    #np.savetxt('./txtfiles/photo_z_test/dNdzs_in_Nph_sample_A.txt', save_dNdzs)
	
    z_ph_vec = np.linspace(z_a_def_ph, z_b_def_ph, 1000)
    z_ph_vec_norm = np.linspace(z_a_norm_ph, z_b_norm_ph, 1000)
	
    int_dzs = np.zeros(len(z_ph_vec))
    for i in range(0,len(z_ph_vec)):
        int_dzs[i] = scipy.integrate.simps(dNdZ*setup.p_z(z_ph_vec[i], z, pzpar, pztype), z)
		
    int_dzs_norm = np.zeros(len(z_ph_vec_norm))
    for i in range(0,len(z_ph_vec_norm)):
        int_dzs_norm[i] = scipy.integrate.simps(dNdZ_norm*setup.p_z(z_ph_vec_norm[i], z_norm, pzpar, pztype), z_norm)
		
    norm = scipy.integrate.simps(int_dzs_norm, z_ph_vec_norm)
    #print("norm, pz=", norm)
	
    return (z_ph_vec, int_dzs / norm)

def sigma_e(z_s_):
	""" Returns a value for the model for the per-galaxy noise as a function of source redshift"""
		
	if hasattr(z_s_, "__len__"):
		sig_e = 2. / pa.S_to_N * np.ones(len(z_s_))
	else:
		sig_e = 2. / pa.S_to_N
			
	return sig_e
	
def weights(e_rms, z_):
	""" Returns the inverse variance weights as a function of redshift. """
	
	weights = (1./(sigma_e(z_)**2 + e_rms**2)) * np.ones(len(z_))
	
	return weights
    

################### THEORETICAL VALUES FOR FRACTIONAL ERROR CALCULATION ########################333
	
def sum_weights(photoz_sample, specz_cut, color_cut, rp_bins, dNdz_par, pz_par, dNdztype, pztype):
    """ Returns the sum over weights for each projected radial bin. 
    photoz_sample = 'A', 'B', or 'full'
    specz_cut = 'close', or 'nocut'
    """
	
    # Get lens redshift distribution
    zL = np.linspace(pa.zLmin, pa.zLmax, 100)
    dNdzL = setup.get_dNdzL(zL, SURVEY)
    #chiL = com_of_z(zL)
    chiL = ccl.comoving_radial_distance(cosmo_fid, 1./(1.+zL))
    chiSmin = ccl.comoving_radial_distance(cosmo_fid, 1./(1.+pa.zsmin))
    #if (min(chiL)> (pa.close_cut + com_of_z(pa.zsmin))):
    if (min(chiL)> (pa.close_cut + chiSmin)):
        #zminclose = z_of_com(chiL - pa.close_cut)
        zminclose = 1./(ccl.scale_factor_of_chi(cosmo_fid, chiL - pa.close_cut)) - 1.
    else:
        zminclose = np.zeros(len(chiL))
        for cli in range(0,len(chiL)):
            if (chiL[cli]>pa.close_cut + chiSmin):
                #zminclose[cli] = z_of_com(chiL[cli] - pa.close_cut)
                zminclose[cli] = 1./(ccl.scale_factor_of_chi(cosmo_fid, chiL[cli]-pa.close_cut))-1.
            else:
                zminclose[cli] = pa.zsmin
    #zmaxclose = z_of_com(chiL + pa.close_cut)
    zmaxclose = 1./(ccl.scale_factor_of_chi(cosmo_fid, chiL + pa.close_cut)) - 1.
	
    # Get norm, required for the color cut case:
    zph_norm = np.linspace(pa.zphmin, pa.zphmax, 1000)
    (zs_norm, dNdzs_norm) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 1000, SURVEY)
    zs_integral_norm = np.zeros(len(zph_norm))
    for zpi in range(0,len(zph_norm)):
        pz = setup.p_z(zph_norm[zpi], zs_norm, pa.pzpar_fid, pa.pztype)
        zs_integral_norm[zpi] = scipy.integrate.simps(pz * dNdzs_norm, zs_norm)
    norm = scipy.integrate.simps(zs_integral_norm, zph_norm)
	
    # Loop over lens redshift values
    sum_ans_zph = np.zeros(len(zL))
    for zi in range(0,len(zL)):

        if (color_cut=='all'):
            if (photoz_sample == 'A'):
			
                if (specz_cut == 'nocut'):
                    (z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi], zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
                    weight = weights(pa.e_rms_Bl_a, z_ph)
                    sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
                elif (specz_cut == 'close'):
                    (z_ph, dNdz_ph) = N_of_zph(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, zL[zi], zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
                    weight = weights(pa.e_rms_Bl_a, z_ph)
                    sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
                else:
                    print("We do not have support for that spec-z cut. Exiting.")
                    exit()
					
            elif(photoz_sample =='assocBl'):
                if (pa.delta_z<zL[zi]):
                    zminph = zL[zi] - pa.delta_z
                else:
                    zminph = 0.
						
                if (specz_cut == 'nocut'):
                    (z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zminph, zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
                    weight = weights(pa.e_rms_Bl_a, z_ph)
                    sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
                elif (specz_cut == 'close'):
                    (z_ph, dNdz_ph) = N_of_zph(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, zminph, zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
                    weight = weights(pa.e_rms_Bl_a, z_ph)
                    sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
                else:
                    print("We do not have support for that spec-z cut. Exiting.")
                    exit()
			
            elif(photoz_sample == 'B'):
			
                if (specz_cut == 'nocut'):
                    (z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi] + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
                    weight = weights(pa.e_rms_Bl_b, z_ph)
                    sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
                elif (specz_cut == 'close'):
                    (z_ph, dNdz_ph) = N_of_zph(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, zL[zi] + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
                    weight = weights(pa.e_rms_Bl_b, z_ph)
				
                    sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
                else:
                    print("We do not have support for that spec-z cut. Exiting.")
                    exit()
		
            elif(photoz_sample == 'full'):
			
                if (specz_cut == 'nocut'):
                    (z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
                    weight = weights(pa.e_rms_Bl_full, z_ph)
				
                    sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
                elif (specz_cut == 'close'):
                    (z_ph, dNdz_ph) = N_of_zph(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
                    weight = weights(pa.e_rms_Bl_full, z_ph)
				
                    sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
                else:
                    print("We do not have support for that spec-z cut. Exiting.")
                    exit()
            else:
                print("We do not have support for that photo-z sample cut. Exiting.")
                print(photoz_sample)
                exit()
	
        else:
            print("We do not have support for that color cut, exiting.")
            exit()
			
    # Now sum over lens redshift:
    sum_ans = scipy.integrate.simps(sum_ans_zph * dNdzL, zL)
	
    return sum_ans
	
def get_boost(rp_cents_, sample):
	"""Returns the boost factor in radial bins. propfact is a tunable parameter giving the proportionality constant by which boost goes like projected correlation function (= value at 1 Mpc/h). """

	#propfact = np.loadtxt('./txtfiles/boosts/Boost_'+str(sample)+'_gamt_survey='+str(SURVEY)+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')

	#Boost = propfact *(rp_cents_)**(-0.8) + np.ones((len(rp_cents_))) # Empirical power law fit to the boost, derived from the fact that the boost goes like projected correlation function.
	#if sample=='assocBl':
	#    Boost = np.loadtxt('./txtfiles/boosts/Boost_full_close_survey='+str(SURVEY)+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt') + np.ones((len(rp_cents_)))
	#else:
	Boost = np.loadtxt('./txtfiles/boosts/Boost_full_'+str(sample)+'_survey='+str(SURVEY)+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt') + np.ones((len(rp_cents_)))

	return Boost
	
def get_F(photoz_sample, rp_bins_, dNdz_par, pz_par, dNdztype, pztype):
	""" Returns F (the weighted fraction of lens-source pairs from the smooth dNdz which are contributing to IA) """

	# Sum over `rand-close'
	numerator = sum_weights(photoz_sample, 'close', 'all', rp_bins_, dNdz_par, pz_par, dNdztype, pztype)

	#Sum over all `rand'
	denominator = sum_weights(photoz_sample, 'nocut', 'all', rp_bins_, dNdz_par, pz_par, dNdztype, pztype)

	F = np.asarray(numerator) / np.asarray(denominator)

	return F

def get_SigmaC_inv(z_s_, z_l_):
    """ Returns the theoretical value of 1/Sigma_c, (Sigma_c = the critcial surface mass density).
    z_s_ and z_l_ can be 1d arrays, so the returned value will in general be a 2d array. """

    #com_s = com_of_z(z_s_) 
    #com_l = com_of_z(z_l_) 
    com_s = ccl.comoving_radial_distance(cosmo_fid, 1./(1.+z_s_))
    com_l = ccl.comoving_radial_distance(cosmo_fid, 1./(1.+z_l_))

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
    
def get_SigmaC_avg(photoz_sample):
    """ Get the average over Sigma C for the given sample.
    This is only used for the estimated SigmaCinv_avg_inv,
    this function is not called when converting Delta Sigma to gammat
    from pure lensing"""
			
    if(photoz_sample == 'B'):
        (z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, pa.zeff + pa.delta_z, pa.zphmax, pa.zeff + pa.delta_z, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
        
    elif(photoz_sample=='A'):
        (z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, pa.zeff, pa.zeff + pa.delta_z, pa.zeff, pa.zeff + pa.delta_z, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
        

    #print("z_ph=", z_ph)		
    Siginv = get_SigmaC_inv(z_ph, pa.zeff)
		
    Siginv_avg = scipy.integrate.simps(dNdz_ph * Siginv, z_ph)
    
    # testing
    #savedndz = np.column_stack((z_ph, dNdz_ph))
    #np.savetxt('./txtfiles/photo_z_test/dNdzph_'+photoz_sample+'.txt', savedndz)
	
    Sigavg =  1. / Siginv_avg
	
    return Sigavg
    
def get_DeltaSig_theory(rp_bins, rp_bins_c):
    """ Returns the theoretical value of Delta Sigma in bin using projection over the NFW profile and over the 2-pt correlation function at larger scales.

    We load correlation functions which have been computed externally using FFTlog; these are from power spectra that have already been averaged over the lens distribution. """
	
    ###### First get the term from halofit (valid at larger scales) ######
    # Import correlation functions, obtained via getting P(k) from CAMB OR CLASS and then using FFT_log, Anze Slozar version. 
    # Note that since CAMB / class uses comoving distances, all distances here should be comoving. rpvec and Pivec are in Mpc/h.	

    # Get a more well sampled rp, and Pi	
    rpvec 	= np.logspace(np.log10(0.00002), np.log10(rp_bins[-1]), 300)
    # Pivec a little more complicated because we want it log-spaced about zero
    Pi_neg = -np.logspace(np.log10(rpvec[0]), np.log10(500), 250)
    Pi_pos = np.logspace(np.log10(rpvec[0]), np.log10(500), 250)
    Pi_neg_list = list(Pi_neg)
    Pi_neg_list.reverse()
    Pi_neg_rev = np.asarray(Pi_neg_list)
    Pivec = np.append(Pi_neg_rev, Pi_pos)
	
    # Get rho_m in comoving coordinates (independent of redshift)
    rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h (comoving distances)
    rho_m = (pa.OmC + pa.OmB) * rho_crit # units of Msol h^2 / Mpc^3 (comoving distances)
		
    # Import the appropriate correlation function (already integrated over lens redshift distribution)
	
    if SURVEY=='SDSS':
        print("Loading single xi2h file for all dNdz tests - correct later.")
        r_hf, corr_hf = np.loadtxt('./txtfiles/halofit_xi/xi_2h_zavg_'+SURVEY+'_ext_theta.txt', unpack=True)
    elif SURVEY=='LSST_DESI':
        r_hf, corr_hf = np.loadtxt('./txtfiles/halofit_xi/xi_2h_zavg_'+SURVEY+'_ext_theta.txt', unpack=True)
    else:
        print("We don't have support for that survey, exiting.")
        exit()
		
    # Interpolate in 2D separations
    corr_hf_interp = scipy.interpolate.interp1d(r_hf, corr_hf)
    corr_2D_hf = np.zeros((len(rpvec), len(Pivec)))
    for ri in range(0, len(rpvec)):
        for pi in range(0, len(Pivec)):
            corr_2D_hf[ri, pi] = corr_hf_interp(np.sqrt(rpvec[ri]**2 + Pivec[pi]**2))
		
    # Get Sigma(r_p) for the 2halo term.
    Sigma_HF = np.zeros(len(rpvec))
    for ri in range(0,len(rpvec)):
        # This will have units Msol h / Mpc^2 in comoving distances.
        Sigma_HF[ri] = rho_m * scipy.integrate.simps(corr_2D_hf[ri, :], Pivec) 
		
        # Now average Sigma_HF(R) over R to get the first averaged term in Delta Sigma
        barSigma_HF = np.zeros(len(rpvec))
        for ri in range(0,len(rpvec)):
            barSigma_HF[ri] = 2. / rpvec[ri]**2 * scipy.integrate.simps(rpvec[0:ri+1]**2*Sigma_HF[0:ri+1], np.log(rpvec[0:ri+1]))
	
        # Units Msol h / Mpc^2 (comoving distances).
        DeltaSigma_HF = pa.bd*(barSigma_HF - Sigma_HF)
			
    ####### Now get the 1 halo term #######

    # Get the max R associated to our max M = 10**16
    Rmax = ws.Rhalo(10**16, SURVEY)
	
    # Import the 1halo correlation function from the power spectrum computed in get_Pkgm_1halo and fourier transformed using FFTlog. Already averaged over dndlz.
    print("Loading single xi1h file for all dNdz tests - correct later")
    r_1h, corr_1h = np.loadtxt('./txtfiles/xi_1h_terms/xigm_1h_'+SURVEY+'_ext_theta.txt', unpack=True)
	
    # Set xi_gg_1h to zero above Rmax Mpc/h.
    for ri in range(0, len(r_1h)):
        if (r_1h[ri]>Rmax):
            corr_1h[ri] = 0.0
	
    corr_1h_interp = scipy.interpolate.interp1d(r_1h, corr_1h)
	
    corr_2D_1h = np.zeros((len(rpvec), len(Pivec)))
    for ri in range(0, len(rpvec)):
        for pi in range(0, len(Pivec)):
            corr_2D_1h[ri, pi] = corr_1h_interp(np.sqrt(rpvec[ri]**2 + Pivec[pi]**2))
	
    Sigma_1h = np.zeros(len(rpvec))
    for ri in range(0,len(rpvec)):
        # Units Msol h / Mpc^2, comoving distances
        Sigma_1h[ri] = rho_m * scipy.integrate.simps(corr_2D_1h[ri, :], Pivec)
		
    #plt.figure()
    #plt.loglog(rpvec, Sigma_1h/ 10.**12, 'm+') # Plot in Msol h / pc^2.
    #plt.hold(True)
    #plt.loglog(rpvec, Sigma_HF/ 10.**12, 'g+')
    #plt.xlim(0.0003, 8)
    #plt.ylim(0.1,10**4)
    #plt.savefig('./plots/Sigma_1h.png')
    #plt.close()

	
    # Now average over R to get the first averaged term in Delta Sigma
    barSigma_1h = np.zeros(len(rpvec))
    for ri in range(0,len(rpvec)):
        barSigma_1h[ri] = 2. / rpvec[ri]**2 * scipy.integrate.simps(rpvec[0:ri+1]**2*Sigma_1h[0:ri+1], np.log(rpvec[0:ri+1]))
		
    DeltaSigma_1h = (barSigma_1h - Sigma_1h)
	
    """plt.figure()
    plt.loglog(rpvec, DeltaSigma_1h  / (10**12), 'g+', label='1-halo')
    plt.hold(True)
    plt.loglog(rpvec, DeltaSigma_HF  / (10**12), 'm+', label='halofit')
    plt.hold(True)
    plt.loglog(rpvec, (DeltaSigma_HF + DeltaSigma_1h)  / (10**12), 'k+', label='total')
    plt.xlim(0.05,20)
    plt.ylim(0.3,200)
    plt.xlabel('$r_p$, Mpc/h')
    plt.ylabel('$\Delta \Sigma$, $h M_\odot / pc^2$')
    plt.legend()
    plt.savefig('./plots/test_DeltaSigmatot_extzl_survey='+SURVEY+'.pdf')
    plt.close()"""
	
    # Interpolate and output at r_bins_c:
    ans_interp = scipy.interpolate.interp1d(rpvec, (DeltaSigma_1h + DeltaSigma_HF) / (10**12))
    ans = ans_interp(rp_bins_c)
	
    return ans # outputting as Msol h / pc^2
    
def get_gammat_purelensing(sample, limtype='pz'):
    """ Get gammat for a given photometric sample with only the lensing signal (not IA)"""
    
    # First get Delta Sigma, this is the same for all source samples
    DeltaSigma = get_DeltaSig_theory(rp_bins, rp_cent)
    
    # Now we need to get <Sigma_c^{-1}>^{-1}
    # This function supports setting the limits of this integration in terms of photo-z (closer to the real scenario) 
    # and in terms of spec-z / true-z (to cross check how much this matters)
    if limtype=='pz':
        # The limits are in terms of photo-z
        if sample=='A':
            zphmin = pa.zeff
            zphmax = pa.zeff+pa.delta_z
        elif sample=='B':
            zphmin = pa.zeff+pa.delta_z
            zphmax = pa.zphmax
        else:
            ValueError("We don't support that sample for the calculation of gammat from pure lensing.")
        
        # Set up two vectors of spec-z limits over which we will integrate
        zsi = np.linspace(pa.zsmin, pa.zsmax, 1000)
        zsf = np.linspace(pa.zsmin, pa.zsmax, 1000)
        
        # For each of these limits in both cases, call the dNdz and get the integral
        
        inner_integral = np.zeros((len(zsi), len(zsf)))
        for i in range(0,len(zsi)):
            print("zsi=", zsi[i])
            for f in range(0,len(zsf)):
                
                (zs, dNdzs) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zsi[i], zsf[f], 1000, SURVEY)
                
                Siginv = get_SigmaC_inv(zs, pa.zeff)
                
                int_num_temp = scipy.integrate.simps(dNdzs*Siginv, zs)
                int_norm = scipy.integrate.simps(dNdzs,zs)
                
                inner_integral[i,f] = int_num_temp / int_norm
                
        # Now integrate the spec-z limit over the photo-z uncertainty function in both cases:
        pzi = setup.p_z(zphmin, zsi, pa.pzpar_fid, pa.pztype)
        int_pz_1 = np.zeros((len(zsf)))
        for f in range(0,len(zsf)):
            int_pz_1[f] = scipy.integrate.simps(pzi*inner_integral[:,f], zsi)
        
        psf = setup.p_z(zphmax, zsf, pa.pzpar_fid, pa.pztype)
        SigInv_avg = scipy.integrate.simps(pzf*int_pz_1, zsf)
        print("SigInv_avg=", 1./SigInv_avg)
                  
    elif limtype=='truez':
        # The limits are in terms of spec-z
        if sample=='A':
            zsmin = pa.zeff
            zsmax = pa.zeff+pa.delta_z
        elif sample=='B':
            zsmin = pa.zeff+pa.delta_z
            zsmax = pa.zphmax
        else:
            ValueError("We don't support that sample for the calculation of gammat from pure lensing.")
        
        (zs, dNdzs) = setup.get_NofZ_unnormed(pa.dNdzpar_true, pa.dNdztype, zsmin, zsmax, 1000, SURVEY) 
        
        #print("zs=", zs)
        
        Siginv = get_SigmaC_inv(zs, pa.zeff)
        
        int_num = scipy.integrate.simps(Siginv*dNdzs, zs)
        int_norm = scipy.integrate.simps(dNdzs, zs)
        #print("norm, zs=", int_norm)
        
        # testing
        #save_dndz = np.column_stack((zs, dNdzs/int_norm))
        #np.savetxt('./txtfiles/photo_z_test/dNdzs_'+sample+'.txt', save_dndz)
        
        SigInv_avg = int_num / int_norm  
        print("SigInv_avg inv=", 1./SigInv_avg) 
           
    else:
        raise ValueError("We don't have support for that type of limit on the pure lensing integral.")
        
    
    gammat_lens = DeltaSigma * SigInv_avg
    
    # save answer
    save_gammat = np.column_stack((rp_cent, gammat_lens))
    np.savetxt('./txtfiles/photo_z_test/gammat_lens_'+sample+'_'+limtype+'_'+SURVEY+'_'+endfile+'.dat', save_gammat)
    
    return gammat_lens
		
def get_gammaIA_estimator():
    """ Calculate gammaIA from the estimator used on data for the Blazek et al. 2012 + F method with gammat, as in Sara's project. """
    
    # Get F factors
    F_a = get_F('A', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
    F_b = get_F('B', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
    
    print("F_a=", F_a)
    print("F_b=", F_b)
    
    # Write to file:
    np.savetxt('./txtfiles/photo_z_test/F_a_'+SURVEY+'_'+endfile+'.txt', [F_a])
    np.savetxt('./txtfiles/photo_z_test/F_b_'+SURVEY+'_'+endfile+'.txt', [F_a])

    # Load boosts
    B_a = get_boost(rp_cent, 'A')
    B_b = get_boost(rp_cent, 'B')
    
    print("B_a=", B_a)
    print("B_b=", B_b)
    
    # Write to file:
    np.savetxt('./txtfiles/photo_z_test/B_a_'+SURVEY+'_'+endfile+'.txt', B_a)
    np.savetxt('./txtfiles/photo_z_test/B_b_'+SURVEY+'_'+endfile+'.txt', B_b)
    
    # Get SigmaC
    SigA = get_SigmaC_avg('A')
    SigB = get_SigmaC_avg('B')
    
    print("Sigma_c_inv_avg_inv A=", SigA)
    print("Sigma_c_inv_avg_inv B=", SigB)
    
    # Write to file:
    np.savetxt('./txtfiles/photo_z_test/SigmaC_a_'+SURVEY+'_'+endfile+'.txt', [SigA])
    np.savetxt('./txtfiles/photo_z_test/SigmaC_b_'+SURVEY+'_'+endfile+'.txt', [SigB])
    
    # Get theoretical lensing-only gammat
    gammat_a = get_gammat_purelensing('A', limtype='truez')
    gammat_b = get_gammat_purelensing('B', limtype='truez')
    
    
    
    # Assemble estimator
    gamma_IA_est = (gammat_b * SigB - gammat_a*SigA) / ( (B_b - 1 + F_b)*SigB - (B_a - 1 + F_a)*SigA)
    
    # Stack rp or theta with gamma_IA_est to output
    save_gammaIA = np.column_stack((rp_cent, gamma_IA_est))
    np.savetxt('./txtfiles/photo_z_test/gamma_IA_est_'+SURVEY+'_'+endfile+'.txt', save_gammaIA)

    return


######## MAIN CALLS ##########

# Import the parameter file:
if (SURVEY=='SDSS'):
    import params_SDSS_testpz as pa
elif (SURVEY=='LSST_DESI'):
	import params_LSST_DESI as pa
else:
	print("We don't have support for that survey yet; exiting.")
	exit()
	
endfile = 'dndz_'+str(pa.percent_change)+'percent'

print("endfile=", endfile)
print("dNdztruepar=", pa.dNdzpar_true)

	
# Set up the fiducial cosmology object
cosmo_fid = ccl.Cosmology(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), sigma8 = pa.sigma8, n_s=pa.n_s)

# Set up projected bins

# Option to provide theta min and theta max and convert to rp for a given effective lens redshift:
theta_min = 0.1
theta_max = 200
rp_min = setup.arcmin_to_rp(theta_min, pa.zeff,cosmo_fid)
rp_max = setup.arcmin_to_rp(theta_max, pa.zeff,cosmo_fid)
print("rp_min=", rp_min, "rp_max=", rp_max)

rp_bins 	= 	setup.setup_rp_bins(rp_min, rp_max, pa.N_bins)
rp_cent	=	setup.rp_bins_mid(rp_bins)

get_gammaIA_estimator()

# Get dNdz's for plotting
(zs_true, dNdzs_true) = setup.get_NofZ_unnormed(pa.dNdzpar_true, pa.dNdztype, 0., 5.0, 1000, SURVEY)
norm = scipy.integrate.simps(dNdzs_true, zs_true)

save_dNdz_true = np.column_stack((zs_true, dNdzs_true/norm))
np.savetxt('./txtfiles/photo_z_test/dNdzs_true_'+endfile+'.txt', save_dNdz_true)

(zs_fid, dNdzs_fid) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, 0., 5.0, 1000, SURVEY)
norm_fid = scipy.integrate.simps(dNdzs_fid, zs_fid)

save_dNdz_fid = np.column_stack((zs_fid, dNdzs_fid/norm_fid))
np.savetxt('./txtfiles/photo_z_test/dNdzs_fid.txt', save_dNdz_fid)



