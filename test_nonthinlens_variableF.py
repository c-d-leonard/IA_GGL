# This is a script which predicts constraints on intrinsic alignments, using an updated version of the method of Blazek et al 2012 in which it is not assumed that only excess galaxies contribute to IA (rather it is assumed that source galaxies which are close to the lens along the line-of-sight can contribute.)

SURVEY = 'DESY1'
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

################### THEORETICAL VALUES FOR FRACTIONAL ERROR CALCULATION ########################333
	
def sum_weights_DESY1(source_sample, z_cut):
    """ Returns the sum over weights for each projected radial bin. 
    photoz_sample = 'A', 'B', or 'full'
    specz_cut = 'close', or 'nocut'
    """

    # Load lens redshift distribution from file
    zL, dNdzL = np.loadtxt('./txtfiles/'+pa.dNdzL_file, unpack=True)
    
    # Use the assumed parameters because this is for the cut on F which is an analysis choice we make.
    chiL = ccl.comoving_radial_distance(cosmo_a, 1./(1.+zL)) * (pa.HH0_a / 100.) # CCL returns in Mpc but we want Mpc/h
	
    # Load weighted source redshift distributions
    
    if (source_sample == 'A'):    
        # Load the weighted dNdz_mc for source sample A:
        z_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin0_zmc_centres.dat')
        dNdz_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin0_zmc_weighted')
        							
    elif(source_sample == 'B'): 
        # Load the weighted dNdz_mc for source sample B:
        z_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_centres.dat')
        dNdz_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_weighted')
            
    else:
        print("We do not have support for that z sample cut. Exiting.")
        exit()
    
    
    """
    if (source_sample == 'A'):    
        # Load the weighted dNdz_mc for source sample A:
        print("Using perturbed source redshift distribution in sum_weights for F for sample A")
        z_mc, dNdz_mc = setup.dNdz_perturbed(source_sample, sigmaz, deltaz)
        							
    elif(source_sample == 'B'): 
        # Load the weighted dNdz_mc for source sample B:
        z_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_centres.dat')
        dNdz_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_weighted')
            
    else:
        print("We do not have support for that z sample cut. Exiting.")
        exit()
    """

    if z_cut=='nocut':
    
        sum_lens = scipy.integrate.simps(dNdzL, zL)
        sum_source = scipy.integrate.simps(dNdz_mc, z_mc)
        sum_ans = sum_lens*sum_source
        
    elif z_cut=='close':
    
        # We implement here a r_p dependent projection for F where for r_p< 2 Mpc we integrate out to 2 Mpc and for r_p>2 Mpc we integrate to r_p
        sum_ans = np.zeros(len(rp_cent))
        for ri in range(0,len(rp_cent)):
            if rp_cent[ri] <= 2.0 * (pa.HH0_t / 100.): # (use true value of h for unit conversion because that is what we used to convert from theta)
                proj_len = 2.0 * (pa.HH0_t / 100.)
            else:
                proj_len = rp_cent[ri]
            print("rp=", rp_cent[ri], "proj=", proj_len) 
              
            chiSmin = ccl.comoving_radial_distance(cosmo_a, 1./(1.+min(z_mc))) * (pa.HH0_a / 100.) # CCL returns in Mpc but we want Mpc/h
            if (min(chiL)> (proj_len + chiSmin)):
                zminclose = 1./(ccl.scale_factor_of_chi(cosmo_a, (chiL - proj_len)/(pa.HH0_a / 100.))) - 1.
            else:
                zminclose = np.zeros(len(chiL))
                for cli in range(0,len(chiL)):
                    if (chiL[cli]>proj_len + chiSmin):
                        zminclose[cli] = 1./(ccl.scale_factor_of_chi(cosmo_a, (chiL[cli]-proj_len)/(pa.HH0_a / 100.)))-1.
                    else:
                        zminclose[cli] = min(z_mc)

            zmaxclose = 1./(ccl.scale_factor_of_chi(cosmo_a, (chiL + pa.close_cut)/(pa.HH0_a / 100.))) - 1.  
           
            sum_close = np.zeros(len(zL))
            for zi in range(0,len(zL)):
                indlow=next(j[0] for j in enumerate(z_mc) if j[1]>=(zminclose[zi]))
                indhigh=next(j[0] for j in enumerate(z_mc) if j[1]>=(zmaxclose[zi]))
            	
                sum_close[zi] = scipy.integrate.simps(dNdz_mc[indlow:indhigh], z_mc[indlow:indhigh])
					
            # Now sum over lens redshift:
            sum_ans[ri] = scipy.integrate.simps(sum_close * dNdzL, zL)
	
    return sum_ans
    
	
def get_boost(theta_vec, sample):
	"""Returns the boost factor in radial bins. propfact is a tunable parameter giving the proportionality constant by which boost goes like projected correlation function (= value at 1 Mpc/h). """

	#Boost = np.loadtxt('./txtfiles/boosts/Boost_'+str(sample)+'_survey='+str(SURVEY)+'_'+endfile+'.txt') + np.ones((len(theta_vec)))
	
	#print("Loading boost from previous run")
	#Boost = np.loadtxt('./txtfiles/boosts/Boost_'+str(sample)+'_survey='+str(SURVEY)+'_true-redshifts-different_sigma='+str(pa.sigma)+'deltaz='+str(pa.del_z)+'.txt') + np.ones((len(theta_vec)))
	Boost = np.loadtxt('./txtfiles/boosts/Boost_'+str(sample)+'_survey='+str(SURVEY)+'_with1halo.txt') + np.ones((len(theta_vec)))

	return Boost
	
def get_F(photoz_sample):
	""" Returns F (the weighted fraction of lens-source pairs from the smooth dNdz which are contributing to IA) """

	# Sum over `rand-close'
	numerator = sum_weights_DESY1(photoz_sample, 'close')

	#Sum over all `rand'
	denominator = sum_weights_DESY1(photoz_sample, 'nocut')

	F = np.asarray(numerator) / np.asarray(denominator)

	return F

def get_SigmaC_inv(z_s_, z_l_, cosmo_, HH0_):
    """ Returns the theoretical value of 1/Sigma_c, (Sigma_c = the critcial surface mass density).
    z_s_ and z_l_ can be 1d arrays, so the returned value will in general be a 2d array. """
    
    # We need these in units of Mpc/h but CCL returns units of Mpc
    com_s = ccl.comoving_radial_distance(cosmo_, 1./(1.+z_s_)) * (HH0_/100.)
    com_l = ccl.comoving_radial_distance(cosmo_, 1./(1.+z_l_)) * (HH0_/100.)

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

    
    # Load weighted source distributions			
    if(photoz_sample == 'B'):
        z_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_centres.dat')
        dNdz_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_weighted')
        
    elif(photoz_sample=='A'):
        z_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin0_zmc_centres.dat')
        dNdz_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin0_zmc_weighted')
        
    """if(photoz_sample == 'B'):
        z_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_centres.dat')
        dNdz_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_weighted')
        
    elif(photoz_sample=='A'):
        print("Using perturbed source redshift distribution in SigmaC_avg")
        z_mc, dNdz_mc = setup.dNdz_perturbed(photoz_sample, sigmaz, deltaz)"""
        
    norm_mc = scipy.integrate.simps(dNdz_mc, z_mc)  
    
    # Load lens distribution:
    #zL, dNdzL = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/z_dNdz_lenses_subbin.dat', unpack=True)
    zL, dNdzL = np.loadtxt('./txtfiles/'+pa.dNdzL_file, unpack=True)
    norm_L = scipy.integrate.simps(dNdzL, zL)
	
    Siginv = get_SigmaC_inv(z_mc, zL, cosmo_a, pa.HH0_a)
    
    Siginv_zL = np.zeros(len(zL))
    for zi in range(0,len(zL)):
        Siginv_zL[zi] = scipy.integrate.simps(dNdz_mc*Siginv[:,zi], z_mc) / norm_mc
    
    #np.savetxt('./txtfiles/siginv_zl_avg_debug_'+photoz_sample+'.txt', Siginv_zL)
    		
    Siginv_avg = scipy.integrate.simps(dNdzL * Siginv_zL, zL) / norm_L
    
    # testing
    #savedndz = np.column_stack((z_ph, dNdz_ph))
    #np.savetxt('./txtfiles/photo_z_test/dNdzph_'+photoz_sample+'.txt', savedndz)
	
    Sigavg =  1. / Siginv_avg
	
    return Sigavg
    
def get_DeltaSig_theory():
    """ Returns the theoretical value of Delta Sigma in bin using projection over the NFW profile and over the 2-pt correlation function at larger scales.

    We load correlation functions which have been computed externally using FFTlog; these are from power spectra that have already been averaged over the lens distribution. """
	
    ###### First get the term from halofit (valid at larger scales) ######
    # Import correlation functions, obtained via getting P(k) from CAMB OR CLASS and then using FFT_log, Anze Slozar version. 
    # Note that since CAMB / class uses comoving distances, all distances here should be comoving. rpvec and Pivec are in Mpc/h.	

    # Get a more well sampled rp, and Pi	
    rpvec 	= np.logspace(np.log10(0.00002), np.log10(rp_max), 300)
    # Pivec a little more complicated because we want it log-spaced about zero
    Pi_neg = -np.logspace(np.log10(rpvec[0]), np.log10(500), 250)
    Pi_pos = np.logspace(np.log10(rpvec[0]), np.log10(500), 250)
    Pi_neg_list = list(Pi_neg)
    Pi_neg_list.reverse()
    Pi_neg_rev = np.asarray(Pi_neg_list)
    Pivec = np.append(Pi_neg_rev, Pi_pos)
	
    # Get rho_m in comoving coordinates (independent of redshift)
    rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h (comoving distances)
    rho_m = (pa.OmC_t + pa.OmB) * rho_crit # units of Msol h^2 / Mpc^3 (comoving distances)
    
    # Import z-list
    zL = np.loadtxt('./txtfiles/z_list_DESY1.txt')
    
    DeltaSigma_centbins = np.zeros((len(theta_radians), len(zL)))
    for zi in range(len(zL)):
        print("Delta_Sig_theory, zi=", zL[zi])
        zload=str('{:1.12f}'.format(zL[zi]))
        # Import the appropriate correlation function
        r_hf, corr_hf_2h = np.loadtxt('./txtfiles/halofit_xi/xi2h_z='+zload+'_with1halo.txt', unpack=True)
        r_hf, corr_1h = np.loadtxt('./txtfiles/xi_1h_terms/xi1h_ls_z='+zload+'_with1halo.txt', unpack=True)
        for ri in range(0,len(r_hf)):
            if r_hf[ri]>3:
                corr_1h[ri] = 0.
        corr_hf = corr_hf_2h + corr_1h	
	
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
			
        ans_interp = scipy.interpolate.interp1d(rpvec, (DeltaSigma_HF) / (10**12))
        
        chi = ccl.comoving_radial_distance(cosmo_t, 1./(1.+zL[zi])) * (pa.HH0_t / 100.) # CCL returns in Mpc but we want Mpc/h
        # Now use theta instead of rp.
        DeltaSigma_centbins[:,zi] = ans_interp(theta_radians *chi) # comoving distance chi at THIS redshift.
	
    return DeltaSigma_centbins # outputting as Msol h / pc^2
    
def get_DeltaSig_theory_zavg(rp_bins, rp_bins_c):
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
    rho_m = (pa.OmC_t + pa.OmB) * rho_crit # units of Msol h^2 / Mpc^3 (comoving distances)
		
    # Import the appropriate correlation function (already integrated over lens redshift distribution)

    r_hf, corr_hf = np.loadtxt('./txtfiles/halofit_xi/xi_2h_zavg_'+SURVEY+'_'+endfile+'.txt', unpack=True)	
		
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
	
    # Interpolate and output at r_bins_c:
    ans_interp = scipy.interpolate.interp1d(rpvec, (DeltaSigma_HF) / (10**12))
    
    zL, dNdzL = np.loadtxt('./txtfiles/'+pa.dNdzL_file, unpack=True)
    norm_L = scipy.integrate.simps(dNdzL, zL)
    zeff = scipy.integrate.simps(dNdzL*zL, zL)/ norm_L
    print(zeff)
    chi = ccl.comoving_radial_distance(cosmo_t, 1./(1.+zeff)) * (pa.HH0_t / 100.) # CCL returns in Mpc but we want Mpc/h
    # Now use theta instead of rp.
    ans = ans_interp(theta_radians*chi)
	
    return ans # outputting as Msol h / pc^2
    
def get_gammat_purelensing(DeltaSigma, sample, limtype='pz'):
    """ Get gammat for a given photometric sample with only the lensing signal (not IA)"""
    
    # Now we need to get <Sigma_c^{-1}>^{-1}
    # This function supports setting the limits of this integration in terms of photo-z (closer to the real scenario) 
    # and in terms of spec-z / true-z (to cross check how much this matters)
    if limtype=='pz': # NOT UPDATED FOR DES REDSHIFT DISTRIBUTIONS
        """print(" the version of pz limits is not updated to allow for des y1 input redshift distributions")
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
        print("SigInv_avg=", 1./SigInv_avg)"""
        print("we don't have support for pz limits actually, sorry!")
                  
    elif limtype=='truez':
        # The limits are in terms of spec-z
     
        
        if(sample == 'B'):
            z_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_centres.dat')
            dNdz_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_weighted')
        
        elif(sample=='A'):
            z_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin0_zmc_centres.dat')
            dNdz_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin0_zmc_weighted')
            
        #print("Using perturbed source redshift distribution for gammat pure lensing")
        #z_mc, dNdz_mc = setup.dNdz_perturbed(sample, pa.sigma, pa.del_z)
        
        norm_mc = scipy.integrate.simps(dNdz_mc, z_mc)   
    
        # Load lens distribution:
        #zL, dNdzL = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/z_dNdz_lenses_subbin.dat', unpack=True)
        zL, dNdzL = np.loadtxt('./txtfiles/'+pa.dNdzL_file, unpack=True)
        norm_L = scipy.integrate.simps(dNdzL, zL)
        
        Siginv = get_SigmaC_inv(z_mc, zL, cosmo_t, pa.HH0_t)
        #print("sample="+sample+" Sig=", 1./Siginv)
        
        Siginv_zL = np.zeros(len(zL))
        for zi in range(len(zL)):
            Siginv_zL[zi] = scipy.integrate.simps(Siginv[:, zi]*dNdz_mc, z_mc) / norm_mc
            #Siginv_zL[zi] = scipy.integrate.simps(dNdz_mc, z_mc) / norm_mc
        #print("Siginv=", 1./Siginv_zL)
        
        #np.savetxt('./txtfiles/siginv_zl_gammat_debug_'+sample+'.txt', Siginv_zL)
           
    else:
        raise ValueError("We don't have support for that type of limit on the pure lensing integral.")
        
    #int_Siginv_zL = scipy.integrate.simps(Siginv_zL*dNdzL, zL) / norm_L
    #print("sample="+sample+",Sigavg in gammat=", 1./int_Siginv_zL)
        
    gammat_lens = np.zeros(len(theta_vec))
    for ti in range(len(theta_vec)):
        #gammat_lens[ri] = scipy.integrate.simps(Siginv_zL * dNdzL, zL) / norm_L
        gammat_lens[ti] = scipy.integrate.simps(DeltaSigma[ti,:] * Siginv_zL * dNdzL, zL) / norm_L
        #gammat_lens[ri] = scipy.integrate.simps(DeltaSigma[ri,:] * dNdzL, zL) / norm_L
    
    # save answer
    save_gammat = np.column_stack((theta_vec, gammat_lens))
    np.savetxt('./txtfiles/photo_z_test/gammat_lens_'+sample+'_'+limtype+'_'+SURVEY+'_'+endfile+'.dat', save_gammat)
    
    return gammat_lens
    
def get_fred(sample):
    """ This function returns the zl- and zs- averaged red fraction for the given sample."""
	
    #zL = np.linspace(pa.zLmin, pa.zLmax, 100)
    zL, dNdzL = np.loadtxt('./txtfiles/'+pa.dNdzL_file, unpack=True)
	
    if(sample == 'B'):
        zs = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_centres.dat')
        dNdz_s = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_weighted')
        
    elif(sample=='A'):
        zs = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin0_zmc_centres.dat')
        dNdz_s = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin0_zmc_weighted')

    # for the fiducial value of gamma_IA, we only want to consider f_red in the range of redshifts around the lenses which are subject to IA, so set up the cuts for that.
    # this is for a fiducial signal so use the "true" parameters.
    chiLmin = ccl.comoving_radial_distance(cosmo_t, 1./(1.+min(zL))) * (pa.HH0_t / 100.) # CCL returns in Mpc but we want Mpc/h
    chiSmin = ccl.comoving_radial_distance(cosmo_t, 1./(1.+min(zs))) * (pa.HH0_t / 100.) # CCL returns in Mpc but we want Mpc/h
    chiL = ccl.comoving_radial_distance(cosmo_t, 1./(1.+zL)) * (pa.HH0_t / 100.) # CCL returns in Mpc but we want Mpc/h
	
    if (chiLmin> (pa.close_cut + chiSmin)):
        aminclose = ccl.scale_factor_of_chi(cosmo_t, chiL-pa.close_cut)
        zminclose = 1./aminclose - 1.
        #zminclose = z_of_com(com_of_z(zL) - pa.close_cut)
    else:
        zminclose = np.zeros(len(zL))
        for zli in range(0,len(zL)):
            if (chiL[zli]>(pa.close_cut + chiSmin)):
                aminclose = ccl.scale_factor_of_chi(cosmot_t, chiL-pa.close_cut)
                zminclose = 1./aminclose - 1.
                #zminclose[zli] = z_of_com(com_of_z(zL[zli]) - pa.close_cut)
            else:
                zminclose[zli] = min(zs)
	
    amaxclose = ccl.scale_factor_of_chi(cosmo_t, chiL + pa.close_cut)
    zmaxclose = 1./amaxclose - 1.
    #zmaxclose = z_of_com(chiL + pa.close_cut)			
    # Check the max z for which we have kcorr and ecorr corrections	
    (z_k, kcorr, x,x,x) = np.loadtxt('./txtfiles/kcorr.dat', unpack=True)
    (z_e, ecorr, x,x,x) = np.loadtxt('./txtfiles/ecorr.dat', unpack=True)
    zmaxke = min(max(z_k), max(z_e))	
    for cli in range(0,len(zL)):
        if (zmaxclose[cli]>zmaxke):
            zmaxclose[cli] = zmaxke
	
    fred_of_zL = np.zeros(len(zL))
    #ans2 = np.zeros(len(zL))
    #norm = np.zeros(len(zL))
    for zi in range(0, len(zL)):
        print("zi=", zi)
	
        # Cut the zs and dNdzs at zminclose and zmaxclose
        #indmin = next(j[0] for j in enumerate(zs) if j[1]>zminclose[zi])
        #indmax = next(j[0] for j in enumerate(zs) if j[1]>zmaxclose[zi])
        #print("indmin=", indmin)
        #print("indmax=", indmax)
        
        # Get a version of dNdzs more well sampled in the region we care about:
        zs_cut = np.linspace(zminclose[zi], zmaxclose[zi], 100)
        print("zminclose=", zminclose[zi])
        print("zmaxclose=", zmaxclose[zi])
        interp_dNdzs = scipy.interpolate.interp1d(zs, dNdz_s)
        dNdzs_cut = interp_dNdzs(zs_cut)
        
        #zs_cut = zs[indmin:indmax]
        #dNdzs_cut = dNdz_s[indmin:indmax]
        norm_zs = scipy.integrate.simps(dNdzs_cut, zs_cut)

        #(zs, dNdzs_unnormed) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zminclose[zi], zmaxclose[zi], 500, SURVEY)
        #norm_zs = scipy.integrate.simps(dNdzs_unnormed, zs)
        #dNdzs = dNdzs_unnormed / norm_zs
        fred=  setup.get_fred_ofz(zs_cut, SURVEY)
		
        # Average over dNdzs
        fred_of_zL[zi] = scipy.integrate.simps(fred*dNdzs_cut, zs_cut) / norm_zs
		
    fred_avg = scipy.integrate.simps(dNdzL * fred_of_zL, zL) / scipy.integrate.simps(dNdzL, zL)
    
    print("fred_avg=", fred_avg)
    fred_save = [0]
    fred_save[0] = fred_avg
    np.savetxt("./txtfiles/photo_z_test/fred_"+sample+".txt", fred_save)
	
    return fred_avg
    
def get_IA_gammat_term(sample, F, Boost):
    """ Returns gammaIA *not per contributing galaxy* such that it can be incorporated in gammat:
    gammat = gammat_pure_lensing + gammaIA_not_per_galay 
    as part of the modelling of the estimator."""
    
    # Get or load red fraction:
    fred = np.loadtxt("./txtfiles/photo_z_test/fred_"+sample+"_"+SURVEY+".txt")
    
    # Load the projected correlations functions calculated elsewhere:
    # This assumes the same rp values have been used elsewhere as in this file, be careful.
    #rp, wgp = np.loadtxt('./txtfiles/photo_z_test/wgp2h_'+sample+'_'+SURVEY+'_'+endfile+'.txt', unpack=True)
    #rp, wgg = np.loadtxt('./txtfiles/photo_z_test/wgg2h_'+sample+'_'+SURVEY+'_'+endfile+'.txt', unpack=True)
    rp, wgp = np.loadtxt('./txtfiles/photo_z_test/wgp2h_'+sample+'_'+SURVEY+'.txt', unpack=True)
    rp, wgg = np.loadtxt('./txtfiles/photo_z_test/wgg2h_'+sample+'_'+SURVEY+'.txt', unpack=True)
    
    gamma_IA_not_per_galaxy = (fred*wgp / (wgg + 2* pa.close_cut))* (Boost-1. + F)
    gamma_IA_fiducial_per_gal = (fred*wgp / (wgg + 2* pa.close_cut))
    
    plt.figure()
    plt.loglog(theta_vec, gamma_IA_fiducial_per_gal, 'o')
    plt.xlabel('$\\theta$, arcmin')
    plt.ylabel('IA contribution to $\gamma_t$')
    plt.ylim(10**(-8), 10**(-2))
    #plt.title('sources bin 0')
    plt.title('Injected gammaIA')
    plt.savefig('./test_gammaIA_fid_pergal_'+sample+'.png')
    plt.close()
    
    #save_gammaIA = np.column_stack((theta_vec, gamma_IA_not_per_galaxy))
    #np.savetxt('./txtfiles/photo_z_test/gamma_IA_not_per_gal_'+sample+'_'+SURVEY+'.txt', save_gammaIA)
    
    return gamma_IA_not_per_galaxy
		
def get_gammaIA_estimator():
    """ Calculate gammaIA from the estimator used on data for the Blazek et al. 2012 + F method with gammat, as in Sara's project. """
    
    # Get F factors
    F_a = get_F('A')
    F_b = get_F('B')
    
    print("F_a=", F_a)
    print("F_b=", F_b)
    
    # Write to file:
    np.savetxt('./txtfiles/photo_z_test/F_a_'+SURVEY+'_'+endfile+'.txt', [F_a])
    np.savetxt('./txtfiles/photo_z_test/F_b_'+SURVEY+'_'+endfile+'.txt', [F_b])

    # Load boosts
    B_a = get_boost(theta_vec, 'A')
    B_b = get_boost(theta_vec, 'B')
    
    print("B_a=", B_a)
    print("B_b=", B_b)
    
    # Write to file:
    #np.savetxt('./txtfiles/photo_z_test/B_a_'+SURVEY+'_'+endfile+'.txt', B_a)
    #np.savetxt('./txtfiles/photo_z_test/B_b_'+SURVEY+'_'+endfile+'.txt', B_b)
    
    # Get SigmaC
    SigA = get_SigmaC_avg('A')
    SigB = get_SigmaC_avg('B')
    
    print("Sigma_c_inv_avg_inv A=", SigA)
    print("Sigma_c_inv_avg_inv B=", SigB)
    
    # Write to file:
    np.savetxt('./txtfiles/photo_z_test/SigmaC_a_'+SURVEY+'_'+endfile+'.txt', [SigA])
    np.savetxt('./txtfiles/photo_z_test/SigmaC_b_'+SURVEY+'_'+endfile+'.txt', [SigB])
    
    #print("before delta sigma theory")
    # First get Delta Sigma, this is the same for all source samples
    #DeltaSigma = get_DeltaSig_theory()
    #print("after delta sigma theory")
    #np.savetxt('./txtfiles/DeltaSigma_with1halo.txt', DeltaSigma)
    #exit()
    print("Loading Delta Sigma from previous run")
    DeltaSigma = np.loadtxt('./txtfiles/DeltaSigma_with1halo.txt')
    
    # Get theoretical lensing-only gammat
    gammat_a_lens = get_gammat_purelensing(DeltaSigma, 'A', limtype='truez')
    gammat_b_lens = get_gammat_purelensing(DeltaSigma, 'B', limtype='truez')
    
    """
    print("Get gamma IA for fiducial")
    gamma_IA_A = get_IA_gammat_term("A", F_a, B_a)
    gamma_IA_B = get_IA_gammat_term("B", F_b, B_b)
    
    plt.figure()
    plt.loglog(theta_vec, gamma_IA_A, 'o', label='IA')
    plt.loglog(theta_vec, gammat_a_lens, 'o', label='lensing')
    plt.title('Contribution to $\gamma_t$ from lensing and IA, bin0')
    plt.ylabel('$\gamma_x$')
    plt.xlabel('$\\theta$, arcmin')
    plt.legend()
    plt.savefig('./gamma_t_contributions_A.png')
    plt.close()
    
    plt.figure()
    plt.loglog(theta_vec, gamma_IA_B, 'o', label='IA')
    plt.loglog(theta_vec, gammat_b_lens, 'o', label='lensing')
    plt.title('Contribution to $\gamma_t$ from lensing and IA, bin1')
    plt.ylabel('$\gamma_x$')
    plt.xlabel('$\\theta$, arcmin')
    plt.legend()
    plt.savefig('./gamma_t_contributions_B.png')
    plt.close()
    
    gammat_a = gammat_a_lens + gamma_IA_A
    gammat_b = gammat_b_lens + gamma_IA_B"""
    
    gammat_a = gammat_a_lens
    gammat_b = gammat_b_lens
    
    # Assemble estimator
    gamma_IA_est = (gammat_b * SigB - gammat_a*SigA) / ( (B_b - 1 + F_b)*SigB - (B_a - 1 + F_a)*SigA)
    
    # Stack rp or theta with gamma_IA_est to output
    #numerator = gammat_b * SigB - gammat_a*SigA
    #save_numerator= np.column_stack((theta_vec, numerator))
    #np.savetxt('./txtfiles/photo_z_test/numerator_'+SURVEY+'_'+endfile+'.txt', save_numerator)
    
    # Stack rp or theta with gamma_IA_est to output
    save_gammaIA = np.column_stack((theta_vec, gamma_IA_est))
    np.savetxt('./txtfiles/photo_z_test/gamma_IA_est_'+SURVEY+'_'+endfile+'.txt', save_gammaIA)
    
    """# Load the version with the true dNdzL for our distribution to see the difference:
    #rp_load, gamma_IA_load = np.loadtxt('./txtfiles/photo_z_test/gamma_IA_est_DESY1_test.txt', unpack=True)
    
    plt.figure()
    plt.loglog(theta_vec, gamma_IA_est, 'o', color='tab:orange')
    #plt.loglog(rp_load, gamma_IA_load, 'o', label='true dNdzL')
    plt.ylim(10**(-6), 10**(-3))
    plt.xlabel('$\\theta$, arcmin')
    plt.title('True data dNdzL')
    plt.ylabel('$\gamma_{IA}$ estimated in pure lensing case')
    plt.savefig('./gamma_IA_purelensing_test_angular_projection.png')
    plt.close()"""

    return
    
def test_thin_lens_approx(sample):
    """ Check if the approximation of pulling Delta Sigma outside the integral over lens redshift holds. """
    
    # Load source and lens distributions
    if(sample == 'B'):
        z_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_centres.dat')
        dNdz_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_weighted')
        
    elif(sample=='A'):
        z_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin0_zmc_centres.dat')
        dNdz_mc = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin0_zmc_weighted')
    
    norm_mc = scipy.integrate.simps(dNdz_mc, z_mc)    
    
    # Load lens distribution:
    #zL, dNdzL = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/z_dNdz_lenses_subbin.dat', unpack=True)
    zL, dNdzL = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/z_dNdz_lenses.dat', unpack=True)
    norm_L = scipy.integrate.simps(dNdzL, zL)
    
    # Get Sigma crit inverse as a function of zL and zs (need this for both cases)
    Siginv = get_SigmaC_inv(z_mc, zL, cosmo_t)
    
    # Integrate this over the source distribution (need this for both cases)
    Siginv_zL = np.zeros(len(zL))
    for zi in range(len(zL)):
        Siginv_zL[zi] = scipy.integrate.simps(Siginv[:, zi]*dNdz_mc, z_mc) / norm_mc
        
    # Without the approximation:
    
    # Load Delta Sigma as a function of lens redshift and rp because this takes ages to compute
    DeltaSigma = np.loadtxt('./txtfiles/DeltaSigma_with1halo.txt')
    
    no_approx = np.zeros(len(rp_cent))
    for ri in range(len(rp_cent)):
        no_approx[ri] = scipy.integrate.simps(DeltaSigma[ri,:] * Siginv_zL * dNdzL, zL) / norm_L
        
    # Now with the approximation
    
    Delta_Sig_zavg = get_DeltaSig_theory_zavg(rp_bins, rp_cent)
    
    with_approx = Delta_Sig_zavg * scipy.integrate.simps(Siginv_zL * dNdzL, zL) / norm_L
    
    frac_diff = np.abs(no_approx - with_approx) / np.abs(no_approx)
    
    plt.figure()
    plt.loglog(rp_cent, no_approx, 'o', label='No approx')
    plt.loglog(rp_cent, with_approx, 'o',label='Thin lens approx')
    plt.legend()
    plt.xlabel('$r_p$, Mpc/h')
    plt.ylabel('$\gamma_t$, pure lensing')
    plt.savefig('./thin_lens_approx_sourcebin='+sample+'.png')
    plt.close()
    
    plt.figure()
    plt.semilogx(rp_cent, frac_diff,'o')
    plt.xlabel('$r_p$, Mpc/h')
    plt.ylabel('Frac diff to $\gamma_t$ from thin lens approx')
    plt.savefig('./frac_diff_thin_lens_approx_sourcebin='+sample+'.png')
    plt.close()
   
    return


######## MAIN CALLS ##########

# Import the parameter file:
if (SURVEY=='SDSS'):
    import params_SDSS_testpz as pa
elif (SURVEY=='LSST_DESI'):
	import params_LSST_DESI as pa
elif (SURVEY=='DESY1'):
	import params_DESY1_testpz as pa
else:
	print("We don't have support for that survey yet; exiting.")
	exit()

#sigz = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1] #np.linspace(0.01,0.1, 11)
#delz =  [-0.18, -0.17, -0.16, -0.15, -0.14, -0.13, -0.12, -0.11, -0.10, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, 
#         -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1] #np.linspace(-0.18, 0.1, 29)

        
endfile = 'no_fidIA_assumedpar_OmM='+str(pa.OmC_a+pa.OmB)+'_HH0='+str(pa.HH0_a)
	
# Set up the 'true' and 'assumed' cosmology objects.
#'true' parameters feed into gammat, boost. 'assumed' parameters feed into the distances which go into calculating sigma_crit and F.
cosmo_t = ccl.Cosmology(Omega_c = pa.OmC_t, Omega_b = pa.OmB, h = (pa.HH0_t/100.), sigma8 = pa.sigma8, n_s=pa.n_s)
cosmo_a = ccl.Cosmology(Omega_c = pa.OmC_a, Omega_b = pa.OmB, h = (pa.HH0_a/100.), sigma8 = pa.sigma8, n_s=pa.n_s)

# Set up projected bins

# Option to provide theta min and theta max and convert to rp for a given effective lens redshift:
theta_vec = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/theta_mid.txt')
# Using 'true' parameters here because I am only changing to rp for convenience. Change back before reporting anything.
rp_cent = setup.arcmin_to_rp(theta_vec, pa.zeff,cosmo_t)
rp_max = setup.arcmin_to_rp(200, pa.zeff, cosmo_t)
print(rp_max)

theta_radians = theta_vec / 60.*np.pi/180.


get_gammaIA_estimator()
        

exit()

# Get dNdz's for plotting
(zs_true, dNdzs_true) = setup.get_NofZ_unnormed(pa.dNdzpar_true, pa.dNdztype, 0., 5.0, 1000, SURVEY)
norm = scipy.integrate.simps(dNdzs_true, zs_true)

save_dNdz_true = np.column_stack((zs_true, dNdzs_true/norm))
np.savetxt('./txtfiles/photo_z_test/dNdzs_true_'+endfile+'.txt', save_dNdz_true)

(zs_fid, dNdzs_fid) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, 0., 5.0, 1000, SURVEY)
norm_fid = scipy.integrate.simps(dNdzs_fid, zs_fid)

save_dNdz_fid = np.column_stack((zs_fid, dNdzs_fid/norm_fid))
np.savetxt('./txtfiles/photo_z_test/dNdzs_fid.txt', save_dNdz_fid)



