# This is a script which predicts constraints on intrinsic alignments, using an updated version of the method of Blazek et al 2012 in which it is not assumed that only excess galaxies contribute to IA (rather it is assumed that source galaxies which are close to the lens along the line-of-sight can contribute.)

SURVEY = 'LSST_DESI'
print "SURVEY=", SURVEY
endfile = 'fixDls'

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
	
	(z, dNdZ) = setup.get_NofZ_unnormed(dNdzpar, dNdztype, z_a_def_s, z_b_def_s, 200, SURVEY)
	
	(z_norm, dNdZ_norm) = setup.get_NofZ_unnormed(dNdzpar, dNdztype, z_a_norm_s, z_b_norm_s, 200, SURVEY)
	
	z_ph_vec = scipy.linspace(z_a_def_ph, z_b_def_ph, 200)
	z_ph_vec_norm = scipy.linspace(z_a_norm_ph, z_b_norm_ph, 200)
	
	int_dzs = np.zeros(len(z_ph_vec))
	for i in range(0,len(z_ph_vec)):
		int_dzs[i] = scipy.integrate.simps(dNdZ*setup.p_z(z_ph_vec[i], z, pzpar, pztype), z)
		
	int_dzs_norm = np.zeros(len(z_ph_vec_norm))
	for i in range(0,len(z_ph_vec_norm)):
		int_dzs_norm[i] = scipy.integrate.simps(dNdZ_norm*setup.p_z(z_ph_vec_norm[i], z_norm, pzpar, pztype), z_norm)
		int_dzs_norm[i] = scipy.integrate.simps(dNdZ_norm*setup.p_z(z_ph_vec_norm[i], z_norm, pzpar, pztype), z_norm)
		
	norm = scipy.integrate.simps(int_dzs_norm, z_ph_vec_norm)
	
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

#### THIS IS AN OLD FUNCTIONS WHICH COMPUTES THE SHAPE-NOISE ONLY COVARIANCE IN REAL SPACE #####
# I'm just keeping this to compare with its output

def shapenoise_cov(photoz_samp, rp, dNdzpar, pzpar, dNdztype, pztype):
	""" Returns a diagonal covariance matrix in bins of projected radius for a measurement dominated by shape noise. Elements are 1 / (sum_{ls} w), carefully normalized, in each bin. """
	
	# Get the area of each projected radial bin in square arcminutes
	bin_areas       =       setup.get_areas(rp, pa.zeff, SURVEY) # We're just going to use the mean redshift for this.
	
	# Define the vector of lens redshifts
	zL = scipy.linspace(pa.zLmin, pa.zLmax, 200)
	dndzl = setup.get_dNdzL(zL, SURVEY)
	
	# We have the effective source surface density, n_s, over the full distribution of sources. For each lens redshift, there is a different subset of these sources that contribute to neff. Get the corresponding neff as a function of zL
	ns_eff = np.zeros(len(zL))
	for zi in range(0,len(zL)):
		
		# We require two different normalizations for Nofzph - one for getting the partial neff and another for averaging over SigmaC
		if (photoz_samp =='A'):
			(z_ph_ns, Nofzph_ns) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi], zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
			e_rms = pa.e_rms_Bl_a
		elif (photoz_samp == 'B'):
			(z_ph_ns, Nofzph_ns) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi] + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
			e_rms = pa.e_rms_Bl_b
		elif (photoz_samp == 'full'):
			# This is sort of pointless because if we are using the full sample we expect to recover the full sample value but including for completeness
			(z_ph_ns, Nofzph_ns) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
			e_rms = pa.e_rms_Bl_full
		elif (photoz_samp == 'src'):
			(z_ph_ns, Nofzph_ns) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi], pa.zphmax, pa.zphmin, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
			e_rms = pa.e_rms_Bl_full
		elif (photoz_samp == 'assocBl'):
			if (pa.delta_z<zL[zi]):
				zphmin = zL[zi] - pa.delta_z
			else:
				zphmin = 0.
			(z_ph_ns, Nofzph_ns) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zphmin, zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
			e_rms = pa.e_rms_Bl_a	
				
		else:
			print "We do not have support for that sample. Exiting."
		
			
		ns_eff[zi] = pa.n_s * scipy.integrate.simps(Nofzph_ns, z_ph_ns)
		
	ns_avg = scipy.integrate.simps(dndzl * ns_eff, zL)

	cov = e_rms**2 / ( ns_avg * pa.n_l * pa.Area_l * bin_areas  )
	
	return cov

################### THEORETICAL VALUES FOR FRACTIONAL ERROR CALCULATION ########################333
	
def sum_weights(photoz_sample, specz_cut, color_cut, rp_bins, dNdz_par, pz_par, dNdztype, pztype):
	""" Returns the sum over weights for each projected radial bin. 
	photoz_sample = 'A', 'B', or 'full'
	specz_cut = 'close', or 'nocut'
	"""
	
	# Get lens redshift distribution
	zL = np.linspace(pa.zLmin, pa.zLmax, 100)
	dNdzL = setup.get_dNdzL(zL, SURVEY)
	chiL = com_of_z(zL)
	if (min(chiL)> (pa.close_cut + com_of_z(pa.zsmin))):
		zminclose = z_of_com(chiL - pa.close_cut)
	else:
		zminclose = np.zeros(len(chiL))
		for cli in range(0,len(chiL)):
			if (chiL[cli]>pa.close_cut + com_of_z(pa.zsmin)):
				zminclose[cli] = z_of_com(chiL[cli] - pa.close_cut)
			else:
				zminclose[cli] = pa.zsmin
	zmaxclose = z_of_com(chiL + pa.close_cut)
	
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
					print "We do not have support for that spec-z cut. Exiting."
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
					print "We do not have support for that spec-z cut. Exiting."
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
					print "We do not have support for that spec-z cut. Exiting."
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
					print "We do not have support for that spec-z cut. Exiting."
					exit()
			else:
				print "We do not have support for that photo-z sample cut. Exiting."
				print photoz_sample
				exit()
	
		else:
			print "We do not have support for that color cut, exiting."
			exit()
			
	# Now sum over lens redshift:
	sum_ans = scipy.integrate.simps(sum_ans_zph * dNdzL, zL)
	
	return sum_ans
	
def get_boost(rp_cents_, sample):
	"""Returns the boost factor in radial bins. propfact is a tunable parameter giving the proportionality constant by which boost goes like projected correlation function (= value at 1 Mpc/h). """

	#propfact = np.loadtxt('./txtfiles/boosts/Boost_'+str(sample)+'_gamt_survey='+str(SURVEY)+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')

	#Boost = propfact *(rp_cents_)**(-0.8) + np.ones((len(rp_cents_))) # Empirical power law fit to the boost, derived from the fact that the boost goes like projected correlation function.
	if sample=='assocBl':
	    Boost = np.loadtxt('./txtfiles/boosts/Boost_full_close_survey='+str(SURVEY)+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt') + np.ones((len(rp_cents_)))
	else:
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
		
def get_fred():
	""" This function returns the zl- and zph- averaged red fraction for the given sample."""
	
	zL = np.linspace(pa.zLmin, pa.zLmax, 100)

	# for the fiducial value of gamma_IA, we only want to consider f_red in the range of redshifts around the lenses which are subject to IA, so set up the spectroscopic cuts for that.
	if (com_of_z(pa.zLmin)> (pa.close_cut + com_of_z(pa.zsmin))):
		zminclose = z_of_com(com_of_z(zL) - pa.close_cut)
	else:
		zminclose = np.zeros(len(zL))
		for zli in range(0,len(zL)):
			if (com_of_z(zL[zli])>(pa.close_cut + com_of_z(pa.zsmin))):
				zminclose[zli] = z_of_com(com_of_z(zL[zli]) - pa.close_cut)
			else:
				zminclose[zli] = pa.zsmin
	
	chiL = 	com_of_z(zL)
	zmaxclose = z_of_com(chiL + pa.close_cut)			
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
		print "zi=", zi
		
		(zs, dNdzs_unnormed) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zminclose[zi], zmaxclose[zi], 500, SURVEY)
		norm_zs = scipy.integrate.simps(dNdzs_unnormed, zs)
		dNdzs = dNdzs_unnormed / norm_zs
		fred=  setup.get_fred_ofz(zs, SURVEY)
		
		# Average over dNdzs
		fred_of_zL[zi] = scipy.integrate.simps(fred*dNdzs, zs)
		
		"""if (photoz_samp=='A'):
			
			zph = np.linspace(zL[zi], zL[zi]+pa.delta_z, 500)
			
		elif (photoz_samp == 'B'):
			zph = np.linspace(zL[zi]+ pa.delta_z,pa.zphmax, 500)
		elif (photoz_samp == 'A+B'):
			# This is the only one we should be using.
			zph = np.linspace(zL[zi],pa.zphmax, 500)	
		elif (photoz_sampe =='full'):
			zph = np.linspace(pa.zphmin, pa.zphmax, 500)
		else:
			print "That photo-z cut is not supported; exiting."
			exit()"""
		
		#ans1 = np.zeros(len(zph))
		#norm1 = np.zeros(len(zph))
		#for zpi in range(0,len(zph)):
		#	pz = setup.p_z(zph[zpi], zs, pa.pzpar_fid, pa.pztype)
		#	ans1[zpi] = scipy.integrate.simps(pz * dNdzs * fred_of_z, zs)
		#	norm1[zpi] = scipy.integrate.simps(pz * dNdzs, zs)
		#ans2[zi] = scipy.integrate.simps(ans1, zph)
		#norm[zi] = scipy.integrate.simps(norm1, zph)
		
	dndzl = setup.get_dNdzL(zL, SURVEY)
	
	#fred_avg = scipy.integrate.simps(dndzl * ans2 / norm, zL)
	fred_avg = scipy.integrate.simps(dndzl * fred_of_zL, zL)
	
	return fred_avg

def get_SigmaC_inv(z_s_, z_l_):
    """ Returns the theoretical value of 1/Sigma_c, (Sigma_c = the critcial surface mass density).
    z_s_ and z_l_ can be 1d arrays, so the returned value will in general be a 2d array. """

    com_s = com_of_z(z_s_) 
    com_l = com_of_z(z_l_) 

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

def get_DeltaSig_theory(rp_bins, rp_bins_c):
	""" Returns the theoretical value of Delta Sigma in bin using projection over the NFW profile and over the 2-pt correlation function at larger scales, rather than using the lensing-related definition.
	
	We load correlation functions which have been computed externally using FFTlog; these are from power spectra that have already been averaged over the lens distribution. """
	
	###### First get the term from halofit (valid at larger scales) ######
	# Import correlation functions, obtained via getting P(k) from CAMB OR CLASS and then using FFT_log, Anze Slozar version. 
	# Note that since CAMB / class uses comoving distances, all distances here should be comoving. rpvec and Pivec are in Mpc/h.	
	
	# Get a more well sampled rp, and Pi	
	rpvec 	= scipy.logspace(np.log10(0.00002), np.log10(rp_bins[-1]), 300)
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
			
	####### Now get the 1 halo term #######

	# Get the max R associated to our max M = 10**16
	Rmax = ws.Rhalo(10**16, SURVEY)
	
	# Import the 1halo correlation function from the power spectrum computed in get_Pkgm_1halo and fourier transformed using FFTlog. Already averaged over dndlz.
	r_1h, corr_1h = np.loadtxt('./txtfiles/xi_1h_terms/xigm_1h_'+SURVEY+'_'+endfile+'.txt', unpack=True)
	
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

def get_gammat_theory(photoz_sample):
	""" Get some version of gammat theoretical for each sample. 
	WE ASSUME A SINGLE LENS REDSHIFT HERE BECAUSE WE ARE ONLY USING THIS FOR ROUGH DEBUGGING."""
	
	if(photoz_sample =='assocBl'):
		if (pa.delta_z<pa.zeff):
			zminph = pa.zeff - pa.delta_z
		else:
			zminph = 0.
						
		(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zminph, pa.zeff + pa.delta_z, pa.zphmin, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
			
	elif(photoz_sample == 'B'):

		(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, pa.zeff + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
		
	Siginv = get_SigmaC_inv(z_ph, pa.zeff)
		
	Siginv_avg = scipy.integrate.simps(dNdz_ph * Siginv, z_ph)
	
	gammat = DeltaSig_the * Siginv_avg
	
	return gammat
	
def get_SigmaC_avg(photoz_sample):
	""" Get the average over Sigma C for the given sample.
	WE ASSUME A SINGLE LENS REDSHIFT HERE BECAUSE WE ARE ONLY USING THIS FOR ROUGH DEBUGGING."""
	
	if(photoz_sample =='assocBl'):
		if (pa.delta_z<pa.zeff):
			zminph = pa.zeff - pa.delta_z
		else:
			zminph = 0.
						
		(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zminph, pa.zeff + pa.delta_z, pa.zphmin, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
			
	elif(photoz_sample == 'B'):

		(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, pa.zeff + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
		
	Siginv = get_SigmaC_inv(z_ph, pa.zeff)
		
	Siginv_avg = scipy.integrate.simps(dNdz_ph * Siginv, z_ph)
	
	Sigavg =  1. / Siginv_avg
	
	return Sigavg
	
			
##### ERRORS FOR FRACTIONAL ERROR CALCULATION #####
	
def get_gammaIA_cov(rp_bins, rp_bins_c):
	""" Takes information about the uncertainty on constituent elements of gamma_{IA} and combines them to get the covariance matrix of gamma_{IA} in projected radial bins."""
	""" We are only interested right now in the diagonal elements of the covariance matrix, so we assume it is diagonal. """ 

	# Get the fiducial quantities required for the statistical error variance terms. #	
	Boost_a = get_boost(rp_cent, 'assocBl')
	Boost_b = get_boost(rp_cent, 'B')
	
	# Run F, cz, and SigIA, this takes a while so we don't always do it.
	if pa.run_quants == True:
	
		################ F's #################
	
		F_assoc_fid = get_F('assocBl', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
		F_b_fid = get_F('B', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
		print "F_assoc_fid=", F_assoc_fid , "F_b_fid=", F_b_fid
	
		save_F = np.column_stack(([F_assoc_fid], [F_b_fid]))
		np.savetxt('./txtfiles/F/F_assocfid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', save_F)
		
	############ gamma_IA ###########
		
	if pa.run_quants==False :
		# Load stuff if we haven't computed it this time around:
		(F_assoc_fid, F_b_fid) = np.loadtxt('./txtfiles/F/F_assocfid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', unpack=True)
	
	############ Get statistical error ############
	
	# Import the covariance matrix of Delta Sigma for each sample as calculated from Fourier space in a different script, w / cosmic variance terms
	gamt_assocBl_import_CV = np.loadtxt('./txtfiles/covmats/cov_gamtBl_zLext_'+SURVEY+'_sample=assocBl_rpts2000_lpts100000_deltaz='+str(pa.delta_z)+'.txt')
	gamt_B_import_CV = np.loadtxt('./txtfiles/covmats/cov_gamtBl_zLext_'+SURVEY+'_sample=B_rpts2000_lpts100000_deltaz='+str(pa.delta_z)+'.txt')
	
	# Uncomment the following section to plot comparison of diagonal covariance elements against shape-noise only real space case.
	"""gamt_assocBl_import_CV  = np.diag(gamtCov_assoc)
	gamt_B_import_CV = np.diag(gamtCov_b)"""
	
	SigCavg_asc = get_SigmaC_avg('assocBl')
	SigCavg_b = get_SigmaC_avg('B')
	
	
	"""plt.figure()
	plt.loglog(rp_bins_c, gamtCov_assoc, 'mo', label='shape noise: real')
	plt.hold(True)
	plt.loglog(rp_bins_c, np.diag(gamt_assocBl_import_CV), 'go', label='with CV: Fourier')
	plt.xlabel('r_p')
	plt.ylabel('Variance')
	plt.legend()
	plt.savefig('./plots/check_gamtBl_'+SURVEY+'_assoc.pdf')
	plt.close()
	
	plt.figure()
	plt.loglog(rp_bins_c, gamtCov_b, 'mo', label='shape noise: real')
	plt.hold(True)
	plt.loglog(rp_bins_c, np.diag(gamt_B_import_CV), 'bo', label='with CV: Fourier')
	plt.xlabel('r_p')
	plt.ylabel('Variance')
	plt.legend()
	plt.savefig('./plots/check_gamtBl_'+SURVEY+'_B.pdf')"""
	
	# Get the systematic error on boost-1. 
	boosterr_sq_assoc = pa.boost_sys**2
	boosterr_sq_b = pa.boost_sys**2
	
	gammaIA_stat_cov_withF = np.zeros((len(rp_bins_c), len(rp_bins_c)))
	gammaIA_sysB_cov_withF = np.zeros((len(rp_bins_c), len(rp_bins_c)))
	
	# Calculate the covariance - we are letting the signal be rp * gamma_IA here
	for i in range(0,len((rp_bins_c))):	 
		for j in range(0, len((rp_bins_c))):
			
			# Statistical
			num_term_stat = SigCavg_asc**2 * gamt_assocBl_import_CV[i,j] + SigCavg_b**2 * gamt_B_import_CV[i,j]
			
			denom_term_stat_withF =( SigCavg_asc * ( Boost_a[i] -1. + F_assoc_fid ) -  SigCavg_b * ( Boost_b[i] -1. + F_b_fid) ) * ( SigCavg_asc * (Boost_a[j] -1. + F_assoc_fid) - SigCavg_b * (Boost_b[j] -1.+ F_b_fid) )
			
			gammaIA_stat_cov_withF[i,j] = num_term_stat / denom_term_stat_withF
			
			if (i==j):
				gammaIA_sysB_cov_withF[i,j] = g_IA_fid[i]**2 * ( boosterr_sq_assoc + boosterr_sq_b ) / ( (Boost_a[i] -1. + F_assoc_fid) -  ( Boost_b[i] -1. + F_b_fid) )**2

				
	# For the systematic cases, we need to add off-diagonal elements - we assume fully correlated
	for i in range(0,len((rp_bins_c))):	
		for j in range(0,len((rp_bins_c))):
			if (i != j):
				gammaIA_sysB_cov_withF[i,j] = np.sqrt(gammaIA_sysB_cov_withF[i,i]) * np.sqrt(gammaIA_sysB_cov_withF[j,j])
		
	# Get the stat + sysB covariance matrix for showing the difference between using excess and using all physically associated galaxies:
	gammaIA_cov_stat_sysB_withF = gammaIA_sysB_cov_withF + gammaIA_stat_cov_withF
	
	#save_gIA = np.column_stack((rp_bins_c, g_IA_fid / np.sqrt(np.diag(gammaIA_cov_stat_sysB_withF))))
	#np.savetxt('./txtfiles/StoN_stat_sysB_Blazek_gamt_'+SURVEY+'_deltaz='+str(pa.delta_z)+'_rlim='+str(pa.mlim)+'.txt', save_gIA)
	
	Cov_inv_stat = np.linalg.inv(gammaIA_stat_cov_withF)
	StoNsq_stat = np.dot( g_IA_fid , np.dot(Cov_inv_stat, g_IA_fid))
	
	Cov_inv_stat_sysB = np.linalg.inv(gammaIA_cov_stat_sysB_withF)
	StoNsq_stat_sysB = np.dot(g_IA_fid, np.dot(Cov_inv_stat_sysB, g_IA_fid))

	return (StoNsq_stat, StoNsq_stat_sysB)
	
def gamma_fid(rp):
	""" Returns the fiducial gamma_IA from a combination of terms from different models which are valid at different scales """
	
	if (SURVEY =='SDSS'):
		
		wgg1hfile = './txtfiles/wgg_wgp_terms/wgg_1h_survey='+pa.survey+'_'+endfile+'.txt'
		wgg2hfile = './txtfiles/wgg_wgp_terms/wgg_2h_survey='+pa.survey+'_kpts='+str(pa.kpts_wgg)+'_'+endfile+'.txt'
		wgg_rp = ws.wgg_full(rp, pa.fsky, pa.bd, pa.bs, wgg1hfile, wgg2hfile, endfile, SURVEY)
		
		wgp1hfile = './txtfiles/wgg_wgp_terms/wgp_1h_survey='+pa.survey+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt'
		wgp2hfile = './txtfiles/wgg_wgp_terms/wgp_2h_survey='+pa.survey+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt'
		wgp_rp = ws.wgp_full(rp, pa.bd, pa.q11, pa.q12, pa.q13, pa.q21, pa.q22, pa.q23, pa.q31, pa.q32, pa.q33, wgp1hfile, wgp2hfile, SURVEY)
		
	elif (SURVEY=='LSST_DESI'):
		
		wgg1hfile = './txtfiles/wgg_wgp_terms/wgg_1h_survey='+pa.survey+'_'+endfile+'.txt'
		wgg2hfile = './txtfiles/wgg_wgp_terms/wgg_2h_survey='+pa.survey+'_kpts='+str(pa.kpts_wgg)+'_'+endfile+'.txt'
		wgg_rp = ws.wgg_full(rp, pa.fsky, pa.bd, pa.bs, wgg1hfile, wgg2hfile, endfile, SURVEY)
		
		wgp1hfile = './txtfiles/wgg_wgp_terms/wgp_1h_survey='+pa.survey+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt'
		wgp2hfile = './txtfiles/wgg_wgp_terms/wgp_2h_survey='+pa.survey+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt'
		wgp_rp = ws.wgp_full(rp, pa.bd, pa.q11, pa.q12, pa.q13, pa.q21, pa.q22, pa.q23, pa.q31, pa.q32, pa.q33, wgp1hfile, wgp2hfile, SURVEY)
	else:
		print "We don't have support for that survey yet. Exiting."
		exit()
	
	# Get the red fraction for the full source sample (A+B)
	fred_file = './txtfiles/f_red/f_red_'+SURVEY+'_deltaz='+str(pa.delta_z)+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt'
	if (os.path.isfile(fred_file)):
		f_red = np.loadtxt(fred_file)
	else:
		f_red = get_fred()
		fred_save = [0]
		fred_save[0] = f_red
		np.savetxt('./txtfiles/f_red/f_red_'+SURVEY+'_deltaz='+str(pa.delta_z)+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt', fred_save)
	
	gammaIA = (f_red * wgp_rp) / (wgg_rp + 2. * pa.close_cut)  # We assume wgg is the same for red and blue galaxies.

	gammaIA = (f_red * wgp_rp) / (wgg_rp + 2. * pa.close_cut)  # We assume wgg is the same for red and blue galaxies.

	return gammaIA


def check_covergence():
	""" Check how the covariance matrices calculated in Fourier space (in another file) have converged with rpts """
	
	#rpts_1 = '2000'; rpts_2 = '2500'; 
	#DeltaCov_a_import_CV_rpts1 = np.loadtxt('./txtfiles/cov_DelSig_1h2h_'+SURVEY+'_sample=A_rpts'+rpts_1+'_lpts100000_SNanalytic_deltaz=0.17.txt')
	#DeltaCov_b_import_CV_rpts1 = np.loadtxt('./txtfiles/cov_DelSig_1h2h_'+SURVEY+'_sample=B_rpts'+rpts_1+'_lpts100000_SNanalytic_deltaz=0.17.txt')
	
	#DeltaCov_a_import_CV_rpts2 = np.loadtxt('./txtfiles/cov_DelSig_1h2h_'+SURVEY+'_sample=A_rpts'+rpts_2+'_lpts100000_SNanalytic_deltaz=0.17.txt')
	#DeltaCov_b_import_CV_rpts2 = np.loadtxt('./txtfiles/cov_DelSig_1h2h_'+SURVEY+'_sample=B_rpts'+rpts_2+'_lpts100000_SNanalytic_deltaz=0.17.txt')
	
	#fracdiff_a = np.abs(DeltaCov_a_import_CV_rpts2 - DeltaCov_a_import_CV_rpts1) / np.abs(DeltaCov_a_import_CV_rpts1)*100
	#print "max percentage difference, sample a=", np.amax(fracdiff_a), "%"
	
	#fracdiff_b = np.abs(DeltaCov_b_import_CV_rpts2 - DeltaCov_b_import_CV_rpts1) / np.abs(DeltaCov_b_import_CV_rpts1)*100
	#print "max percentage difference, sample b=", np.amax(fracdiff_b), "%"
	
	lpts_1 = '90000.0'; lpts_2 = '100000'; 
	DeltaCov_a_import_CV_lpts1 = np.loadtxt('./txtfiles/cov_DelSig_1h2h_'+SURVEY+'_sample=A_rpts2500_lpts'+lpts_1+'_SNanalytic_deltaz=0.17.txt')
	DeltaCov_b_import_CV_lpts1 = np.loadtxt('./txtfiles/cov_DelSig_1h2h_'+SURVEY+'_sample=B_rpts2500_lpts'+lpts_1+'_SNanalytic_deltaz=0.17.txt')
	
	DeltaCov_a_import_CV_lpts2 = np.loadtxt('./txtfiles/cov_DelSig_1h2h_'+SURVEY+'_sample=A_rpts2500_lpts'+lpts_2+'_SNanalytic_deltaz=0.17.txt')
	DeltaCov_b_import_CV_lpts2 = np.loadtxt('./txtfiles/cov_DelSig_1h2h_'+SURVEY+'_sample=B_rpts2500_lpts'+lpts_2+'_SNanalytic_deltaz=0.17.txt')
	
	fracdiff_a = np.abs(DeltaCov_a_import_CV_lpts2 - DeltaCov_a_import_CV_lpts1) / np.abs(DeltaCov_a_import_CV_lpts1)*100
	print "max percentage difference, sample a=", np.amax(fracdiff_a), "%"
	
	fracdiff_b = np.abs(DeltaCov_b_import_CV_lpts2 - DeltaCov_b_import_CV_lpts1) / np.abs(DeltaCov_b_import_CV_lpts1)*100
	print "max percentage difference, sample b=", np.amax(fracdiff_b), "%"
	
	return

######## MAIN CALLS ##########

# Import the parameter file:
if (SURVEY=='SDSS'):
	import params as pa
elif (SURVEY=='LSST_DESI'):
	import params_LSST_DESI as pa
else:
	print "We don't have support for that survey yet; exiting."
	exit()

# Set up projected bins
rp_bins 	= 	setup.setup_rp_bins(pa.rp_min, pa.rp_max, pa.N_bins)
rp_cent		=	setup.rp_bins_mid(rp_bins)

# Set up a function to get z as a function of comoving distance
(z_of_com, com_of_z) = setup.z_interpof_com(SURVEY) 

# Get fiducial gamma_IA
g_IA_fid = gamma_fid(rp_cent)

# Get the theoretical quantities needed 
DeltaSig_the = get_DeltaSig_theory(rp_bins, rp_cent)
gamt_assoc = get_gammat_theory('assocBl')
gamt_B = get_gammat_theory('B')

# Get the real-space shape-noise-only covariance matrices for Delta Sigma for each sample if we want to compare against them (pass this to get_IA_cov)
#gamtCov_assoc = shapenoise_cov('assocBl', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
#gamtCov_b = shapenoise_cov('B', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)

"""# Write to file sigma (gammat) / gammat for assoc and B to compare.
save_gt_fracassoc = np.column_stack((rp_cent, np.sqrt(gamtCov_assoc) / gamt_assoc))
np.savetxt('./txtfiles/gt_fracerr_survey='+SURVEY+'_sample=assoc.txt', save_gt_fracassoc)

save_gt_fracB = np.column_stack((rp_cent, np.sqrt( gamtCov_b) / gamt_B))
np.savetxt('./txtfiles/gt_fracerr_survey='+SURVEY+'_sample=B.txt', save_gt_fracB)"""

(StoNstat, StoN_stat_sysB) = get_gammaIA_cov(rp_bins, rp_cent)

# Save the statistical-only S-to-N and from stat + sysB  
StoNstat_save = [StoNstat]; np.savetxt('./txtfiles/StoN/StoNstat_gamt_Blazek_'+SURVEY+'_deltaz='+str(pa.delta_z)+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt', StoNstat_save)  
StoNstat_sysB_save = [StoN_stat_sysB]; np.savetxt('./txtfiles/StoN/StoN_stat_sysB_gamt_Blazek_'+SURVEY+'_deltaz='+str(pa.delta_z)+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt', StoNstat_sysB_save)

exit()

print "StoNstat=", StoNstat

# Save the statistical-only S-to-N   
StoNstat_save = [StoNstat]
np.savetxt('./txtfiles/StoN/StoNstat_Blazek_'+SURVEY+'_deltaz='+str(pa.delta_z)+'_gamtBl_rlim='+str(pa.mlim)+'_'+endfile+'.txt', StoNstat_save)

# Save the statistical + sysB S-to-N   
StoNstat_sysB_save = [StoN_stat_sysB]
np.savetxt('./txtfiles/StoN/StoN_stat_sysB_Blazek_'+SURVEY+'_deltaz='+str(pa.delta_z)+'_gamtBl_rlim='+str(pa.mlim)+'_'+endfile+'.txt', StoNstat_sysB_save)

exit()

# Save the sysZ stuff	
saveSN_sysZ = np.column_stack(( pa.fudge_frac_level, StoN_cza, StoN_czb, StoN_Fa, StoN_Fb, StoN_Siga, StoN_Sigb))
np.savetxt('./txtfiles/StoN/StoN_SysToStat_Blazek_'+SURVEY+'_deltaz='+str(pa.delta_z)+'_SNonly.txt', saveSN_sysZ)


# Load information for Ncorr from other method to use in this plot. StoN(stat) and StoN(sysz).
#Both are 2D matrices. The sysz one is a function of a and fraclevel - pick the a value we want to use. The stat one is a function of a and rho - pick the value of a and rho we want to use.
StoNsq_sysz_Ncorr_mat = np.loadtxt('./txtfiles/StoN/StoNsq_sysz_shapes_survey='+SURVEY+'.txt')
StoNsq_stat_Ncorr_mat = np.loadtxt('./txtfiles/StoN/StoNsq_stat_shapes_survey='+SURVEY+'.txt')
avec = np.loadtxt('./txtfiles/a_survey='+SURVEY+'.txt')
rhovec = np.loadtxt('./txtfiles/rho_survey='+SURVEY+'.txt')

# here we pick our a and rho values and find their indices:
a = 0.3; rho = 0.7
ind_a = next(j[0] for j in enumerate(avec) if j[1]>=a)
ind_rho = next(j[0] for j in enumerate(rhovec) if j[1]>=rho)
#print "For Ncorr sysz: a=", avec[ind_a], "rho=", rhovec[ind_rho]

StoN_sq_sysz_Ncorr = StoNsq_sysz_Ncorr_mat[ind_a, :]
StoN_sq_stat_Ncorr = StoNsq_stat_Ncorr_mat[ind_a, ind_rho]

# If necessary, load the stuff from this method:
#levels, StoN_cza, StoN_czb, StoN_Fa, StoN_Fb, StoN_Siga, StoN_Sigb = np.loadtxt('./txtfiles/StoN_SysToStat_Blazek_'+SURVEY+'_deltaz='+str(pa.delta_z)+'.txt', unpack=True)
#StoNstat = np.loadtxt('./txtfiles/StoNstat_Blazek_'+SURVEY+'_deltaz='+str(pa.delta_z)+'.txt')
print "fudge frac level="
print pa.fudge_frac_level
print "slope, cza="
print (np.sqrt(StoNstat) / np.sqrt(StoN_cza)) / pa.fudge_frac_level
print "slope, czb="
print (np.sqrt(StoNstat) / np.sqrt(StoN_czb)) / pa.fudge_frac_level
print "slope, Fa="
print (np.sqrt(StoNstat) / np.sqrt(StoN_Fa)) / pa.fudge_frac_level
print "slope, Fb="
print (np.sqrt(StoNstat) / np.sqrt(StoN_Fb)) / pa.fudge_frac_level
print "slope, Siga="
print (np.sqrt(StoNstat) / np.sqrt(StoN_Siga)) / pa.fudge_frac_level
print "slope, Sigb="
print (np.sqrt(StoNstat) / np.sqrt(StoN_Sigb)) / pa.fudge_frac_level
print "slope, A_ph=", (np.sqrt(StoN_sq_stat_Ncorr) / np.sqrt(StoN_sq_sysz_Ncorr))/ pa.fudge_frac_level

"""# Make plot of (S/N)_sys / (S/N)_stat as a function of fractional z-related systematic error on each relevant parameter.
if (SURVEY=='SDSS'):
	plt.figure()
	plt.rc('font', family='serif', size=16)
	plt.loglog(pa.fudge_frac_level,  (np.sqrt(StoNstat) / np.sqrt(StoN_cza)) / pa.fudge_frac_level, linewidth='3', color='#006cc0', label='$c_z^a$')
	plt.hold(True)
	#plt.axvline(x=0.19, linewidth='3', color='#006cc0')
	plt.loglog(pa.fudge_frac_level,  (np.sqrt(StoNstat) / np.sqrt(StoN_czb)) / pa.fudge_frac_level, linewidth='3', linestyle='--',color='#006cc0', label='$c_z^b$')
	plt.hold(True)
	#plt.axvline(x=0.19, linewidth='3', color='#006cc0', linestyle='--')
	#plt.hold(True)
	plt.loglog(pa.fudge_frac_level,  (np.sqrt(StoNstat) / np.sqrt(StoN_Fa)) / pa.fudge_frac_level, 'g', linewidth='3', label='$F_a$')
	plt.hold(True)
	#plt.axvline(x=353., linewidth='3', color='g')
	#plt.hold(True)
	plt.loglog(pa.fudge_frac_level,(np.sqrt(StoNstat) /  np.sqrt(StoN_Fb)) / pa.fudge_frac_level, 'g',linestyle='--',linewidth='3', label='$F_b$')
	plt.hold(True)
	#plt.axvline(x=4009., linewidth='3', color='g', linestyle='--')
	#plt.hold(True)
	plt.loglog(pa.fudge_frac_level, (np.sqrt(StoNstat) / np.sqrt(StoN_Siga))/ pa.fudge_frac_level, linewidth='3', color='#FFA500', label='$<\\Sigma_{IA}^a>$')
	plt.hold(True)
	#plt.axvline(x=25., linewidth='3', color='#FFA500')
	#plt.hold(True)
	plt.loglog(pa.fudge_frac_level,  (np.sqrt(StoNstat) / np.sqrt(StoN_Sigb))/ pa.fudge_frac_level, linestyle='--', linewidth='3',color='#FFA500', label='$<\\Sigma_{IA}^b>$')
	plt.hold(True)
	#plt.axvline(x=283., linewidth='3', color='#FFA500', linestyle='--')
	#plt.hold(True)
	#plt.loglog(pa.fudge_frac_level, (np.sqrt(StoN_sq_stat_Ncorr) / np.sqrt(StoN_sq_sysz_Ncorr))/ pa.fudge_frac_level , 'm', linewidth='3',label='$A_{ph}$')
	plt.hold(True)
	#plt.axhline(y=1, color='k', linewidth=2, linestyle='--')
	plt.xlabel('Fractional error', fontsize=20)
	plt.ylabel('$\left(\\frac{S/N_{\\rm stat}}{S/N_{\\rm sys}}\\right) \div {\\rm Fractional \, error}$', fontsize=20)
	plt.tick_params(axis='both', which='major', labelsize=18)
	plt.tick_params(axis='both', which='minor', labelsize=18)
	#plt.xlim(0.008, 0.35)
	plt.ylim(0.001, 300)
	plt.legend(ncol=4, numpoints=1, fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.05))
	plt.tight_layout()
	plt.savefig('./plots/SysNoiseToStatNoise_'+SURVEY+'.png')
	plt.close()
elif(SURVEY=='LSST_DESI'):
	plt.figure()
	plt.rc('font', family='serif', size=16)
	plt.loglog(pa.fudge_frac_level, np.sqrt(StoNstat) / np.sqrt(StoN_cza)/ pa.fudge_frac_level,linewidth='3',  color='#006cc0', label='$c_z^a$')
	plt.hold(True)
	plt.loglog(pa.fudge_frac_level, np.sqrt(StoNstat) / np.sqrt(StoN_czb)/ pa.fudge_frac_level , linewidth='3', linestyle='--', color='#006cc0', label='$c_z^b$')
	plt.hold(True)
	plt.loglog(pa.fudge_frac_level, np.sqrt(StoNstat) /np.sqrt(StoN_Fa)/ pa.fudge_frac_level, 'g', linewidth='3',  label='$F_a$')
	plt.hold(True)
	plt.loglog(pa.fudge_frac_level, np.sqrt(StoNstat) / np.sqrt(StoN_Fb)/ pa.fudge_frac_level , 'g',linewidth='3', linestyle='--', label='$F_b$')
	plt.hold(True)
	plt.loglog(pa.fudge_frac_level, np.sqrt(StoNstat) / np.sqrt(StoN_Siga)/ pa.fudge_frac_level, linewidth='3',color='#FFA500', label='$<\\Sigma_{IA}^a>$')
	plt.hold(True)
	plt.loglog(pa.fudge_frac_level, np.sqrt(StoNstat) / np.sqrt(StoN_Sigb)/ pa.fudge_frac_level , linewidth='3',  linestyle='--', color='#FFA500', label='$<\\Sigma_{IA}^b>$')
	plt.hold(True)
	plt.loglog(pa.fudge_frac_level, (np.sqrt(StoN_sq_stat_Ncorr) / np.sqrt(StoN_sq_sysz_Ncorr))/ pa.fudge_frac_level, 'm', linewidth='3', label='$A_{ph}$')
	#plt.axhline(y=1, color='k', linewidth=2, linestyle='--')
	plt.xlabel('Fractional error', fontsize=20)
	plt.ylabel('$\left(\\frac{S/N_{\\rm stat}}{S/N_{\\rm sys}}\\right) \div {\\rm Fractional \, error}$', fontsize=20)
	plt.tick_params(axis='both', which='major', labelsize=18)
	plt.tick_params(axis='both', which='minor', labelsize=18)
	plt.xlim(0.008, 0.35)
	plt.legend(ncol=4, numpoints=1, fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.05))
	plt.ylim(0.001, 300)
	plt.tight_layout()
	plt.savefig('./plots/SysNoiseToStatNoise_'+SURVEY+'.png')
	plt.close()"""

