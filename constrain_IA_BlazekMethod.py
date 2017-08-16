# This is a script which predicts constraints on intrinsic alignments, using an updated version of the method of Blazek et al 2012 in which it is not assumed that only excess galaxies contribute to IA (rather it is assumed that source galaxies which are close to the lens along the line-of-sight can contribute.)

SURVEY = 'SDSS'
print "SURVEY=", SURVEY

import numpy as np
import scipy
import scipy.integrate
import matplotlib.pyplot as plt
import scipy.interpolate
import shared_functions_setup as setup
import shared_functions_wlp_wls as ws
import pyccl as ccl
	
############## GENERIC FUNCTIONS ###############
	
def N_of_zph(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, dNdzpar, pzpar, dNdztype, pztype):
	""" Returns dNdz_ph, the number density in terms of photometric redshift, defined over photo-z range (z_a_def_ph, z_b_def_ph), normalized over the photo-z range (z_a_norm_ph, z_b_norm_ph), normalized over the spec-z range (z_a_norm, z_b_norm), and defined on the spec-z range (z_a_def, z_b_def)"""
	
	(z, dNdZ) = setup.get_NofZ_unnormed(dNdzpar, dNdztype, z_a_def_s, z_b_def_s, 1000)
	
	(z_norm, dNdZ_norm) = setup.get_NofZ_unnormed(dNdzpar, dNdztype, z_a_norm_s, z_b_norm_s, 1000)
	
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
	
	if (pa.survey=='SDSS'):
		
		if hasattr(z_s_, "__len__"):
			sig_e = 2. / pa.S_to_N * np.ones(len(z_s_))
		else:
			sig_e = 2. / pa.S_to_N
			
	elif(pa.survey=='LSST_DESI'):
		if hasattr(z_s_, "__len__"):
			sig_e = pa.a_sm / pa.SN_med * ( 1. + (pa.b_sm / pa.R_med)**pa.c_sm) * np.ones(len(z_s_))
		else:
			sig_e = pa.a_sm / pa.SN_med * ( 1. + (pa.b_sm / pa.R_med)**pa.c_sm) 

	return sig_e
	
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
   
def weights_times_SigC(e_rms, z_, z_l_):

    """ Returns the inverse variance weights as a function of redshift. """
        
    SigC_t_inv = get_SigmaC_inv(z_, z_l_)

    if ((hasattr(z_, "__len__")==True) and (hasattr(z_l_, "__len__")==True)):
        weights = np.zeros((len(z_), len(z_l_)))
        for zsi in range(0,len(z_)):
            for zli in range(0,len(z_l_)):
                weights[zsi, zli] = SigC_t_inv[zsi, zli]/(sigma_e(z_)[zsi]**2 + e_rms**2)
    else:
        if (hasattr(z_, "__len__")):
            weights = SigC_t_inv/(sigma_e(z_)**2 + e_rms**2 * np.ones(len(z_)))
        else:
            weights = SigC_t_inv/(sigma_e(z_)**2 + e_rms**2 )

    return weights

def get_SigmaC_inv(z_s_, z_l_):
    """ Returns the theoretical value of 1/Sigma_c, (Sigma_c = the critcial surface mass density).
    z_s_ and z_l_ can be 1d arrays, so the returned value will in general be a 2d array. """

    com_s = com_of_z(z_s_) 
    com_l = com_of_z(z_l_) 

    # Get scale factors for converting between angular-diameter and comoving distances.
    a_l = 1. / (z_l_ + 1.)
    a_s = 1. / (z_s_ + 1.)
    
    D_s = a_s * com_s # Angular diameter source distance.
    D_l = a_l * com_l # Angular diameter lens distance
    
    # The dimensions of D_ls depend on the dimensions of z_s_ and z_l_
    if ((hasattr(z_s_, "__len__")==True) and (hasattr(z_l_, "__len__")==True)):
        D_ls = np.zeros((len(z_s_), len(z_l_)))
        Sigma_c_inv = np.zeros((len(z_s_), len(z_l_)))
        for zsi in range(0,len(z_s_)):
            for zli in range(0,len(z_l_)):
                D_ls[zsi, zli] = D_s[zsi] - D_l[zli]
                # Units are pc^2 / (h Msun), comoving
                if(D_ls[zsi, zli]<0.):
                    Sigma_c_inv[zsi, zli] = 0.
                else:
                    Sigma_c_inv[zsi, zli] = 4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc *   D_l[zli] * D_ls[zsi, zli] * (1 + z_l_[zli])**2 / D_s[zsi]
    else:
        D_ls = (D_s - D_l) 
        # Units are pc^2 / (h Msun), comoving
        if hasattr(z_s_, "__len__"):
            Sigma_c_inv = np.zeros(len(z_s_))
            for zsi in range(0, len(z_s_)):
                if(D_s[zsi]<=D_l):
                    Sigma_c_inv[zsi] = 0.
                else:
                    Sigma_c_inv[zsi] = 4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc *   D_l * D_ls[zsi]* (1 + z_l_)**2 / D_s[zsi]
        elif hasattr(z_l_, "__len__"): 
            Sigma_c_inv = np.zeros(len(z_l_))
            for zli in range(0,len(z_l_)):
                if(D_s<=D_l[zli]):
                    Sigma_c_inv[zli] = 0.
                else:
                    Sigma_c_inv[zli] = 4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc *   D_l[zli] * D_ls[zli]* (1 + z_l_[zli])**2 / D_s
        else:
            if (D_s<D_l):
                Sigma_c_inv=0.
            else:
                 Sigma_c_inv= 4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc *   D_l * D_ls* (1 + z_l_)**2 / D_s
                    
    return Sigma_c_inv

#### THESE ARE OLD FUNCTIONS WHICH COMPUTE THE SHAPE-NOISE ONLY COVARIANCE IN REAL SPACE #####
# I'm just keeping these to be able to compare with their output

def shapenoise_cov(photoz_samp, rp, dNdzpar, pzpar, dNdztype, pztype):
	""" Returns a diagonal covariance matrix in bins of projected radius for a measurement dominated by shape noise. Elements are 1 / (sum_{ls} w), carefully normalized, in each bin. """
	
	# Get the area of each projected radial bin in square arcminutes
	bin_areas       =       setup.get_areas(rp, pa.zeff, SURVEY) # We're just going to use the mean redshift for this.
	
	weighted_frac_sources = sum_weights(photoz_samp, 'nocut', rp, dNdzpar, pzpar, dNdztype, pztype) / sum_weights('full', 'nocut', rp, dNdzpar, pzpar, dNdztype, pztype)
	
	# Get weighted SigmaC values
	SigC_avg = sum_weights_SigC(photoz_samp, 'nocut', rp, dNdzpar, pzpar, dNdztype, pztype)/ sum_weights(photoz_samp, 'nocut', rp, dNdzpar, pzpar, dNdztype, pztype)
	
	if (photoz_samp=='A'):
		e_rms = pa.e_rms_Bl_a
	elif (photoz_samp =='B'):
		e_rms = pa.e_rms_Bl_b
	elif (phtoz_samp == 'full'):
		e_rms = pa.e_rms_Bl_full
	else:
		print "We do not have support for that photoz sample"
		exit()
	
	cov = e_rms**2 * SigC_avg**2 / ( pa.n_l * pa.Area_l * bin_areas * pa.n_s * weighted_frac_sources)
	
	return cov

################### THEORETICAL VALUES FOR FRACTIONAL ERROR CALCULATION ########################333
	
def sum_weights(photoz_sample, specz_cut, rp_bins, dNdz_par, pz_par, dNdztype, pztype):
	""" Returns the sum over weights for each projected radial bin. 
	photoz_sample = 'A', 'B', or 'full'
	specz_cut = 'close', or 'nocut'
	"""
	
	# Get lens redshift distribution
	zL = np.linspace(pa.zLmin, pa.zLmax, 100)
	dNdzL = setup.get_dNdzL(zL, SURVEY)
	
	
	# Loop over lens redshift values
	sum_ans_zph = np.zeros(len(zL))
	for zi in range(0,len(zL)):
		if (photoz_sample == 'A'):
			
			if (specz_cut == 'nocut'):
				(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi], zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
				weight = weights(pa.e_rms_Bl_a, z_ph, zL[zi])
				sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
			elif (specz_cut == 'close'):
				(z_ph, dNdz_ph) = N_of_zph(zL[zi]-z_of_com(pa.close_cut), zL[zi]+z_of_com(pa.close_cut), pa.zsmin, pa.zsmax, zL[zi], zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
				weight = weights(pa.e_rms_Bl_a, z_ph, zL[zi])
				sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
			else:
				print "We do not have support for that spec-z cut. Exiting."
				exit()
			
		elif(photoz_sample == 'B'):
			
			if (specz_cut == 'nocut'):
				(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi] + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
				weight = weights(pa.e_rms_Bl_b, z_ph, zL[zi])
				sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
			elif (specz_cut == 'close'):
				(z_ph, dNdz_ph) = N_of_zph(zL[zi]-z_of_com(pa.close_cut), zL[zi]+z_of_com(pa.close_cut), pa.zsmin, pa.zsmax, zL[zi] + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
				weight = weights(pa.e_rms_Bl_b, z_ph, zL[zi])
				
				sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
			else:
				print "We do not have support for that spec-z cut. Exiting."
				exit()
		
		elif(photoz_sample == 'full'):
			
			if (specz_cut == 'nocut'):
				(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
				weight = weights(pa.e_rms_Bl_full, z_ph, zL[zi])
				
				sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
			elif (specz_cut == 'close'):
				(z_ph, dNdz_ph) = N_of_zph(zL[zi]-z_of_com(pa.close_cut), zL[zi]+z_of_com(pa.close_cut), pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
				weight = weights(pa.e_rms_Bl_full, z_ph, zL[zi])
				
				sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
			else:
				print "We do not have support for that spec-z cut. Exiting."
				exit()
		
		else:
			print "We do not have support for that photo-z sample cut. Exiting."
			print photoz_sample
			exit()
	
	# Now sum over lens redshift:
	sum_ans = scipy.integrate.simps(sum_ans_zph * dNdzL, zL)
	
	return sum_ans

def sum_weights_SigC(photoz_sample, specz_cut, rp_bins, dNdz_par, pz_par, dNdztype, pztype):
	""" Returns the sum over weights * SigmaC for each projected radial bin. 
	photoz_sample = 'A', 'B', or 'full'
	specz_cut = 'close', or 'nocut'
	"""
	
	# Get lens redshift distribution
	zL = np.linspace(pa.zLmin, pa.zLmax, 100)
	dNdzL = setup.get_dNdzL(zL, SURVEY)
	
	# Loop over lens redshift values
	sum_ans_zph = np.zeros(len(zL))
	for zi in range(0,len(zL)):
		if (photoz_sample == 'A'):
			
			if (specz_cut == 'nocut'):
				(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi], zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
				weightSigC = weights_times_SigC(pa.e_rms_Bl_a, z_ph, zL[zi])
				
				sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weightSigC, z_ph)
				
			elif (specz_cut == 'close'):
				(z_ph, dNdz_ph) = N_of_zph(zL[zi]-z_of_com(pa.close_cut), zL[zi]+z_of_com(pa.close_cut), pa.zsmin, pa.zsmax, zL[zi], zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
				weightSigC = weights_times_SigC(pa.e_rms_Bl_a, z_ph, zL[zi])
				
				sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weightSigC, z_ph)
				
			else:
				print "We do not have support for that spec-z cut. Exiting."
				exit()
			
		elif(photoz_sample == 'B'):
			
			if (specz_cut == 'nocut'):
				(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi] + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
				weightSigC = weights_times_SigC(pa.e_rms_Bl_b, z_ph, zL[zi])
				
				sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weightSigC, z_ph)
				
			elif (specz_cut == 'close'):
				(z_ph, dNdz_ph) = N_of_zph(zL[zi]-z_of_com(pa.close_cut), zL[zi]+z_of_com(pa.close_cut), pa.zsmin, pa.zsmax, zL[zi] + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
				weightSigC = weights_times_SigC(pa.e_rms_Bl_b, z_ph, zL[zi])
				
				sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weightSigC, z_ph)
				
			else:
				print "We do not have support for that spec-z cut. Exiting."
				exit()
		
		elif(photoz_sample == 'full'):
			
			if (specz_cut == 'nocut'):
				(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
				weightSigC = weights_times_SigC(pa.e_rms_Bl_full, z_ph, zL[zi])
				
				sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weightSigC, z_ph)
				
			elif (specz_cut == 'close'):
				(z_ph, dNdz_ph) = N_of_zph(zL[zi]-z_of_com(pa.close_cut), zL[zi]+z_of_com(pa.close_cut), pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
				weightSigC = weights_times_SigC(pa.e_rms_Bl_full, z_ph, zL[zi])
				
				sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weightSigC, z_ph)
				
			else:
				print "We do not have support for that spec-z cut. Exiting."
				exit()
		
		else:
			print "We do not have support for that photo-z sample cut. Exiting."
			print photoz_sample
			exit()
	
	# Now sum over lens redshift:
	sum_ans = scipy.integrate.simps(sum_ans_zph * dNdzL, zL)
	
	return sum_ans
	
def get_boost(rp_cents_, propfact):
	"""Returns the boost factor in radial bins. propfact is a tunable parameter giving the proportionality constant by which boost goes like projected correlation function (= value at 1 Mpc/h). """

	Boost = propfact *(rp_cents_)**(-0.8) + np.ones((len(rp_cents_))) # Empirical power law fit to the boost, derived from the fact that the boost goes like projected correlation function.

	return Boost
	
def get_F(photoz_sample, rp_bins_, dNdz_par, pz_par, dNdztype, pztype):
	""" Returns F (the weighted fraction of lens-source pairs from the smooth dNdz which are contributing to IA) """

	# Sum over `rand-close'
	numerator = sum_weights(photoz_sample, 'close', rp_bins_, dNdz_par, pz_par, dNdztype, pztype)

	#Sum over all `rand'
	denominator = sum_weights(photoz_sample, 'nocut', rp_bins_, dNdz_par, pz_par, dNdztype, pztype)

	F = np.asarray(numerator) / np.asarray(denominator)

	return F

def get_cz(photoz_sample, rp_bins_, dNdzpar, pzpar, dNdztype, pztype):
	""" Returns the value of the photo-z bias parameter c_z"""

	# The denominator (of 1+bz) is just a sum over tilde(weights) of all random-source pairs
	denominator = sum_weights(photoz_sample, 'nocut', rp_bins_, dNdzpar, pzpar, dNdztype, pztype)
	
	# Now get the mean of tilde(weight) tilde(Sigma_c) Sigma_c^{-1} 
	bSigw = get_bSigW(photoz_sample, dNdzpar, dNdztype, pzpar, pztype)

	cz = denominator / bSigw

	return cz

def get_bSigW(photoz_samp, dNdzpar, dNdztype, pzpar, pztype):
	""" Returns an interpolating function for tilde(w)tilde(SigC)SigC^{-1} as a function of source photo-z. Used in computing photo-z bias c_z / 1+b_z."""

	# Get lens redshift distribution
	zLvec = scipy.linspace(pa.zLmin, pa.zLmax, 100) # Using 100 points here (and in sum-weights) is sufficient for convergence to well below a 10^{-3} level. August 15, 2017.
	dNdzl = setup.get_dNdzL(zLvec, SURVEY)

	Dlvec = com_of_z(zLvec) / (1. + zLvec)

	# Loop over each zl.
	bsw = [0] * len(zLvec)
	bsw_ofzL = np.zeros(len(zLvec))
	for zi in range(0,len(zLvec)):
		
		# Get dNdzph
		if (photoz_samp == 'A'):
			(zph, dNdzph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zLvec[zi], zLvec[zi] + pa.delta_z, pa.zphmin, pa.zphmax, dNdzpar, pzpar, dNdztype, pztype)
			e_rms = pa.e_rms_Bl_a
		elif (photoz_samp == 'B'):
			(zph, dNdzph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zLvec[zi] + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, dNdzpar, pzpar, dNdztype, pztype)
			e_rms = pa.e_rms_Bl_b
		elif (photoz_samp == 'full'):
			(zph, dNdzph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, dNdzpar, pzpar, dNdztype, pztype)
			e_rms = pa.e_rms_Bl_full
		
		# Get angular diamter distance to sources using photometric redshift ditributions
		Ds_photo = com_of_z(zph) / (1. + zph)
		#print "Ds_photo=", Ds_photo
		
		bsw[zi] = np.zeros(len(zph))
			
		# Loop over photometric redshift distances.
		for zpi in range(0,len(zph)):
			#print "zpi=", zpi
			
			if (Ds_photo[zpi]>Dlvec[zi]):
				Dls_photo = Ds_photo[zpi] - Dlvec[zi]
			else:
				Dls_photo = 0.
				
			#print "Dls_photo=", Dls_photo
		
			if (pztype == 'Gaussian'):
				# Draw a set of points from a normal dist with mean zspec. These are the spec-z's for this photo-z.
				zsvec = np.random.normal(zph[zpi], pzpar[0]*(1. + zph[zpi]), 10000)
				# Using 10000 sample points in the above line is enough for better than 10^{-3} convergence of cz. August 15,2017
				# Set points below 0 to zero, it doesn't matter because they will always be lower redshift than the lens, just prevents errors in getting Ds:
				for i in range(0, len(zsvec)):
					if (zsvec[i]<0.):
						zsvec[i] = 0.0001
			else:
				print "Photo-z type "+str(pztype)+" is not supported. Exiting."
				exit()
			# Get angular diameter disances to sources using spec-z's. 
			Ds_spec = com_of_z(zsvec) / (1. + zsvec)
			
			Dls_spec = np.zeros(len(zsvec))
			for zsi in range(0,len(zsvec)):
				if (Ds_spec[zsi]> Dlvec[zi]):
					Dls_spec[zsi] = Ds_spec[zsi] - Dlvec[zi]
				else:
					Dls_spec[zsi] = 0.
				
			# Find the mean bsigma at this zphoto
			bsw[zi][zpi] = (4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc)**2   / (e_rms**2 + sigma_e(zph[zpi])**2) * (1. +zLvec[zi])**4 * Dlvec[zi]**2 * (Dls_photo) / Ds_photo[zpi] * np.mean(Dls_spec / Ds_spec)
		
		# Integrate over zph
		bsw_ofzL[zi] = scipy.integrate.simps(bsw[zi] * dNdzph, zph)
			
		
	# Integrate over zL
	bsw_return = scipy.integrate.simps(bsw_ofzL * dNdzl, zLvec)

	return bsw_return
	
def get_Sig_IA(photoz_sample, rp_bins_, boost, dNdz_par, pz_par, dNdztype, pztype):
	""" Returns the value of <\Sigma_c>_{IA} in radial bins. Parameters labeled '2' are for the `rand-close' sums and '1' are for the `excess' sums. """
	
	# There are four terms here. The two in the denominators are sums over randoms (or sums over lenses that can be written as randoms * boost), and these are already set up to calculate.
	denom_rand_close =  sum_weights(photoz_sample, 'close', rp_bins_, dNdz_par, pz_par, dNdztype, pztype)
	denom_rand =  sum_weights(photoz_sample, 'nocut', rp_bins_, dNdz_par, pz_par, dNdztype, pztype)
	denom_excess = (boost - 1.) * denom_rand
	
	# The two in the numerator require summing over weights and Sigma_C. 
	
	#For the sum over rand-close in the numerator, this follows directly from the same type of expression as when summing weights:
	num_rand_close = sum_weights_SigC(photoz_sample, 'close', rp_bins, dNdz_par, pz_par, dNdztype, pztype)

	# The other numerator sum is a term which represents the a sum over excess. We have to get the normalization indirectly so there are a bunch of terms here. See notes.
	# We assume all excess galaxies are at the lens redshift.
	
	# Get the redshift distribution for the lenses
	zLvec = np.linspace(pa.zLmin, pa.zLmax, 500)
	dndzl = setup.get_dNdzL(zLvec, SURVEY)
	
	exc_SigC_arbnorm_ofzL = np.zeros(len(zLvec))
	exc_arbnorm_ofzL = np.zeros(len(zLvec))
	for zi in range(0, len(zLvec)):
		if (photoz_sample=='A'):
			z_ph = scipy.linspace(zLvec[zi], zLvec[zi] + pa.delta_z, 1000)
			
			weight = weights(pa.e_rms_Bl_a, z_ph, zLvec[zi])
			weightSigC = weights_times_SigC(pa.e_rms_Bl_a, z_ph, zLvec[zi])
			
			# We first compute a sum over excess of weights and Sigma_C with arbitrary normalization:
			exc_SigC_arbnorm_ofzL[zi] = scipy.integrate.simps(weightSigC * setup.p_z(z_ph, zLvec[zi], pz_par, pztype), z_ph)
			# We do the same for a sum over excess of just weights with the same arbitrary normalization:
			exc_arbnorm_ofzL[zi] = scipy.integrate.simps(weight * setup.p_z(z_ph, zLvec[zi], pz_par, pztype), z_ph)
			
		elif (photoz_sample=='B'):
			z_ph = scipy.linspace(zLvec[zi] + pa.delta_z, pa.zphmax, 1000)
			
			weight = weights(pa.e_rms_Bl_b, z_ph, zLvec[zi])
			weightSigC = weights_times_SigC(pa.e_rms_Bl_b, z_ph, zLvec[zi])
			
			# We first compute a sum over excess of weights and Sigma_C with arbitrary normalization:
			exc_SigC_arbnorm_ofzL[zi] = scipy.integrate.simps(weightSigC * setup.p_z(z_ph, zLvec[zi], pz_par, pztype), z_ph)
			# We do the same for a sum over excess of just weights with the same arbitrary normalization:
			exc_arbnorm_ofzL[zi] = scipy.integrate.simps(weight * setup.p_z(z_ph, zLvec[zi], pz_par, pztype), z_ph)
		
		elif (photoz-samlple == 'full'):
			z_ph = scipy.linspace(pa.zphmin, pa.zphmax, 1000)
			
			weight = weights(pa.e_rms_Bl_full, z_ph, zLvec[zi])
			weightSigC = weights_times_SigC(pa.e_rms_Bl_full, z_ph, zLvec[zi])
			
			# We first compute a sum over excess of weights and Sigma_C with arbitrary normalization:
			exc_SigC_arbnorm_ofzL[zi] = scipy.integrate.simps(weightSigC * setup.p_z(z_ph, zLvec[zi], pz_par, pztype), z_ph)
			# We do the same for a sum over excess of just weights with the same arbitrary normalization:
			exc_arbnorm_ofzL[zi] = scipy.integrate.simps(weight * setup.p_z(z_ph, zLvec[zi], pz_par, pztype), z_ph)
	
		else:
			print "We don't have support for that photo-z sample. Exiting."
			exit()
			
	exc_SigC_arbnorm = scipy.integrate.simps(exc_SigC_arbnorm_ofzL * dndzl, zLvec)
	exc_arbnorm = scipy.integrate.simps(exc_arbnorm_ofzL * dndzl, zLvec)
	
	# We already have an appropriately normalized sum over excess weights, from above (denom_excess), via the relationship with the boost.
	# Put these components together to get the appropriately normalized sum over excess of weights and SigmaC:
	num_excess = exc_SigC_arbnorm / exc_arbnorm * denom_excess
	
	# Sigma_C_inv is in units of pc^2 / (h Msol) (comoving), so Sig_IA is in units of h Msol / pc^2 (comoving).
	Sig_IA = (np.asarray(num_excess + num_rand_close)) / (np.asarray(denom_excess + denom_rand_close)) 
	#print "num=", (np.asarray(num_excess + num_rand_close))
	#print "denom=", (np.asarray(denom_excess + denom_rand_close)) 

	return Sig_IA 

def get_est_DeltaSig(boost, F, cz, SigIA, g_IA_fid):
	""" Returns the value of tilde Delta Sigma in bins"""
		
	EstDeltaSig = np.asarray(DeltaSigma_theoretical) / cz + (boost-1.+ F) * SigIA * g_IA_fid

	return EstDeltaSig

def get_Pkgm_1halo():
	""" Returns (and more usefully saves) the 1halo lens galaxies x dark matter power spectrum, for the calculation of Delta Sigma (theoretical) """
	
	#Define the full k vector over which we will do the Fourier transform to get the correlation function
	logkmin = -6; kpts =40000; logkmax = 5
	kvec_FT = np.logspace(logkmin, logkmax, kpts)
	# Define the downsampled k vector over which we will compute Pk_{gm}^{1h}
	kvec_short = np.logspace(np.log10(kvec_FT[0]), np.log10(kvec_FT[-1]), 40)
	
	# Define the mass vector.
	Mhalo = np.logspace(7., 16., 30)
	# Define the vector of lens redshifts over which we will average.
	zLvec = np.linspace(pa.zLmin, pa.zLmax, 500)
	
	# Get the halo mass function
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = 2.1*10**(-9), n_s=0.96)
	cosmo = ccl.Cosmology(p)
	HMF = np.zeros((len(Mhalo), len(zLvec)))
	for zi in range(0, len(zLvec)):
		HMF[:, zi]= ccl.massfunction.massfunc( cosmo, Mhalo / (pa.HH0/100.), 1./ (1. + zLvec[zi]), odelta=200. )
	
	# Get HOD quantities we need
	if (SURVEY=='SDSS'):
		Ncen_lens = ws.get_Ncen_Reid(Mhalo, SURVEY) # We use the LRG model for the lenses from Reid & Spergel 2008
		Nsat_lens = ws.get_Nsat_Reid(Mhalo, SURVEY) 
	elif (SURVEY=='LSST_DESI'):
		Ncen_lens = ws.get_Ncen(Mhalo, 'nonsense', SURVEY)
		Nsat_lens = ws.get_Nsat(Mhalo, 'nonsense', SURVEY)
	else:
		print "We don't have support for that survey yet!"
		exit()
		
	# Check total number of galaxies:
	tot_ng= np.zeros(len(zLvec))
	for zi in range(0,len(zLvec)):
		tot_ng[zi] = scipy.integrate.simps( ( Ncen_lens + Nsat_lens) * HMF[:, zi], np.log10(Mhalo / (pa.HH0/100.) ) ) / (pa.HH0 / 100.)**3
		# Because the number density comes out a little different than the actual case, especially for DESI, we are going to use this number to get the right normalization.

	# Get the fourier space NFW profile equivalent
	y = ws.gety(Mhalo, kvec_short, SURVEY) 
	
	# Get the density of matter in comoving coordinates
	rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h (comoving distances)
	rho_m = pa.OmM * rho_crit # units of Msol h^2 / Mpc^3 (comoving distances)
	
	# Get Pk
	Pkgm = np.zeros((len(kvec_short), len(zLvec)))
	for ki in range(0,len(kvec_short)):
		for zi in range(0, len(zLvec)):
			Pkgm[ki, zi] = scipy.integrate.simps( HMF[:, zi] * (Mhalo / rho_m) * (Ncen_lens * y[ki, :] + Nsat_lens * y[ki, :]**2), np.log10(Mhalo / (pa.HH0/ 100.))) / (tot_ng[zi]) / (pa.HH0 / 100.)**3
		
	# Now integrate this over the appropriate lens redshift distribution:
	dndzl = setup.get_dNdzL(zLvec, SURVEY)
	Pk_zavg = np.zeros(len(kvec_short))
	for ki in range(0,len(kvec_short)):
		Pk_zavg[ki] = scipy.integrate.simps(dndzl * Pkgm[ki, :], zLvec)
	
	plt.figure()
	plt.loglog(kvec_short, 4* np.pi * kvec_short**3 * Pk_zavg / (2* np.pi)**3, 'mo')
	plt.ylim(0.1, 100000)
	plt.xlim(0.05, 1000)
	plt.ylabel('$4\pi k^3 P_{gg}^{1h}(k)$, $(Mpc/h)^3$, com')
	plt.xlabel('$k$, h/Mpc, com')
	plt.savefig('./plots/Pkgm_1halo_survey='+SURVEY+'.pdf')
	plt.close()
	
	# Get this in terms of the more well-sampled k, for fourier transforming, and save.
	Pkgm_interp = scipy.interpolate.interp1d(np.log(kvec_short), np.log(Pk_zavg))
	logPkgm = Pkgm_interp(np.log(kvec_FT))
	Pkgm = np.exp(logPkgm)
	Pkgm_save = np.column_stack((kvec_FT, Pkgm))
	np.savetxt('./txtfiles/Pkgm_1h_extl_survey='+SURVEY+'.txt', Pkgm_save)
	
	plt.figure()
	plt.loglog(kvec_FT, 4* np.pi * kvec_FT**3 * Pkgm / (2* np.pi)**3, 'mo')
	plt.ylim(0.001, 100000)
	plt.xlim(0.01, 10000)
	plt.ylabel('$4\pi k^3 P_{gg}^{1h}(k)$, $(Mpc/h)^3$, com')
	plt.xlabel('$k$, h/Mpc, com')
	plt.savefig('./plots/Pkgm_1halo_longerkvec_survey='+SURVEY+'.pdf')
	plt.close()
	
	return Pkgm

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
	rho_m = pa.OmM * rho_crit # units of Msol h^2 / Mpc^3 (comoving distances)
		
	# Import the appropriate correlation function (already integrated over lens redshift distribution)
	r_hf, corr_hf = np.loadtxt('./txtfiles/xi_2h_'+SURVEY+'.txt', unpack=True)
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
	DeltaSigma_HF = pa.bd_Bl*(barSigma_HF - Sigma_HF)
			
	####### Now get the 1 halo term #######

	# Get the max R associated to our max M = 10**16
	Rmax = ws.Rhalo(10**16, SURVEY)
	
	# Import the 1halo correlation function from the power spectrum computed in get_Pkgm_1halo and fourier transformed using FFTlog. Already averaged over dndlz.
	r_1h, corr_1h = np.loadtxt('./txtfiles/xigm_1h_'+SURVEY+'.txt', unpack=True)
	
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
	
	plt.figure()
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
	plt.close()
	
	# Interpolate and output at r_bins_c:
	ans_interp = scipy.interpolate.interp1d(rpvec, (DeltaSigma_1h + DeltaSigma_HF) / (10**12))
	ans = ans_interp(rp_bins_c)
	
	return ans # outputting as Msol h / pc^2 

##### ERRORS FOR FRACTIONAL ERROR CALCULATION #####
	
def boost_errors(rp_bins_c, filename):
	""" For the SDSS case, imports a file with 2 columns, [rp (kpc/h), sigma(boost-1)]. Interpolates and returns the value of the error on the boost at the center of each bin. """
	
	if (pa.survey == 'SDSS'):
		(rp_kpc, boost_error_raw) = np.loadtxt(filename, unpack=True)
		# Convert the projected radius to Mpc/h
		rp_Mpc = rp_kpc / 1000.	
		interpolate_boost_error = scipy.interpolate.interp1d(rp_Mpc, boost_error_raw)
		boost_error = interpolate_boost_error(rp_bins_c)
		
	elif (pa.survey == 'LSST_DESI'):
		# At the moment I don't have a good model for the boost errors for LSST x DESI, so I'm assuming it's zero (aka subdominant)
		print "The boost statistical error is currently assumed to be subdominant and set to zero."
		boost_error = np.zeros(len(rp_bins_c))
	else:
		print "That survey doesn't have a boost statistical error model yet."
		exit()
	
	return boost_error

def get_gammaIA_cov(rp_bins, rp_bins_c, fudgeczA, fudgeczB, fudgeFA, fudgeFB, fudgeSigA, fudgeSigB):
	""" Takes information about the uncertainty on constituent elements of gamma_{IA} and combines them to get the covariance matrix of gamma_{IA} in projected radial bins."""
	""" We are only interested right now in the diagonal elements of the covariance matrix, so we assume it is diagonal. """ 

	# Get the fiducial quantities required for the statistical error variance terms. #	
	Boost_a = get_boost(rp_cent, pa.boost_close)
	Boost_b = get_boost(rp_cent, pa.boost_far)
	
	# Run F, cz, and SigIA, this takes a while so we don't always do it.
	if pa.run_quants == True:
	
		################ F's #################
	
		# F factors - first, fiducial
		F_a_fid = get_F('A', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
		F_b_fid = get_F('B', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
		print "F_a_fid=", F_a_fid , "F_b_fid=", F_b_fid
	
		save_F = np.column_stack(([F_a_fid], [F_b_fid]))
		np.savetxt('./txtfiles/F_afid_bfid_extL_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'.txt', save_F)
	
		############# Sig_IA's ##############
	
		# Sig IA - first, fiducial
		# TO INCLUDE AN EXTENDED Z DIST FOR LENSES, THE ARGUMENTS OF THIS FUNCTION WILL CHANGE TO PASS A OR B AS THE SAMPLE LABEL, SEE NOTEBOOK JULY 26.
		Sig_IA_a_fid = get_Sig_IA('A', rp_bins, Boost_a,  pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
		Sig_IA_b_fid = get_Sig_IA('B', rp_bins, Boost_b,  pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
		print "Sig_IA_a_fid=", Sig_IA_a_fid
		print "Sig_IA_b_fid=", Sig_IA_b_fid
	
		save_SigIA = np.column_stack((rp_bins_c, Sig_IA_a_fid, Sig_IA_b_fid))
		np.savetxt('./txtfiles/Sig_IA_afid_bfid_extL_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'.txt', save_SigIA)
	
		############ c_z's ##############
	
		# Photometric biases to estimated Delta Sigmas, fiducial
		cz_a_fid = get_cz('A', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
		cz_b_fid = get_cz('B', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
		print "cz_a_fid =", cz_a_fid, "cz_b_fid=", cz_b_fid
		
		save_cz = np.column_stack(([cz_a_fid], [cz_b_fid]))
		np.savetxt('./txtfiles/cz_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'.txt', save_cz)
	
	############ gamma_IA ###########
	# gamma_IA_fiducial, from model
	g_IA_fid = gamma_fid(rp_bins_c)
	
	if pa.run_quants==False :
		# Load stuff if we haven't computed it this time around:
		(F_a_fid, F_b_fid) = np.loadtxt('./txtfiles/F_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'.txt', unpack=True)
		(rp_bins_c, Sig_IA_a_fid, Sig_IA_b_fid) = np.loadtxt('./txtfiles/Sig_IA_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'.txt', unpack=True)
		(cz_a_fid, cz_b_fid) = np.loadtxt('./txtfiles/cz_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'.txt', unpack=True)
	
	# Estimated Delta Sigmas
	#TO INCLUDE AN EXTENDED REDSHIFT DISTRIBUTION FOR LENSES, JUST REMOVE THE DEPENDENCING ON ZEFF HERE (WE DON'T USE IT ANYMORE).
	DeltaSig_est_a = get_est_DeltaSig(Boost_a, F_a_fid, cz_a_fid, Sig_IA_a_fid, g_IA_fid)
	DeltaSig_est_b = get_est_DeltaSig(Boost_b, F_b_fid, cz_b_fid, Sig_IA_b_fid, g_IA_fid)
	
	############ Get statistical error ############
	
	# Get the real-space shape-noise-only covariance matrices for Delta Sigma for each sample if we want to compare against them.
	DeltaCov_a = shapenoise_cov('A', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
	DeltaCov_b = shapenoise_cov('B', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
	
	# Import the covariance matrix of Delta Sigma for each sample as calculated from Fourier space in a different script, w / cosmic variance terms
	DeltaCov_a_import_CV = np.loadtxt('./txtfiles/cov_DelSig_1h2h_'+SURVEY+'_sample=A_rpts2000_lpts100000_SNanalytic_deltaz=0.17.txt')
	DeltaCov_b_import_CV = np.loadtxt('./txtfiles/cov_DelSig_1h2h_'+SURVEY+'_sample=B_rpts2000_lpts100000_SNanalytic_deltaz=0.17.txt')
	
	plt.figure()
	plt.loglog(rp_bins_c, DeltaCov_a, 'mo', label='shape noise: real')
	plt.hold(True)
	plt.loglog(rp_bins_c, np.diag(DeltaCov_a_import_CV), 'go', label='with CV: Fourier')
	plt.xlabel('r_p')
	plt.ylabel('Variance')
	plt.legend()
	plt.savefig('./plots/check_DeltaSigma_var_SNanalytic_'+SURVEY+'_A_1h2h_extl.pdf')
	plt.close()
	
	plt.figure()
	plt.loglog(rp_bins_c, DeltaCov_b, 'mo', label='shape noise: real')
	plt.hold(True)
	plt.loglog(rp_bins_c, np.diag(DeltaCov_b_import_CV), 'bo', label='with CV: Fourier')
	plt.xlabel('r_p')
	plt.ylabel('Variance')
	plt.legend()
	plt.savefig('./plots/check_DeltaSigma_var_SNanalytic_'+SURVEY+'_B_1h2h_extl.pdf')
	plt.close()

	# Get the systematic error on boost-1. 
	boosterr_sq_a = ((pa.boost_sys - 1.)*(Boost_a-1.))**2
	boosterr_sq_b = ((pa.boost_sys - 1.)*(Boost_b-1.))**2
	
	gammaIA_stat_cov_withF = np.zeros((len(rp_bins_c), len(rp_bins_c)))
	gammaIA_stat_cov_noF = np.zeros((len(rp_bins_c), len(rp_bins_c)))
	gammaIA_sysB_cov_withF = np.zeros((len(rp_bins_c), len(rp_bins_c)))
	gammaIA_sysB_cov_noF = np.zeros((len(rp_bins_c), len(rp_bins_c)))
	gammaIA_sysZ_cov = np.zeros((len(rp_bins_c), len(rp_bins_c)))
	# Calculate the covariance
	for i in range(0,len((rp_bins_c))):	 
		for j in range(0, len((rp_bins_c))):
			
			# Statistical
			num_term_stat = cz_a_fid**2 * DeltaCov_a_import_CV[i,j] + cz_b_fid**2 * DeltaCov_b_import_CV[i,j]
			denom_term_stat_withF =( ( cz_a_fid * (Boost_a[i] -1. + F_a_fid) * Sig_IA_a_fid[i]) -  ( cz_b_fid * (Boost_b[i] -1. + F_b_fid) * Sig_IA_b_fid[i]) ) * ( ( cz_a_fid * (Boost_a[j] -1. + F_a_fid) * Sig_IA_a_fid[j]) -  ( cz_b_fid * (Boost_b[j] -1. + F_b_fid) * Sig_IA_b_fid[j]) )
			denom_term_stat_noF =( ( cz_a_fid * (Boost_a[i] -1.) * Sig_IA_a_fid[i]) -  ( cz_b_fid * (Boost_b[i] -1.) * Sig_IA_b_fid[i]) ) * ( ( cz_a_fid * (Boost_a[j] -1.) * Sig_IA_a_fid[j]) -  ( cz_b_fid * (Boost_b[j] -1.) * Sig_IA_b_fid[j]) )
			gammaIA_stat_cov_withF[i,j] = num_term_stat / denom_term_stat_withF
			gammaIA_stat_cov_noF[i,j] = num_term_stat / denom_term_stat_noF	
			
			if (i==j):
				
				# Systematic, related to redshifts:
				num_term_sysZ = ( cz_a_fid**2 * DeltaSig_est_a[i]**2 * fudgeczA**2 + cz_b_fid**2 * DeltaSig_est_b[i]**2  * fudgeczB**2 ) / ( ( cz_a_fid * (Boost_a[i] -1. + F_a_fid) * Sig_IA_a_fid[i]) -  ( cz_b_fid * (Boost_b[i] -1. + F_b_fid) * Sig_IA_b_fid[i]) )**2
				denom_term_sysZ = ( ( cz_a_fid * (Boost_a[i] -1. + F_a_fid) * Sig_IA_a_fid[i])**2 * ( fudgeczA**2 + (fudgeFA * F_a_fid)**2 / (Boost_a[i] -1. + F_a_fid)**2 + fudgeSigA**2 ) + ( cz_b_fid * (Boost_b[i] -1. + F_b_fid) * Sig_IA_b_fid[i])**2 * ( fudgeczB**2 + (fudgeFB * F_b_fid)**2 / (Boost_b[i] -1. + F_b_fid)**2 + fudgeSigB**2 ) ) / ( ( cz_a_fid * (Boost_a[i] -1. + F_a_fid) * Sig_IA_a_fid[i]) -  ( cz_b_fid * (Boost_b[i] -1. + F_b_fid) * Sig_IA_b_fid[i]) )**2
				gammaIA_sysZ_cov[i,i] = g_IA_fid[i]**2 * (num_term_sysZ + denom_term_sysZ)
				
				# Systematic, related to boost
				gammaIA_sysB_cov_withF[i,j] = g_IA_fid[i]**2 * ( cz_a_fid**2 * Sig_IA_a_fid[i]**2 * boosterr_sq_a[i] + cz_b_fid**2 * Sig_IA_b_fid[i]**2 * boosterr_sq_b[i] ) / ( ( cz_a_fid * (Boost_a[i] -1. + F_a_fid) * Sig_IA_a_fid[i]) -  ( cz_b_fid * (Boost_b[i] -1. + F_b_fid) * Sig_IA_b_fid[i]) )**2
				gammaIA_sysB_cov_noF[i,j] = g_IA_fid[i]**2 * ( cz_a_fid**2 * Sig_IA_a_fid[i]**2 * boosterr_sq_a[i] + cz_b_fid**2 * Sig_IA_b_fid[i]**2 * boosterr_sq_b[i] ) / ( ( cz_a_fid * (Boost_a[i] -1.) * Sig_IA_a_fid[i]) -  ( cz_b_fid * (Boost_b[i] -1.) * Sig_IA_b_fid[i]) )**2
				
	# For the systematic cases, we need to add off-diagonal elements - we assume fully correlated
	for i in range(0,len((rp_bins_c))):	
		for j in range(0,len((rp_bins_c))):
			if (i != j):
				gammaIA_sysB_cov_withF[i,j] = np.sqrt(gammaIA_sysB_cov_withF[i,i]) * np.sqrt(gammaIA_sysB_cov_withF[j,j])
				gammaIA_sysB_cov_noF[i,j] = np.sqrt(gammaIA_sysB_cov_noF[i,i]) * np.sqrt(gammaIA_sysB_cov_noF[j,j])
				gammaIA_sysZ_cov[i,j]	=	np.sqrt(gammaIA_sysZ_cov[i,i]) * np.sqrt(gammaIA_sysZ_cov[j,j])
		
	# Get the stat + sysB covariance matrix for showing the difference between using excess and using all physically associated galaxies:
	gammaIA_cov_stat_sysB_withF = gammaIA_sysB_cov_withF + gammaIA_stat_cov_withF
	gammaIA_cov_stat_sysB_noF = gammaIA_sysB_cov_noF + gammaIA_stat_cov_noF
	
	# Make a plot of the statistical + boost systematic errors with and without F
	"""if (SURVEY=='LSST_DESI'):
		fig_sub=plt.subplot(111)
		plt.rc('font', family='serif', size=14)
		#fig_sub.axhline(y=0, xmax=20., color='k', linewidth=1)
		#fig_sub.hold(True)
		fig_sub.errorbar(rp_bins_c ,g_IA_fid, yerr = np.sqrt(np.diag(gammaIA_cov_stat_sysB_noF)), fmt='go', linewidth='2', label='Excess only')
		fig_sub.hold(True)
		fig_sub.errorbar(rp_bins_c * 1.05,g_IA_fid, yerr = np.sqrt(np.diag(gammaIA_cov_stat_sysB_withF)), fmt='mo', linewidth='2', label='All physically associated')
		fig_sub.set_xscale("log")
		#fig_sub.set_yscale("log") #, nonposy='clip')
		fig_sub.set_xlabel('$r_p$', fontsize=20)
		fig_sub.set_ylabel('$\gamma_{IA}$', fontsize=20)
		fig_sub.set_ylim(-0.002, 0.002)
		#fig_sub.set_ylim(-0.015, 0.015)
		fig_sub.set_xlim(0.05,20.)
		fig_sub.tick_params(axis='both', which='major', labelsize=18)
		fig_sub.tick_params(axis='both', which='minor', labelsize=18)
		fig_sub.legend()
		plt.tight_layout()
		plt.savefig('./plots/InclAllPhysicallyAssociated_stat+sysB_survey='+SURVEY+'_deltaz='+str(pa.delta_z)+'_1h2h.png')
		plt.close()
	elif (SURVEY=='SDSS'):
		fig_sub=plt.subplot(111)
		plt.rc('font', family='serif', size=14)
		#fig_sub.axhline(y=0, xmax=20., color='k', linewidth=1)
		#fig_sub.hold(True)
		fig_sub.errorbar(rp_bins_c ,g_IA_fid, yerr = np.sqrt(np.diag(gammaIA_cov_stat_sysB_noF)), fmt='go', linewidth='2', label='Excess only')
		fig_sub.hold(True)
		fig_sub.errorbar(rp_bins_c * 1.05,g_IA_fid, yerr = np.sqrt(np.diag(gammaIA_cov_stat_sysB_withF)), fmt='mo', linewidth='2',label='All physically associated')
		fig_sub.set_xscale("log")
		#fig_sub.set_yscale("log") #, nonposy='clip')
		fig_sub.set_xlabel('$r_p$', fontsize=20)
		fig_sub.set_ylabel('$\gamma_{IA}$', fontsize=20)
		#fig_sub.set_ylim(-0.002, 0.005)
		fig_sub.set_ylim(-0.015, 0.015)
		fig_sub.set_xlim(0.05,20.)
		fig_sub.tick_params(axis='both', which='major', labelsize=18)
		fig_sub.tick_params(axis='both', which='minor', labelsize=18)
		fig_sub.legend()
		plt.tight_layout()
		plt.savefig('./plots/InclAllPhysicallyAssociated_stat+sysB_survey='+SURVEY+'_deltaz='+str(pa.delta_z)+'_1h2h.png')
		plt.close()"""
	
	# Now get the sysZ + stat covariance matrix assuming all physically associated galaxies can be subject to IA:
	gamma_IA_cov_tot_withF = gammaIA_stat_cov_withF + gammaIA_sysZ_cov
	
	# Okay, let's compute the Signal to Noise things we want in order to compare statistcal-only signal to noise to that from z-related systematics
	Cov_inv_stat = np.linalg.inv(gammaIA_stat_cov_withF)
	StoNsq_stat = np.dot(g_IA_fid, np.dot(Cov_inv_stat, g_IA_fid))
	
	Cov_inv_tot = np.linalg.inv(gamma_IA_cov_tot_withF)
	StoNsq_tot = np.dot(g_IA_fid, np.dot(Cov_inv_tot, g_IA_fid))
	
	#print "StoNsq_tot=", StoNsq_tot
	#print "StoNsq_stat=", StoNsq_stat
	
	# Now subtract stat from total in quadrature to get sys
	NtoSsq_sys = 1./StoNsq_tot - 1./StoNsq_stat
	
	StoNsq_sys = 1. / NtoSsq_sys
	
	return (StoNsq_stat, StoNsq_sys)
	
def gamma_fid(rp):
	""" Returns the fiducial gamma_IA from a combination of terms from different models which are valid at different scales """
	# TO INCLUDE AN EXTENDED REDSHIFT DISTRIBUTION: THESE WILL NEED TO BE RE-RUN USING THE MODS NOTED IN SHARED_FUNCTIONS_WLP_WLS. 
	wgg_rp = ws.wgg_full(rp, pa.fsat_LRG, pa.fsky, pa.bd_Bl, pa.bs_Bl, './txtfiles/wgg_1h_survey='+pa.survey+'_withHMF.txt', './txtfiles/wgg_2h_survey='+pa.survey+'_kpts='+str(pa.kpts_wgg)+'_update.txt', './plots/wgg_full_Blazek_survey='+pa.survey+'.pdf', SURVEY)
	wgp_rp = ws.wgp_full(rp, pa.bd_Bl, pa.Ai_Bl, pa.ah_Bl, pa.q11_Bl, pa.q12_Bl, pa.q13_Bl, pa.q21_Bl, pa.q22_Bl, pa.q23_Bl, pa.q31_Bl, pa.q32_Bl, pa.q33_Bl, './txtfiles/wgp_1h_ahStopgap_survey='+pa.survey+'.txt','./txtfiles/wgp_2h_AiStopgap_survey='+pa.survey+'.txt', './plots/wgp_full_Blazek_survey='+pa.survey+'.pdf', SURVEY)
	
	gammaIA = wgp_rp / (wgg_rp + 2. * pa.close_cut) 
	
	plt.figure()
	plt.loglog(rp, gammaIA, 'go')
	plt.xlim(0.05,30)
	plt.ylabel('$\gamma_{IA}$')
	plt.xlabel('$r_p$')
	plt.title('Fiducial values of $\gamma_{IA}$')
	plt.savefig('./plots/gammaIA_Blazek_survey='+pa.survey+'_MvirFix.pdf')
	plt.close()
	
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

# Uncomment these lines if you want to load two covariance matrices and check how well they have converged.
#check_covergence()
#exit()

# Set up projected bins
rp_bins 	= 	setup.setup_rp_bins(pa.rp_min, pa.rp_max, pa.N_bins)
rp_cent		=	setup.rp_bins_mid(rp_bins)

# Set up a function to get z as a function of comoving distance
(z_of_com, com_of_z) = setup.z_interpof_com(SURVEY) 

DeltaSigma_theoretical = get_DeltaSig_theory(rp_bins, rp_cent)

#StoNstat = get_gammaIA_cov(rp_bins, rp_cent, 0., 0., 0., 0., 0., 0.)
#print "StoNstat", np.sqrt(StoNstat)
#StoNstat_save = [StoNstat]
#np.savetxt('./txtfiles/StoNstat_Blazek_withCV_SNanalytic_survey='+SURVEY+'rpts=2500.txt', StoNstat_save)
#exit()

StoN_cza = np.zeros(len(pa.fudge_frac_level))
StoN_czb = np.zeros(len(pa.fudge_frac_level))
StoN_Fa = np.zeros(len(pa.fudge_frac_level))
StoN_Fb = np.zeros(len(pa.fudge_frac_level))
StoN_Siga = np.zeros(len(pa.fudge_frac_level))
StoN_Sigb = np.zeros(len(pa.fudge_frac_level))

for i in range(0,len(pa.fudge_frac_level)):

	print "Running, systematic level #"+str(i+1)
	
	# Get the statistical error on gammaIA
	(StoNstat, StoN_cza[i]) = get_gammaIA_cov(rp_bins, rp_cent, pa.fudge_frac_level[i], 0., 0., 0., 0., 0.)
	print "StoNstat, StoNczb=", np.sqrt(StoNstat), np.sqrt(StoN_cza[i])
	(StoNstat, StoN_czb[i]) = get_gammaIA_cov(rp_bins, rp_cent, 0., pa.fudge_frac_level[i], 0., 0., 0., 0.)
	print "StoNstat, StoNczb=", np.sqrt(StoNstat), np.sqrt(StoN_czb[i])
	(StoNstat, StoN_Fa[i])  = get_gammaIA_cov(rp_bins, rp_cent, 0., 0., pa.fudge_frac_level[i], 0., 0., 0.)
	print "StoNstat, StoNFa=", np.sqrt(StoNstat), np.sqrt(StoN_Fa[i])
	(StoNstat, StoN_Fb[i])  = get_gammaIA_cov(rp_bins, rp_cent, 0., 0., 0., pa.fudge_frac_level[i], 0., 0.)
	print "StoNstat, StoNFb=", np.sqrt(StoNstat), np.sqrt(StoN_Fb[i])
	(StoNstat, StoN_Siga[i])= get_gammaIA_cov(rp_bins, rp_cent, 0., 0., 0., 0., pa.fudge_frac_level[i], 0.)
	print "StoNstat, StoNSiga=", np.sqrt(StoNstat), np.sqrt(StoN_Siga[i])
	(StoNstat, StoN_Sigb[i])= get_gammaIA_cov(rp_bins, rp_cent, 0., 0., 0., 0., 0., pa.fudge_frac_level[i])
	print "StoNstat, StoNSiga=", np.sqrt(StoNstat), np.sqrt(StoN_Sigb[i])

# Save the statistical-only S-to-N
StoNstat_save = [StoNstat]
print "StoNstat=", StoNstat
np.savetxt('./txtfiles/StoNstat_Blazek_1h2h_survey='+SURVEY+'_ahAistopgap_deltaz='+str(pa.delta_z)+'.txt', StoNstat_save)

# Save the ratios of S/N sys to stat.	
saveSN_ratios = np.column_stack(( pa.fudge_frac_level, np.sqrt(StoN_cza) / np.sqrt(StoNstat), np.sqrt(StoN_czb) / np.sqrt(StoNstat), np.sqrt(StoN_Fa) / np.sqrt(StoNstat), np.sqrt(StoN_Fb) / np.sqrt(StoNstat), np.sqrt(StoN_Siga) / np.sqrt(StoNstat), np.sqrt(StoN_Sigb) / np.sqrt(StoNstat)))
np.savetxt('./txtfiles/StoN_SysToStat_Blazek_survey='+SURVEY+'_MvirFix_ahAistopgap_deltaz='+str(pa.delta_z)+'.txt', saveSN_ratios)

# Uncomment this to load ratios from file and plot. To plot directly use the below case.
"""frac_levels, StoNratio_sqrt_cza, StoNratio_sqrt_czb, StoNratio_sqrt_Fa,  StoNratio_sqrt_Fb, StoNratio_sqrt_Siga, StoNratio_sqrt_Sigb = np.loadtxt('./plots/SN_ratios.txt', unpack=True)
plt.figure()
plt.loglog(pa.fudge_frac_level, StoNratio_sqrt_cza, 'ko', label='$c_z^a$')
plt.hold(True)
plt.loglog(pa.fudge_frac_level, StoNratio_sqrt_czb, 'mo', label='$c_z^b$')
plt.hold(True)
plt.loglog(pa.fudge_frac_level, StoNratio_sqrt_Fa, 'bo', label='$F_a$')
plt.hold(True)
plt.loglog(pa.fudge_frac_level, StoNratio_sqrt_Fb, 'ro', label='$F_b$')
plt.hold(True)
plt.loglog(pa.fudge_frac_level, StoNratio_sqrt_Siga, 'go', label='$<\\Sigma_{IA}^a>$')
plt.hold(True)
plt.loglog(pa.fudge_frac_level, StoNratio_sqrt_Sigb, 'yo', label='$<\\Sigma_{IA}^b>$')
plt.legend()
plt.xlabel('Fractional error level')
plt.ylabel('$\\frac{S/N_{\\rm sys}}{S/N_{\\rm stat}}$')
plt.xlim(0.005, 10)
plt.ylim(0.01, 1000)
plt.legend()
plt.title('Ratio, S/N, sys vs stat')
plt.savefig('./plots/ratio_StoN.pdf')
plt.close()"""	

# Load the Ncorr information from the other method to include this in the plot:
(frac_level, SNsys_squared_ncorr, SNstat_squared_ncorr) = np.loadtxt('./txtfiles/save_Ncorr_StoNsqSys_survey='+SURVEY+'.txt', unpack=True)

# Make plot of (S/N)_sys / (S/N)_stat as a function of fractional z-related systematic error on each relevant parameter.
if (SURVEY=='SDSS'):
	plt.figure()
	plt.loglog(pa.fudge_frac_level,  np.sqrt(StoNstat) / np.sqrt(StoN_cza), 's', color='#006cc0', label='$c_z^a$')
	plt.hold(True)
	plt.loglog(pa.fudge_frac_level,  np.sqrt(StoNstat) / np.sqrt(StoN_czb), '^',color='#006cc0', label='$c_z^b$')
	plt.hold(True)
	plt.loglog(pa.fudge_frac_level, np.sqrt(SNstat_squared_ncorr) / np.sqrt(SNsys_squared_ncorr), 'mo', label='$N_{\\rm corr}$, $a=0.7$')
	plt.hold(True)
	plt.loglog(pa.fudge_frac_level,  np.sqrt(StoNstat) / np.sqrt(StoN_Fa), 'gs', linewidth='2', label='$F_a$')
	plt.hold(True)
	plt.loglog(pa.fudge_frac_level, np.sqrt(StoNstat) /  np.sqrt(StoN_Fb) , 'g^',linewidth='2', label='$F_b$')
	plt.hold(True)
	plt.loglog(pa.fudge_frac_level, np.sqrt(StoNstat) / np.sqrt(StoN_Siga) , 's',linewidth='2', color='#FFA500', label='$<\\Sigma_{IA}^a>$')
	plt.hold(True)
	plt.loglog(pa.fudge_frac_level,   np.sqrt(StoNstat) / np.sqrt(StoN_Sigb), '^', linewidth='2',color='#FFA500', label='$<\\Sigma_{IA}^b>$')
	plt.hold(True)
	plt.axhline(y=1, color='k', linewidth=2, linestyle='--')
	plt.xlabel('Fractional error', fontsize=25)
	plt.ylabel('$\\frac{S/N_{\\rm stat}}{S/N_{\\rm sys}}$', fontsize=25)
	plt.tick_params(axis='both', which='major', labelsize=18)
	plt.tick_params(axis='both', which='minor', labelsize=18)
	plt.xlim(0.008, 2.)
	plt.ylim(0.00005, 500)
	plt.legend(ncol=3, numpoints=1, fontsize=18)
	plt.tight_layout()
	plt.savefig('./plots/SysNoiseToStatNoise_Blazek_survey='+SURVEY+'_NAM_deltaz='+str(pa.delta_z)+'.png')
	plt.close()
elif(SURVEY=='LSST_DESI'):
	plt.figure()
	plt.loglog(pa.fudge_frac_level, np.sqrt(StoNstat) / np.sqrt(StoN_cza) , 's', color='#006cc0', label='$c_z^a$')
	plt.hold(True)
	plt.loglog(pa.fudge_frac_level, np.sqrt(StoNstat) / np.sqrt(StoN_czb) , '^',color='#006cc0', label='$c_z^b$')
	plt.hold(True)
	plt.loglog(pa.fudge_frac_level,  np.sqrt(StoNstat) /np.sqrt(StoN_Fa), 'gs', label='$F_a$')
	plt.hold(True)
	plt.loglog(pa.fudge_frac_level, np.sqrt(StoNstat) / np.sqrt(StoN_Fb), 'g^', label='$F_b$')
	plt.hold(True)
	plt.loglog(pa.fudge_frac_level,  np.sqrt(StoNstat) / np.sqrt(StoN_Siga), 's', color='#FFA500', label='$<\\Sigma_{IA}^a>$')
	plt.hold(True)
	plt.loglog(pa.fudge_frac_level, np.sqrt(StoNstat) / np.sqrt(StoN_Sigb), '^', color='#FFA500', label='$<\\Sigma_{IA}^b>$')
	plt.hold(True)
	plt.loglog(pa.fudge_frac_level, np.sqrt(SNstat_squared_ncorr) / np.sqrt(SNsys_squared_ncorr), 'mo', label='$N_{\\rm corr}$, $a=0.7$')
	plt.axhline(y=1, color='k', linewidth=2, linestyle='--')
	plt.xlabel('Fractional error', fontsize=25)
	plt.ylabel('$\\frac{S/N_{\\rm stat}}{S/N_{\\rm sys}}$', fontsize=25)
	plt.tick_params(axis='both', which='major', labelsize=18)
	plt.tick_params(axis='both', which='minor', labelsize=18)
	plt.xlim(0.008, 2.)
	plt.legend(ncol=3, numpoints=1, fontsize=18)
	plt.ylim(0.00005, 500)
	plt.tight_layout()
	plt.savefig('./plots/NoiseSysToNoiseStat_Blazek_survey='+SURVEY+'_NAM_deltaz='+str(pa.delta_z)+'.png')
	plt.close()

exit()
# Below this is Fisher matrix stuff - don't worry about it for now.

# Get the parameter derivatives required to construct the Fisher matrix
ders		=	par_derivs(pa.par, rp_cent)

# Get the Fisher matrix
fish 		=	get_Fisher(ders, Cov_gIA)

# If desired, cut parameters which you want to fix from Fisher matrix:
fish_cut 	=	cut_Fisher(fish, None)

# Get the covariance matrix from either fish or fish_cut, and marginalise over any desired parameters
parCov		=	get_par_Cov(fish_cut, None)

# Output whatever we want to know about the parameters:
par_const_output(fish_cut, parCov)
