# This is a script which predicts constraints on intrinsic alignments, using an updated version of the method of Blazek et al 2012 in which it is not assumed that only excess galaxies contribute to IA (rather it is assumed that source galaxies which are close to the lens along the line-of-sight can contribute.)

SURVEY = 'LSST_DESI'
print "SURVEY=", SURVEY
endfile= 'test_updates'

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
	avg_SigC_invsq = np.zeros(len(zL))
	for zi in range(0,len(zL)):
		
		# We require two different normalizations for Nofzph - one for getting the partial neff and another for averaging over SigmaC
		if (photoz_samp =='A'):
			(z_ph_ns, Nofzph_ns) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi], zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
			(z_ph_Sig, Nofzph_Sig) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi], zL[zi] + pa.delta_z, zL[zi], zL[zi] + pa.delta_z, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
			e_rms = pa.e_rms_Bl_a
		elif (photoz_samp == 'B'):
			(z_ph_ns, Nofzph_ns) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi] + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
			(z_ph_Sig, Nofzph_Sig) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi] + pa.delta_z, pa.zphmax, zL[zi] + pa.delta_z, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
			e_rms = pa.e_rms_Bl_b
		elif (photoz_samp == 'full'):
			(z_ph_ns, Nofzph_ns) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
			(z_ph_Sig, Nofzph_Sig) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
			e_rms = pa.e_rms_Bl_full
		elif (photoz_samp == 'src'):
			(z_ph_ns, Nofzph_ns) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi], pa.zphmax, pa.zphmin, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
			(z_ph_Sig, Nofzph_Sig) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
			e_rms = pa.e_rms_Bl_full
		else:
			print "We do not have support for that sample. Exiting."
		
		ns_eff[zi] = pa.n_s * scipy.integrate.simps(Nofzph_ns, z_ph_ns)
		
		Sig_inv_sq = get_SigmaC_inv(z_ph_Sig, zL[zi])**2
		avg_SigC_invsq[zi] = scipy.integrate.simps(Sig_inv_sq * Nofzph_Sig, z_ph_Sig)
		
	ns_avg = scipy.integrate.simps(dndzl * ns_eff, zL)
	Sig_inv_sq_avg = scipy.integrate.simps(avg_SigC_invsq * dndzl, zL)
	
	#print "shape noise without SigC=", e_rms**2/ ( ns_avg * pa.n_l * pa.Area_l * bin_areas  )

	cov = e_rms**2 / Sig_inv_sq_avg  / ( ns_avg * pa.n_l * pa.Area_l * bin_areas  )
	
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
				
					weight = weights(pa.e_rms_Bl_a, z_ph, zL[zi])
					sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
				elif (specz_cut == 'close'):
					(z_ph, dNdz_ph) = N_of_zph(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, zL[zi], zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
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
					(z_ph, dNdz_ph) = N_of_zph(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, zL[zi] + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
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
					(z_ph, dNdz_ph) = N_of_zph(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
					weight = weights(pa.e_rms_Bl_full, z_ph, zL[zi])
				
					sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
				else:
					print "We do not have support for that spec-z cut. Exiting."
					exit()
			elif(photoz_sample == 'src'):
			
				if (specz_cut == 'nocut'):
					(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi], pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
					weight = weights(pa.e_rms_Bl_full, z_ph, zL[zi])
				
					sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
				elif (specz_cut == 'close'):
					(z_ph, dNdz_ph) = N_of_zph(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, zL[zi], pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
					weight = weights(pa.e_rms_Bl_full, z_ph, zL[zi])
				
					sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
				else:
					print "We do not have support for that spec-z cut. Exiting."
					exit()	
			else:
				print "We do not have support for that photo-z sample cut. Exiting."
				print photoz_sample
				exit()
		elif (color_cut=='red'):
			if(photoz_sample=='A'):
				if (specz_cut=='close'):
					z_ph = np.linspace(zL[zi], zL[zi] + pa.delta_z, 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zminclose[zi], zmaxclose[zi], 500, SURVEY)
				elif(specz_cut=='nocut'):
					z_ph = np.linspace(zL[zi], zL[zi] + pa.delta_z, 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype,pa.zsmin, pa.zsmax, 500, SURVEY)
				else:
					print "We do not have support for that spec-z cut. Exiting."
					exit()
			elif(photoz_sample=='B'):
				if (specz_cut=='close'):
					z_ph = np.linspace(zL[zi]+pa.delta_z, pa.zphmax, 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zminclose[zi], zmaxclose[zi], 500, SURVEY)
				elif(specz_cut=='nocut'):
					z_ph = np.linspace(zL[zi]+pa.delta_z, pa.zphmax, 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype,pa.zsmin, pa.zsmax, 500, SURVEY)
				else:
					print "We do not have support for that spec-z cut. Exiting."
					exit()
			elif (photoz_sample=='full'):
				if (specz_cut=='close'):
					z_ph = np.linspace(pa.zphmin, pa.zphmax, 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zminclose[zi], zmaxclose[zi], 500, SURVEY)
				elif (specz_cut=='nocut'):
					z_ph = np.linspace(pa.zphmin, pa.zphmax, 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 500, SURVEY)
				else:
					print "We do not have support for that spec-z cut. Exiting."
					exit()
			elif (photoz_sample=='src'):
				if (specz_cut=='close'):
					z_ph = np.linspace(zL[zi], pa.zphmax, 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zminclose[zi], zmaxclose[zi], 500, SURVEY)
				elif (specz_cut=='nocut'):
					z_ph = np.linspace(zL[zi], pa.zphmax, 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 500, SURVEY)
				else:
					print "We do not have support for that spec-z cut. Exiting."
					exit()
			else:
				print "We do not have support for that photo-z sample. Exiting."
				exit()	
			fred = fred_interp(zs)
					
			zs_integral = np.zeros(len(z_ph))
			for zpi in range(0,len(z_ph)):
				pz = setup.p_z(z_ph[zpi], zs, pa.pzpar_fid, pa.pztype)
				zs_integral[zpi] = scipy.integrate.simps(pz*dNdzs*fred, zs)
			dNdz_fred = zs_integral / norm 
					
			weight = weights(pa.e_rms_mean, z_ph, zL[zi])
			sum_ans_zph[zi] = scipy.integrate.simps(weight * dNdz_fred, z_ph)
	
		else:
			print "We do not have support for that color cut, exiting."
			exit()
			
	# Now sum over lens redshift:
	sum_ans = scipy.integrate.simps(sum_ans_zph * dNdzL, zL)
	
	return sum_ans

def sum_weights_SigC(photoz_sample, specz_cut, color_cut, rp_bins, dNdz_par, pz_par, dNdztype, pztype):
	""" Returns the sum over weights * SigmaC for each projected radial bin. 
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
				
					weightSigC = weights_times_SigC(pa.e_rms_Bl_a, z_ph, zL[zi])
				
					sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weightSigC, z_ph)
				
				elif (specz_cut == 'close'):
					(z_ph, dNdz_ph) = N_of_zph(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, zL[zi], zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
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
					(z_ph, dNdz_ph) = N_of_zph(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, zL[zi] + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
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
					(z_ph, dNdz_ph) = N_of_zph(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
					weightSigC = weights_times_SigC(pa.e_rms_Bl_full, z_ph, zL[zi])
				
					sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weightSigC, z_ph)
				
				else:
					print "We do not have support for that spec-z cut. Exiting."
					exit()
		
			elif(photoz_sample == 'src'):
			
				if (specz_cut == 'nocut'):
					(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi], pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
					weightSigC = weights_times_SigC(pa.e_rms_Bl_full, z_ph, zL[zi])
				
					sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weightSigC, z_ph)
				
				elif (specz_cut == 'close'):
					(z_ph, dNdz_ph) = N_of_zph(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, zL[zi], pa.zphmax, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
					weightSigC = weights_times_SigC(pa.e_rms_Bl_full, z_ph, zL[zi])
				
					sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weightSigC, z_ph)
				
				else:
					print "We do not have support for that spec-z cut. Exiting."
					exit()
		
			
		
			else:
				print "We do not have support for that photo-z sample cut. Exiting."
				print photoz_sample
				exit()
		elif (color_cut=='red'):
			if(photoz_sample=='A'):
				if (specz_cut=='close'):
					z_ph = np.linspace(zL[zi], zL[zi] + pa.delta_z, 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zminclose[zi], zmaxclose[zi], 500, SURVEY)
				elif(specz_cut=='nocut'):
					z_ph = np.linspace(zL[zi], zL[zi] + pa.delta_z, 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype,pa.zsmin, pa.zsmax, 500, SURVEY)
				else:
					print "We do not have support for that spec-z cut. Exiting."
					exit()
			elif(photoz_sample=='B'):
				if (specz_cut=='close'):
					z_ph = np.linspace(zL[zi]+pa.delta_z, pa.zphmax, 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zminclose[zi], zmaxclose[zi], 500, SURVEY)
				elif(specz_cut=='nocut'):
					z_ph = np.linspace(zL[zi]+pa.delta_z, pa.zphmax, 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype,pa.zsmin, pa.zsmax, 500, SURVEY)
				else:
					print "We do not have support for that spec-z cut. Exiting."
					exit()
			elif (photoz_sample=='full'):
				if (specz_cut=='close'):
					z_ph = np.linspace(pa.zphmin, pa.zphmax, 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zminclose[zi], zmaxclose[zi], 500, SURVEY)
				elif (specz_cut=='nocut'):
					z_ph = np.linspace(pa.zphmin, pa.zphmax, 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 500, SURVEY)
				else:
					print "We do not have support for that spec-z cut. Exiting."
					exit()
					
			elif (photoz_sample=='src'):
				if (specz_cut=='close'):
					z_ph = np.linspace(zL[zi], pa.zphmax, 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zminclose[zi], zmaxclose[zi], 500, SURVEY)
				elif (specz_cut=='nocut'):
					z_ph = np.linspace(zL[zi], pa.zphmax, 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 500, SURVEY)
				else:
					print "We do not have support for that spec-z cut. Exiting."
					exit()
			else:
				print "We do not have support for that photo-z sample. Exiting."
				exit()	
			fred = fred_interp(zs)
					
			zs_integral = np.zeros(len(z_ph))
			for zpi in range(0,len(z_ph)):
				pz = setup.p_z(z_ph[zpi], zs, pa.pzpar_fid, pa.pztype)
				zs_integral[zpi] = scipy.integrate.simps(pz*dNdzs*fred, zs)
			dNdz_fred = zs_integral / norm 
					
			weight_SigC = weights_times_SigC(pa.e_rms_mean, z_ph, zL[zi])
			sum_ans_zph[zi] = scipy.integrate.simps(weight_SigC * dNdz_fred, z_ph)
	
		else:
			print "We do not have support for that color cut, exiting."
			exit()
	
	
	# Now sum over lens redshift:
	sum_ans = scipy.integrate.simps(sum_ans_zph * dNdzL, zL)
	
	return sum_ans
	
def get_boost(rp_cents_, sample):
	"""Returns the boost factor in radial bins. propfact is a tunable parameter giving the proportionality constant by which boost goes like projected correlation function (= value at 1 Mpc/h). """

	#propfact = np.loadtxt('./txtfiles/boosts/Boost_'+str(sample)+'_survey='+str(SURVEY)+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')
	Boost = np.loadtxt('./txtfiles/boosts/Boost_full_'+str(sample)+'_survey='+str(SURVEY)+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt') + np.ones((len(rp_cents_)))

	#Boost = propfact *(rp_cents_)**(-0.8) + np.ones((len(rp_cents_))) # Empirical power law fit to the boost, derived from the fact that the boost goes like projected correlation function.

	return Boost
	
def get_F(photoz_sample, rp_bins_, dNdz_par, pz_par, dNdztype, pztype):
	""" Returns F (the weighted fraction of lens-source pairs from the smooth dNdz which are contributing to IA) """

	# Sum over `rand-close'
	numerator = sum_weights(photoz_sample, 'close', 'all', rp_bins_, dNdz_par, pz_par, dNdztype, pztype)

	#Sum over all `rand'
	denominator = sum_weights(photoz_sample, 'nocut', 'all', rp_bins_, dNdz_par, pz_par, dNdztype, pztype)

	F = np.asarray(numerator) / np.asarray(denominator)

	return F

def get_cz(photoz_sample, rp_bins_, dNdzpar, pzpar, dNdztype, pztype):
	""" Returns the value of the photo-z bias parameter c_z"""
	
	# Now get the mean of tilde(weight) tilde(Sigma_c) Sigma_c^{-1} 
	bSigw = get_bSig_zs(photoz_sample, dNdzpar, dNdztype, pzpar, pztype)

	cz = 1. / (bSigw)

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
		
		# Get angular diameter distance to sources using photometric redshift ditributions
		Ds_photo = com_of_z(zph) / (1. + zph)
		
		bsw[zi] = np.zeros(len(zph))
			
		# Loop over photometric redshift distances.
		for zpi in range(0,len(zph)):
			
			if (Ds_photo[zpi]>Dlvec[zi]):
				Dls_photo = Ds_photo[zpi] - Dlvec[zi]
			else:
				Dls_photo = 0.
		
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
	
def get_bSig_zs(photoz_samp, dNdzpar, dNdztype, pzpar, pztype):
	""" Returns an interpolating function for tilde(w)tilde(SigC)SigC^{-1} as a function of source photo-z. Used in computing photo-z bias c_z / 1+b_z."""
	
	if (pztype != 'Gaussian'):
		print "Photo-z type "+str(pztype)+" is not supported. Exiting."
		exit()

	# Get lens redshift distribution
	zLvec = scipy.linspace(pa.zLmin, pa.zLmax, 100) # Using 100 points here (and in sum-weights) is sufficient for convergence to well below a 10^{-3} level. August 15, 2017.
	dNdzl = setup.get_dNdzL(zLvec, SURVEY)
	Dlvec = com_of_z(zLvec) / (1. + zLvec)
	
	# Get the spectroscopic source redshift distribution
	zs_dNdzs, dNdzs_unnormed = setup.get_NofZ_unnormed(dNdzpar, dNdztype, pa.zsmin, pa.zsmax, 1000, SURVEY)
	norm = np.sum(dNdzs_unnormed)
	dNdzs = dNdzs_unnormed / norm
	
	# Sample an array of zs which show up with probability distribution dNdzs
	# Using 10,000 points here is enough to get accurate values to <1%. November 21, 2017
	zslist = np.random.choice(zs_dNdzs, p=dNdzs, size = 10000)
	
	Ds_spec = com_of_z(zslist) / (1.+ zslist)

	# Loop over each zl.
	numerator = np.zeros(len(zLvec))
	denominator = np.zeros(len(zLvec))
	counter = 0
	for zi in range(0,len(zLvec)):
		
		# Get photo-z limits for each bin
		if (photoz_samp == 'A'):
			zphmin = zLvec[zi]; zphmax = zLvec[zi] + pa.delta_z
			#zphmin = pa.zsmin; zphmax = pa.zsmax
		elif (photoz_samp == 'B'):
			zphmin = zLvec[zi] + pa.delta_z; zphmax = pa.zphmax
		else:
			print "We don't have support for that photo-z sample in computing photo-z bias yet."
			exit()
			
		# Loop over spectrosopic source redshifts.
		for zsi in range(0,len(zslist)):

			# Draw a point from the Gaussian pz. This is the photo-z for this spec-z
			zph = np.random.normal(zslist[zsi], pzpar[0]*(1. + zslist[zsi]))
			
			if ((zph>=zphmin) and (zph<=zphmax)):
				if (counter == 0):
					zs_hold_list = [zslist[zsi]]
					zph_hold_list = [zph]
					Ds_photo_list = [com_of_z(zph) / (1. + zph)]
					Dls_photo_list = [Ds_photo_list[0] - Dlvec[zi]]
					Ds_spec_list = [com_of_z(zslist[zsi]) / (1. + zslist[zsi])]
					Dls_spec_list = [Ds_spec_list[0] - Dlvec[zi]]
					counter+=1
				else:
					zs_hold_list.append(zslist[zsi])
					zph_hold_list.append(zph)
					Ds_photo_list.append(com_of_z(zph) / (1. + zph))
					Dls_photo_list.append(Ds_photo_list[counter] - Dlvec[zi])
					Ds_spec_list.append(com_of_z(zslist[zsi]) / (1. + zslist[zsi]))
					Dls_spec_list.append(Ds_spec_list[counter] - Dlvec[zi])
					counter+=1		
		zs_hold = np.asarray(zs_hold_list)
		zph_hold = np.asarray(zph_hold_list)
		Ds_photo = np.asarray(Ds_photo_list)
		Dls_photo = np.asarray(Dls_photo_list)
		Ds_spec = np.asarray(Ds_spec_list)
		Dls_spec = np.asarray(Dls_spec_list)
				
		# Sum up over all the zspec samples to get the numerator and denominator at this lens redshift:
		numerator[zi] = (1. +zLvec[zi])**4 * Dlvec[zi]**2 * np.sum((Dls_spec/ Ds_spec* Dls_photo / Ds_photo))
		denominator[zi] = (1. +zLvec[zi])**4 * Dlvec[zi]**2 * np.sum(Dls_photo**2 / Ds_photo**2)
		
	# Integrate over zl
	num_integrate_zl= scipy.integrate.simps(numerator * dNdzl, zLvec)
	denom_integrate_zl= scipy.integrate.simps(denominator * dNdzl, zLvec)	
		
	# Integrate over zL
	bsw_return = num_integrate_zl / denom_integrate_zl

	return bsw_return
	
def get_Sig_IA(photoz_sample, rp_bins_, boost, dNdz_par, pz_par, dNdztype, pztype):
	""" Returns the value of <\Sigma_c>_{IA} in radial bins. """
	
	# There are four terms here. The two in the denominators are sums over randoms (or sums over lenses that can be written as randoms * boost), and these are already set up to calculate.
	denom_rand_close =  sum_weights(photoz_sample, 'close', 'all', rp_bins_, dNdz_par, pz_par, dNdztype, pztype)
	denom_rand =  sum_weights(photoz_sample, 'nocut', 'all', rp_bins_, dNdz_par, pz_par, dNdztype, pztype)
	denom_excess = (boost - 1.) * denom_rand
	
	# The two in the numerator require summing over weights and Sigma_C. 
	
	#For the sum over rand-close in the numerator, this follows directly from the same type of expression as when summing weights:
	num_rand_close = sum_weights_SigC(photoz_sample, 'close', 'all', rp_bins, dNdz_par, pz_par, dNdztype, pztype)

	# The other numerator is a sum over excess. We have to get the normalization indirectly so there are a bunch of terms here. See notes.
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

	return Sig_IA 

def get_Sig_ex(photoz_sample, rp_bins_, boost, dNdz_par, pz_par, dNdztype, pztype):
	""" This function gets the average Sigmac over (red) excess galaxies, to compare the original method with the new method of assuming all physically associated galaxies are subject to IA."""
	
	# Sum over weights in the denominator
	denom_rand =  sum_weights(photoz_sample, 'nocut', 'all', rp_bins_, dNdz_par, pz_par, dNdztype, pztype)
	denom_excess = (boost - 1.) * denom_rand
	
	# The numerator a sum over excess of SigmaC. We have to get the normalization indirectly so there are a bunch of terms here. See notes.
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
	
	# Sigma_C_inv is in units of pc^2 / (h Msol) (comoving), so Sig_ex is in units of h Msol / pc^2 (comoving).
	Sig_ex = (np.asarray(num_excess)) / (np.asarray(denom_excess)) 

	return Sig_ex

def get_est_DeltaSig(boost, F, cz, SigIA, g_IA_fid):
	""" Returns the value of tilde Delta Sigma in bins"""
		
	EstDeltaSig = np.asarray(DeltaSigma_theoretical) / cz + (boost-1.+ F) * SigIA * g_IA_fid

	return EstDeltaSig
		
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

def get_DeltaSig_theory(rp_bins, rp_bins_c):
	""" Returns the theoretical value of Delta Sigma in bin using projection over the NFW profile and over the 2-pt correlation function at larger scales.
	
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
	
	if SURVEY=='SDSS':
		r_hf, corr_hf = np.loadtxt('./txtfiles/halofit_xi/xi_2h_zavg_'+SURVEY+'_'+endfile+'.txt', unpack=True)
	elif SURVEY=='LSST_DESI':
		r_hf, corr_hf = np.loadtxt('./txtfiles/halofit_xi/xi_2h_zavg_'+SURVEY+'_'+endfile+'.txt', unpack=True)
	else:
		print "We don't have support for that survey, exiting."
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

def get_quants(rp_bins_c, rp_bins):
	""" Run quantities that take a while. """
	
	Boost_a = get_boost(rp_bins_c, 'A')
	Boost_b = get_boost(rp_bins_c, 'B')
	
	################ F's #################
	if (os.path.isfile('./txtfiles/F/F_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')):
		print "F values have already been computed for this endfile."
	else:
		F_a_fid = get_F('A', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
		F_b_fid = get_F('B', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
		print "F_a_fid=", F_a_fid , "F_b_fid=", F_b_fid
	
		save_F = np.column_stack(([F_a_fid], [F_b_fid]))
		np.savetxt('./txtfiles/F/F_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', save_F)
	
	############# Sig_IA's ##############
	if (os.path.isfile('./txtfiles/SigIA/Sig_IA_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')):
		print "SigIA values have already been computed for this endfile."
	else:
		Sig_IA_a_fid = get_Sig_IA('A', rp_bins, Boost_a,  pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
		Sig_IA_b_fid = get_Sig_IA('B', rp_bins, Boost_b,  pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
		print "Sig_IA_a_fid=", Sig_IA_a_fid
		print "Sig_IA_b_fid=", Sig_IA_b_fid
	
		save_SigIA = np.column_stack((rp_bins_c, Sig_IA_a_fid, Sig_IA_b_fid))
		np.savetxt('./txtfiles/SigIA/Sig_IA_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', save_SigIA)
		
	############# Sig_ex's ##############
	if (os.path.isfile('./txtfiles/Sig_ex/Sig_ex_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')):
		print "SigEx values have already been computed for this endfile."
	else:
		Sig_ex_a_fid = get_Sig_ex('A', rp_bins, Boost_a,  pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
		Sig_ex_b_fid = get_Sig_ex('B', rp_bins, Boost_b,  pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
		print "Sig_ex_a_fid=", Sig_ex_a_fid
		print "Sig_ex_b_fid=", Sig_ex_b_fid
		save_Sigex = np.column_stack((rp_bins_c, Sig_ex_a_fid, Sig_ex_b_fid))
		np.savetxt('./txtfiles/Sig_ex_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', save_Sigex)

	############ c_z's ##############
	if (os.path.isfile('./txtfiles/cz/cz_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')):
		print "cz values have already been computed for this endfile."
	else:
		# Photometric biases to estimated Delta Sigmas, fiducial
		cz_a_fid = get_cz('A', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
		cz_b_fid = get_cz('B', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
		#cz_all_fid = get_cz('full', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
		print "cz_a_fid =", cz_a_fid, "cz_b_fid=", cz_b_fid
		
		save_cz = np.column_stack(([cz_a_fid], [cz_b_fid]))
		np.savetxt('./txtfiles/cz/cz_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', save_cz)
		#save_cz_full = [cz_all_fid]
		#np.savetxt('./txtfiles/cz_fullfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'.txt', save_cz_full)
	
	return
	

##### ERRORS FOR FRACTIONAL ERROR CALCULATION #####
	
def get_gammaIA_cov_stat_sysB(rp_bins, rp_bins_c):
	""" Gets the covariance matrix of gamma_{IA} in projected radial bins (stat + sysB only)."""

	# Get the fiducial quantities required for the statistical error variance terms. #	
	Boost_a = get_boost(rp_bins_c, 'A')
	Boost_b = get_boost(rp_bins_c, 'B')
	
	############ gamma_IA ###########
		
	# Load stuff if we haven't computed it this time around:
	(F_a_fid, F_b_fid) = np.loadtxt('./txtfiles/F/F_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', unpack=True)
	(rp_bins_c, Sig_IA_a_fid, Sig_IA_b_fid) = np.loadtxt('./txtfiles/SigIA/Sig_IA_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', unpack=True)
	(rp_bins_c, Sig_ex_a_fid, Sig_ex_b_fid) = np.loadtxt('./txtfiles/Sig_ex_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', unpack=True)
	(cz_a_fid, cz_b_fid) = np.loadtxt('./txtfiles/cz/cz_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', unpack=True)
	
	# Estimated Delta Sigmas
	DeltaSig_est_a = get_est_DeltaSig(Boost_a, F_a_fid, cz_a_fid, Sig_IA_a_fid, g_IA_fid)
	DeltaSig_est_b = get_est_DeltaSig(Boost_b, F_b_fid, cz_b_fid, Sig_IA_b_fid, g_IA_fid)
	
	# Import the covariance matrix of Delta Sigma for each sample as calculated from Fourier space in a different script, w / cosmic variance terms
	DeltaCov_a_import_CV = np.loadtxt('./txtfiles/covmats/cov_DelSig_zLext_'+SURVEY+'_sample=A_rpts2000_lpts100000_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')
	DeltaCov_b_import_CV = np.loadtxt('./txtfiles/covmats/cov_DelSig_zLext_'+SURVEY+'_sample=B_rpts2000_lpts100000_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')
	
	# Uncomment the following section to plot comparison of diagonal covariance elements against shape-noise only real space case.
	"""DeltaCov_a_import_CV = np.diag(DeltaCov_a)
	DeltaCov_b_import_CV = np.diag(DeltaCov_b)
	DeltaCov_full_import_CV = np.diag(DeltaCov_full)
	
	plt.figure()
	plt.loglog(rp_bins_c, DeltaCov_a, 'mo', label='shape noise: real')
	plt.hold(True)
	plt.loglog(rp_bins_c, np.diag(DeltaCov_a_import_CV), 'go', label='with CV: Fourier')
	plt.xlabel('r_p')
	plt.ylabel('Variance')
	plt.legend()
	plt.savefig('./plots/check_DeltaSigma_'+SURVEY+'_A_fixSigC.pdf')
	plt.close()
	
	plt.figure()
	plt.loglog(rp_bins_c, DeltaCov_b, 'mo', label='shape noise: real')
	plt.hold(True)
	plt.loglog(rp_bins_c, np.diag(DeltaCov_b_import_CV), 'bo', label='with CV: Fourier')
	plt.xlabel('r_p')
	plt.ylabel('Variance')
	plt.legend()
	plt.savefig('./plots/check_DeltaSigma_'+SURVEY+'_B_fixSigC.pdf')
	plt.close()"""
	
	boosterr_sq_a = pa.boost_sys**2
	boosterr_sq_b = pa.boost_sys**2
	
	gammaIA_stat_cov_withF = np.zeros((len(rp_bins_c), len(rp_bins_c)))
	gammaIA_stat_cov_noF = np.zeros((len(rp_bins_c), len(rp_bins_c)))
	gammaIA_sysB_cov_withF = np.zeros((len(rp_bins_c), len(rp_bins_c)))
	gammaIA_sysB_cov_noF = np.zeros((len(rp_bins_c), len(rp_bins_c)))
	# Calculate the statistical covariance and the systematic related to the boost
	for i in range(0,len((rp_bins_c))):	 
		for j in range(0, len((rp_bins_c))):
			
			# Statistical
			num_term_stat = cz_a_fid**2 * DeltaCov_a_import_CV[i,j] + cz_b_fid**2 * DeltaCov_b_import_CV[i,j]
			
			denom_term_stat_withF =( ( cz_a_fid * ( Boost_a[i] -1. + F_a_fid) * Sig_IA_a_fid[i]) -  ( cz_b_fid * ( Boost_b[i] -1. + F_b_fid) * Sig_IA_b_fid[i]) ) * ( ( cz_a_fid * (Boost_a[j] -1. + F_a_fid) * Sig_IA_a_fid[j]) -  ( cz_b_fid * ( Boost_b[j] -1.+ F_b_fid) * Sig_IA_b_fid[j]) )
			
			denom_term_stat_noF =( ( cz_a_fid * (Boost_a[i] -1.) * Sig_ex_a_fid[i]) -  ( cz_b_fid * (Boost_b[i] -1.) * Sig_ex_b_fid[i]) ) * ( ( cz_a_fid * (Boost_a[j] -1.) * Sig_ex_a_fid[j])  -  ( cz_b_fid * (Boost_b[j] -1.) * Sig_ex_b_fid[j]) )
			
			gammaIA_stat_cov_withF[i,j] = num_term_stat / denom_term_stat_withF
			gammaIA_stat_cov_noF[i,j] = num_term_stat / denom_term_stat_noF	
			
			if (i==j):
				
				# Systematic, related to boost
				gammaIA_sysB_cov_withF[i,j] = g_IA_fid[i]**2 * ( cz_a_fid**2 * Sig_IA_a_fid[i]**2 * boosterr_sq_a + cz_b_fid**2 * Sig_IA_b_fid[i]**2 * boosterr_sq_b ) / ( ( cz_a_fid * (Boost_a[i] -1. + F_a_fid) * Sig_IA_a_fid[i]) -  ( cz_b_fid * ( Boost_b[i] -1. + F_b_fid) * Sig_IA_b_fid[i]) )**2
				
				gammaIA_sysB_cov_noF[i,j] = g_IA_fid[i]**2 * ( cz_a_fid**2 * Sig_ex_a_fid[i]**2 * boosterr_sq_a + cz_b_fid**2 * Sig_ex_b_fid[i]**2 * boosterr_sq_b ) / ( ( cz_a_fid * (Boost_a[i] -1.) * Sig_ex_a_fid[i]) -  ( cz_b_fid * (Boost_b[i] -1.) * Sig_ex_b_fid[i]) )**2
		
	# For the systematic cases, we need to add off-diagonal elements - we assume fully correlated
	for i in range(0,len((rp_bins_c))):	
		for j in range(0,len((rp_bins_c))):
			if (i != j):
				gammaIA_sysB_cov_withF[i,j] = np.sqrt(gammaIA_sysB_cov_withF[i,i]) * np.sqrt(gammaIA_sysB_cov_withF[j,j])
				gammaIA_sysB_cov_noF[i,j] = np.sqrt(gammaIA_sysB_cov_noF[i,i]) * np.sqrt(gammaIA_sysB_cov_noF[j,j])
		
	# Get the stat + sysB covariance matrix for showing the difference between using excess and using all physically associated galaxies:
	gammaIA_cov_stat_sysB_withF = gammaIA_sysB_cov_withF + gammaIA_stat_cov_withF
	gammaIA_cov_stat_sysB_noF = gammaIA_sysB_cov_noF + gammaIA_stat_cov_noF
		
	# Get the signal-to-noise attributable to statistical + boost systematic errors assuming only excess and excess + rand, per R bin and output
	StoN_perbin_stat_withF = g_IA_fid  /  np.sqrt(np.diag(gammaIA_stat_cov_withF))
	StoN_perbin_stat_sysb_noF = g_IA_fid  /  np.sqrt(np.diag(gammaIA_cov_stat_sysB_noF))
	StoN_perbin_stat_sysb_withF = g_IA_fid /  np.sqrt(np.diag(gammaIA_cov_stat_sysB_withF))

	save_ston_stat = np.column_stack((rp_bins_c, StoN_perbin_stat_withF))
	save_ston_sysb_stat = np.column_stack((rp_bins_c, StoN_perbin_stat_sysb_withF))
	np.savetxt('./txtfiles/StoN/StoN_sysb_stat_Blazek_'+SURVEY+'_deltaz='+str(pa.delta_z)+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt', save_ston_sysb_stat)
	np.savetxt('./txtfiles/StoN/StoN_stat_Blazek_'+SURVEY+'_deltaz='+str(pa.delta_z)+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt', save_ston_stat)


	if (SURVEY=='LSST_DESI'):
		plt.rc('font', family='serif', size=20)
		fig_sub=plt.subplot(111)
		fig_sub.scatter(rp_bins_c ,StoN_perbin_stat_sysb_noF, color='#FFA500', marker='o', s=100, label='Excess only')
		fig_sub.hold(True)
		fig_sub.scatter(rp_bins_c ,StoN_perbin_stat_sysb_withF,color='b', marker='^', s=100, label='All physically associated')
		fig_sub.set_xscale("log")
		fig_sub.set_xlabel('$r_p,\, {\\rm Mpc/h}$', fontsize=30)
		fig_sub.set_ylabel('$\\frac{S}{N}$', fontsize=30)
		fig_sub.set_ylim(0, 5.)
		fig_sub.set_xlim(0.05,20.)
		fig_sub.tick_params(axis='both', which='major', labelsize=18)
		fig_sub.tick_params(axis='both', which='minor', labelsize=18)
		fig_sub.legend(loc='upper left', fontsize=16)
		plt.title('LSST+DESI', fontsize=30)
		plt.tight_layout()
		plt.savefig('./plots/FvNoF_stat_sysB_'+SURVEY+'_rlim='+str(pa.mlim)+'_'+endfile+'.png')
		plt.close()
	elif (SURVEY=='SDSS'):
		plt.rc('font', family='serif', size=20)
		fig_sub=plt.subplot(111)
		fig_sub.scatter(rp_bins_c ,StoN_perbin_stat_sysb_noF, color='#FFA500', marker='o', s=100,  label='Excess only')
		fig_sub.hold(True)
		fig_sub.scatter(rp_bins_c ,StoN_perbin_stat_sysb_withF,color='b', marker='^', s=100,  label='All physically associated')
		fig_sub.set_xscale("log")
		fig_sub.set_xlabel('$r_p,\, {\\rm Mpc/h}$', fontsize=30)
		fig_sub.set_ylabel('$\\frac{S}{N}$', fontsize=30)
		fig_sub.set_ylim(0, 1.0)
		fig_sub.set_xlim(0.05,20.)
		fig_sub.tick_params(axis='both', which='major', labelsize=18)
		fig_sub.tick_params(axis='both', which='minor', labelsize=18)
		fig_sub.legend(loc='upper left', fontsize=16)
		plt.title('SDSS', fontsize=30)
		plt.tight_layout()
		plt.savefig('./plots/FvNoF_stat_sysB_'+SURVEY+'_'+endfile+'.png')
		plt.close()
	
	# Compute the Signal to Noise things we want in order to save stat + sysB signal to noise for comparison with our method.
	
	Cov_inv_stat = np.linalg.inv(gammaIA_stat_cov_withF)
	StoNsq_stat = np.dot( g_IA_fid , np.dot(Cov_inv_stat, g_IA_fid))
	
	Cov_inv_stat_sysB = np.linalg.inv(gammaIA_cov_stat_sysB_withF)
	StoNsq_stat_sysB = np.dot(g_IA_fid, np.dot(Cov_inv_stat_sysB, g_IA_fid))

	return (gammaIA_stat_cov_withF, StoNsq_stat, StoNsq_stat_sysB)
	
def get_gammaIAcov_sys(rp_bins_c, gammaIA_stat_cov_withF, fudgeczA, fudgeczB, fudgeFA, fudgeFB, fudgeSigA, fudgeSigB):
	""" Get the covaraince related to systematic errors"""
	
	Boost_a = get_boost(rp_bins_c, 'A')
	Boost_b = get_boost(rp_bins_c, 'B')
	
	# Load stuff if we haven't computed it this time around:
	(F_a_fid, F_b_fid) = np.loadtxt('./txtfiles/F/F_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', unpack=True)
	(rp_bins_c, Sig_IA_a_fid, Sig_IA_b_fid) = np.loadtxt('./txtfiles/SigIA/Sig_IA_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', unpack=True)
	(rp_bins_c, Sig_ex_a_fid, Sig_ex_b_fid) = np.loadtxt('./txtfiles/Sig_ex_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', unpack=True)
	(cz_a_fid, cz_b_fid) = np.loadtxt('./txtfiles/cz/cz_afid_bfid_extl_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', unpack=True)

	# Estimated Delta Sigmas
	DeltaSig_est_a = get_est_DeltaSig(Boost_a, F_a_fid, cz_a_fid, Sig_IA_a_fid, g_IA_fid)
	DeltaSig_est_b = get_est_DeltaSig(Boost_b, F_b_fid, cz_b_fid, Sig_IA_b_fid, g_IA_fid)
	
	gammaIA_sysZ_cov = np.zeros((len(rp_bins_c), len(rp_bins_c)))
	# Calculate the sysz covariance 
	for i in range(0,len((rp_bins_c))):	 
		for j in range(0, len((rp_bins_c))):
			if (i==j):
				num_term_sysZ = ( cz_a_fid**2 * DeltaSig_est_a[i]**2 * fudgeczA**2 + cz_b_fid**2 * DeltaSig_est_b[i]**2  * fudgeczB**2 ) / ( (( cz_a_fid * ( Boost_a[i] -1.+ F_a_fid) * Sig_IA_a_fid[i]) -  ( cz_b_fid * ( Boost_b[i] -1. + F_b_fid) * Sig_IA_b_fid[i]) ) * g_IA_fid[i] )**2
				
				denom_term_sysZ = ( ( cz_a_fid * ( Boost_a[i] -1.+ F_a_fid) * Sig_IA_a_fid[i])**2 * ( fudgeczA**2 + (fudgeFA * F_a_fid)**2 / ( Boost_a[i] -1. + F_a_fid)**2 + fudgeSigA**2 ) + ( cz_b_fid * ( Boost_b[i] -1.+ F_b_fid) * Sig_IA_b_fid[i])**2 * ( fudgeczB**2 + (fudgeFB * F_b_fid)**2 / ( Boost_b[i] -1+ F_b_fid)**2 + fudgeSigB**2 ) ) / ( ( cz_a_fid * ( Boost_a[i] -1. + F_a_fid) * Sig_IA_a_fid[i]) -  ( cz_b_fid * ( Boost_b[i] -1.+ F_b_fid) * Sig_IA_b_fid[i]) )**2
				
				gammaIA_sysZ_cov[i,i] = g_IA_fid[i]**2 * (num_term_sysZ + denom_term_sysZ)
		
	# For the systematic cases, we need to add off-diagonal elements - we assume fully correlated
	for i in range(0,len((rp_bins_c))):	
		for j in range(0,len((rp_bins_c))):
			if (i != j):
				gammaIA_sysZ_cov[i,j]	=	np.sqrt(gammaIA_sysZ_cov[i,i]) * np.sqrt(gammaIA_sysZ_cov[j,j])
	
	# Now get the sysZ + stat covariance matrix assuming all physically associated galaxies can be subject to IA:
	gamma_IA_cov_sysZ_stat_withF = gammaIA_stat_cov_withF + gammaIA_sysZ_cov
	
	# Compute the Signal to Noise things we want in order to compare statistcal-only signal to noise to that from z-related systematics

	Cov_inv_stat = np.linalg.inv(gammaIA_stat_cov_withF)
	StoNsq_stat = np.dot( g_IA_fid , np.dot(Cov_inv_stat, g_IA_fid))
	
	Cov_inv_sysZ_stat =  np.linalg.pinv(gamma_IA_cov_sysZ_stat_withF, rcond = 10**(-15)) 
					
	StoNsq_sysZ_stat = np.dot(g_IA_fid  , np.dot(Cov_inv_sysZ_stat, g_IA_fid ))
	
	# Subtract stat from sysz+stat in quadrature to get sys
	NtoSsq_sys = (1./StoNsq_sysZ_stat - 1./StoNsq_stat)
	StoNsq_sys = 1. / NtoSsq_sys			
	
	return	StoNsq_sys

def get_sys_range():
	"""Get the maximum value of sigma(c_z_b) / c_z_b which is allowed for the systematic error from this source to make up at most 50% of the integrated inverse S/N OF THE LENSING SIGNAL.
	This delimits the regime of systematic errors of interest (if we can't do a lensing measurements, we don't care about IA."""
	
	# First check if we've already done this calculation, if not don't do it again.
	# Load the statistical covariance matrix for the b sample - remember this is the covariance of Estimated Delta Sigma.
	cov_DS_stat = np.loadtxt('./txtfiles/covmats/cov_DelSig_zLext_'+SURVEY+'_sample=B_rpts2000_lpts100000_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')
		
	# Get the estimated Delta Sigma (b sample) from lensing only
	(cz_a_fid, cz_b_fid) = np.loadtxt('./txtfiles/cz/cz_afid_bfid_extl_survey='+SURVEY+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', unpack=True)
		
	# Get the square of the statistical-only signal-to-noise 
	stat_StoNsq = np.dot(DeltaSigma_theoretical, np.dot(np.linalg.inv(cz_b_fid **2 * cov_DS_stat), DeltaSigma_theoretical))
		
	#Get the covariance matrix due to systematic error on cz for a variety of $\frac{\sigma(c_z)}{c_z}$ values. 
	frac_err_sys = np.zeros(len(pa.fudge_frac_level))
	for fi in range(0, len(pa.fudge_frac_level)):
		cov_DS_sys = np.zeros((len(rp_cent), len(rp_cent)))
		for rpi in range(0,len(rp_cent)):
			for rpj in range(0,len(rp_cent)):
				cov_DS_sys[rpi, rpj] = pa.fudge_frac_level[fi]**2 * DeltaSigma_theoretical[rpi] * DeltaSigma_theoretical[rpj]
		#Combine with the statistical covariance matrix to get the total matrix, and from there get the signal to noise due to systematic error on c_z     
		cov_tot = cov_DS_sys + cov_DS_stat
		StoN_tot_sq = np.dot(DeltaSigma_theoretical, np.dot(np.linalg.inv(cov_tot), DeltaSigma_theoretical))
		NtoS_sys = np.sqrt( StoN_tot_sq**(-1) - stat_StoNsq**(-1))
		frac_err_sys[fi] = NtoS_sys / (np.sqrt(StoN_tot_sq)**(-1))
			
	# Now check at which fractional error value we exceed 50% of the error budget from systematics.
	ind = next(j[0] for j in enumerate(frac_err_sys) if j[1]>=0.5)
	frac_err_max = [pa.fudge_frac_level[ind]]
	np.savetxt('./txtfiles/sysfracmax_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', frac_err_max)
		
	# pass out the maximum fractional error so we can use it to get the worst case scenario sys/stat ratio	
		
	return pa.fudge_frac_level[ind]

def get_max_ratio(frac_err):
	""" Now get the maximum possible value of the ratio of redshift-related systematic error to statistical error given all quantities subject to sysz error take the max possible level"""
	
	# Call the systematic error function for all quantities taking this fractional error.
	StoN_sys_sq =   get_gammaIAcov_sys(rp_cent,gammaIA_stat_cov_withF, frac_err, frac_err, frac_err, frac_err, frac_err, frac_err)
	
	# StoNstat is squared StoN it's just badly named
	max_ratio = [np.sqrt(StoNstat) / np.sqrt(StoN_sys_sq)]
	np.savetxt('./txtfiles/maxsysratio_Blazek_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', max_ratio)
	
	return
		

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
	print("We don't have support for that survey yet; exiting.")
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

# Get fiducial gamma_IA
g_IA_fid = gamma_fid(rp_cent)

# Run F, cz, and SigIA if not already computed
get_quants(rp_cent, rp_bins)

# Get the real-space shape-noise-only covariance matrices for Delta Sigma for each sample if we want to compare against them (pass this to get_IA_cov)

#DeltaCov_a = shapenoise_cov('A', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
#DeltaCov_b = shapenoise_cov('B', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
#DeltaCov_full = shapenoise_cov('src', rp_bins, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)

# Get the statistical + systematics due to boost signal to noise and covariance matrix
(gammaIA_stat_cov_withF, StoNstat, StoN_stat_sysB) = get_gammaIA_cov_stat_sysB(rp_bins, rp_cent)

# Save the statistical-only S-to-N and from stat + sysB  
StoNstat_save = [StoNstat]; np.savetxt('./txtfiles/StoN/StoNstat_Blazek_'+SURVEY+'_deltaz='+str(pa.delta_z)+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt', StoNstat_save)  
StoNstat_sysB_save = [StoN_stat_sysB]; np.savetxt('./txtfiles/StoN/StoN_stat_sysB_Blazek_'+SURVEY+'_deltaz='+str(pa.delta_z)+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt', StoNstat_sysB_save)

# Get signal to noise with different levels of sysz on different quantities.
StoN_cza = np.zeros(len(pa.fudge_frac_level)); StoN_czb = np.zeros(len(pa.fudge_frac_level)); StoN_Fa = np.zeros(len(pa.fudge_frac_level))
StoN_Fb = np.zeros(len(pa.fudge_frac_level)); StoN_Siga = np.zeros(len(pa.fudge_frac_level)); StoN_Sigb = np.zeros(len(pa.fudge_frac_level))

for i in range(0,len(pa.fudge_frac_level)):
	print "Running, systematic level #"+str(i+1)
	StoN_cza[i]	=	get_gammaIAcov_sys(rp_cent, gammaIA_stat_cov_withF, pa.fudge_frac_level[i], 0., 0., 0., 0., 0.)
	StoN_czb[i] =	get_gammaIAcov_sys(rp_cent, gammaIA_stat_cov_withF, 0., pa.fudge_frac_level[i], 0., 0., 0., 0.)
	StoN_Fa[i]  = 	get_gammaIAcov_sys(rp_cent, gammaIA_stat_cov_withF,0., 0., pa.fudge_frac_level[i], 0., 0., 0.)
	StoN_Fb[i]  =	get_gammaIAcov_sys(rp_cent,  gammaIA_stat_cov_withF,0., 0., 0., pa.fudge_frac_level[i], 0., 0.)
	StoN_Siga[i]=   get_gammaIAcov_sys(rp_cent,gammaIA_stat_cov_withF, 0., 0., 0., 0., pa.fudge_frac_level[i], 0.)
	StoN_Sigb[i]=   get_gammaIAcov_sys(rp_cent,gammaIA_stat_cov_withF, 0., 0., 0., 0., 0., pa.fudge_frac_level[i])

# Save the sysZ stuff	
saveSN_sysZ = np.column_stack(( pa.fudge_frac_level, StoN_cza, StoN_czb, StoN_Fa, StoN_Fb, StoN_Siga, StoN_Sigb))
np.savetxt('./txtfiles/StoN/StoN_SysToStat_Blazek_'+SURVEY+'_deltaz='+str(pa.delta_z)+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt', saveSN_sysZ)

# Get the maximum value of sigma(c_z_b) / c_z_b which is allowed for the systematic error from this source to make up at most 50% of the integrated inverse S/N OF THE LENSING SIGNAL.
# This delimits the regime of systematic errors of interest (if we can't do a lensing measurements, we don't care about IA.
frac_err_max = get_sys_range()

# Now get the maximum possible value of the ratio of redshift-related systematic error to statistical error given all quantities subject to sysz error take the max possible level
get_max_ratio(frac_err_max)
