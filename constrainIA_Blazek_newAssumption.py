# This is a script which predicts constraints on intrinsic alignments, using an updated version of the method of Blazek et al 2012 in which it is not assumed that only excess galaxies contribute to IA (rather it is assumed that source galaxies which are close to the lens along the line-of-sight can contribute.)

import numpy as np
import params as pa
import scipy
import scipy.integrate
import matplotlib.pyplot as plt
import scipy.interpolate
import subprocess
import shutil
import shared_functions_setup as setup
import shared_functions_wlp_wls as ws


########## FUNCTIONS ##########

	
############## DNDZ ###############
	
def N_of_zph(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, alpha_dN, zs_dN, sigz):
	""" Returns dNdz_ph, the number density in terms of photometric redshift, defined over photo-z range (z_a_def_ph, z_b_def_ph), normalized over the photo-z range (z_a_norm_ph, z_b_norm_ph), normalized over the spec-z range (z_a_norm, z_b_norm), and defined on the spec-z range (z_a_def, z_b_def)"""
	
	(z, dNdZ) = setup.get_NofZ_unnormed(alpha_dN, zs_dN, z_a_def_s, z_b_def_s, pa.zpts)
	(z_norm, dNdZ_norm) = setup.get_NofZ_unnormed(alpha_dN, zs_dN, z_a_norm_s, z_b_norm_s, pa.zpts)
	
	z_ph_vec = scipy.linspace(z_a_def_ph, z_b_def_ph, 5000)
	z_ph_vec_norm = scipy.linspace(z_a_norm_ph, z_b_norm_ph, 5000)
	
	int_dzs = np.zeros(len(z_ph_vec))
	for i in range(0,len(z_ph_vec)):
		int_dzs[i] = scipy.integrate.simps(dNdZ*setup.p_z(z_ph_vec[i], z, sigz), z)
	
	int_dzs_norm = np.zeros(len(z_ph_vec_norm))
	for i in range(0,len(z_ph_vec_norm)):
		int_dzs_norm[i] = scipy.integrate.simps(dNdZ_norm*setup.p_z(z_ph_vec_norm[i], z_norm, sigz), z_norm)
		
	norm = scipy.integrate.simps(int_dzs_norm, z_ph_vec_norm)
	
	return (z_ph_vec, int_dzs / norm)

################### THEORETICAL VALUES FOR FRACTIONAL ERROR CALCULATION ########################333

def sigma_e(z_s_, s_to_n):
	""" Returns a value for the model for the per-galaxy noise as a function of source redshift"""

	# This is a dummy things for now
	if hasattr(z_s_, "__len__"):
		sig_e = 2. / s_to_n * np.ones(len(z_s_))
	else:
		sig_e = 2. / s_to_n

	return sig_e

def weights(e_rms, z_, z_l_):
	
	""" Returns the inverse variance weights as a function of redshift. """
	
	SigC_t_inv = get_SigmaC_inv(z_, z_l_)
	
	weights = SigC_t_inv**2/(sigma_e(z_, pa.S_to_N)**2 + e_rms**2 * np.ones(len(z_)))
	
	return weights
		
def weights_times_SigC(e_rms, z_, z_l_):
	""" Returns the inverse variance weights as a function of redshift. """
	
	SigC_t_inv = get_SigmaC_inv(z_, z_l_)
	
	weights = SigC_t_inv/(sigma_e(z_, pa.S_to_N)**2 + e_rms**2 * np.ones(len(z_)))
	
	return weights

def sum_weights(z_l, z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, erms, rp_bins_, rp_bin_c_, alpha_dN, zs_dN, sigz):
	""" Returns the sum over rand-source pairs of the estimated weights, in each projected radial bin. Pass different z_min_s and z_max_s to get rand-close, rand-far, and all-rand cases."""
	
	(z_ph, dNdz_ph) = N_of_zph(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, alpha_dN, zs_dN, sigz)
	
	norm = scipy.integrate.simps(dNdz_ph, z_ph)
	
	sum_ans = [0]*(len(rp_bins_)-1)
	for i in range(0,len(rp_bins_)-1):
		Integrand = dNdz_ph* weights(erms, z_ph, z_l)
		sum_ans[i] = scipy.integrate.simps(Integrand, z_ph)

	return sum_ans
	
def sum_weights_SigC(z_l, z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, erms, rp_bins_, rp_bin_c_, alpha_dN, zs_dN, sigz):
	""" Returns the sum over rand-source pairs of the estimated weights multiplied by estimated SigC, in each projected radial bin. Pass different z_min_s and z_max_s to get rand-close, rand-far, and all-rand cases."""
	
	(z_ph, dNdz_ph) = N_of_zph(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, alpha_dN, zs_dN, sigz)

	sum_ans = [0]*(len(rp_bins_)-1)
	for i in range(0,len(rp_bins_)-1):
		Integrand = dNdz_ph* weights_times_SigC(erms, z_ph, z_l) 
		sum_ans[i] = scipy.integrate.simps(Integrand, z_ph)

	return sum_ans

def get_bSigW(z_p_min, z_p_max, e_rms, sigz):
	""" Returns an interpolating function for tilde(w)tilde(SigC)SigC^{-1} as a function of source photo-z. Used in computing photo-z bias c_z / 1+b_z."""
	
	# Define a vector of photometric redshifts on the range of source redshifts we care about for this sample:
	z_p_vec = np.logspace(np.log10(z_p_min), np.log10(z_p_max),100)
	
	# Get the angular diameter distances to these photo-z's of sources, and to the lens z.
	Ds_photo = setup.com(z_p_vec) / (1. + z_p_vec)
	Dl = setup.com(pa.zeff) / (1. + pa.zeff)
	
	bsw = np.zeros(len(z_p_vec))
	for zi in range(0,len(z_p_vec)):
		
		# Draw a set of points from a normal dist with mean zspec and variance pa.sigz^2. These are the spec-z's for this photo-z.
		zsvec = np.random.normal(z_p_vec[zi], sigz, 10000)
		
		# Get the components of the terms we care about which depend on the spec-z's.
		Ds_spec = setup.com(zsvec) / (1. + zsvec)
		Dls = np.zeros(len(zsvec))
		for zsi in range(0,len(zsvec)):
			if (zsvec[zsi]>pa.zeff):
				Dls[zsi] = Ds_spec[zsi] - Dl
			else:
				Dls[zsi] = 0.
						
		# Find the mean bsigma at this zphoto
		bsw[zi] = (4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc)**2   / (e_rms**2 + sigma_e(z_p_vec[zi], pa.S_to_N)**2) * (1. +pa.zeff)**4 * Dl**2 * (Ds_photo[zi]-Dl) / Ds_photo[zi] * np.mean(Dls / Ds_spec)
	
	# Interpolate the mean bsigmas such that we can report at any zspec in the range:
	bsig_interp = scipy.interpolate.interp1d(z_p_vec, bsw)

	return bsig_interp

def get_SigmaC_inv(z_s_, z_l_):
	""" Returns the theoretical value of 1/Sigma_c, (Sigma_c = the critcial surface mass density) """

	com_s = setup.com(z_s_) 
	com_l = setup.com(z_l_) 

	# Get scale factors for converting between angular-diameter and comoving distances.
	a_l = 1. / (z_l_ + 1.)
	a_s = 1. / (z_s_ + 1.)
	
	D_s = a_s * com_s # Angular diameter source distance.
	D_l = a_l * com_l # Angular diameter lens distance
	D_ls = (D_s - D_l) 
	
	# Units are pc^2 / (h Msun), comoving
	Sigma_c_inv = 4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc *   D_l * D_ls * (1 + z_l_)**2 / D_s
	
	if hasattr(z_s_, "__len__"):
		for i in range(0,len(z_s_)):
			if(z_s_[i]<=z_l_):
				Sigma_c_inv[i] = 0.
	else:
		if (z_s_<=z_l_):
			Sigam_c_inv = 0.

	return Sigma_c_inv

def get_boost(rp_cents_, propfact):
	"""Returns the boost factor in radial bins. propfact is a tunable parameter giving the proportionality constant by which boost goes like projected correlation function (= value at 1 Mpc/h). """

	Boost = propfact *(rp_cents_)**(-0.8) + np.ones((len(rp_cents_))) # Empirical power law fit to the boost, derived from the fact that the boost goes like projected correlation function.

	return Boost

def get_F(erms, zph_min_samp, zph_max_samp, rp_bins_, rp_bin_c, alpha_dN, zs_dN, sigz):
	""" Returns F (the weighted fraction of lens-source pairs from the smooth dNdz which are contributing to IA) """

	# Sum over `rand-close'
	numerator = sum_weights(pa.zeff, z_close_low, z_close_high, pa.zsmin, pa.zsmax, zph_min_samp, zph_max_samp, zph_min_samp, zph_max_samp, erms, rp_bins_, rp_bin_c, alpha_dN, zs_dN, sigz)

	#Sum over all `rand'
	denominator = sum_weights(pa.zeff, pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zph_min_samp, zph_max_samp, zph_min_samp, zph_max_samp, erms, rp_bins_, rp_bin_c, alpha_dN, zs_dN, sigz)

	F = np.asarray(numerator) / np.asarray(denominator)

	return F

def get_cz(z_l, z_ph_min, z_ph_max, erms, rp_bins_, rp_bins_c, alpha_dN, zs_dN, sigz):
	""" Returns the value of the photo-z bias parameter c_z"""

	# The denominator (of 1+bz) is just a sum over tilde(weights) of all random-source pairs
	# (This outputs a bunch of things that are all the same, one for each rp bin, so we just take the first one.)
	denominator = sum_weights(z_l, pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, z_ph_min, z_ph_max, z_ph_min, z_ph_max, erms, rp_bins_, rp_bins_c, alpha_dN, zs_dN, sigz)[0] 
	
	# The numerator (of 1+bz) is a sum over tilde(weights) tilde(Sigma_c) Sigma_c^{-1} of all random-source pair.s
	
	# Get dNdzph
	(zph, dNdzph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, z_ph_min, z_ph_max, z_ph_min, z_ph_max, alpha_dN, zs_dN, sigz)
	
	# Now get a function of photo-z which gives the mean of tilde(weight) tilde(Sigma_c) Sigma_c^{-1} as a function of photo-z.
	bsigW_interp = get_bSigW(z_ph_min, z_ph_max, erms, sigz)
	
	numerator = scipy.integrate.simps(bsigW_interp(zph) * dNdzph, zph)

	cz = denominator / numerator

	return cz

def get_Sig_IA(z_l, z_ph_min_samp, z_ph_max_samp, erms, rp_bins_, rp_bin_c_, boost, alpha_dN, zs_dN, sigz):
	""" Returns the value of <\Sigma_c>_{IA} in radial bins """
	
	# There are four terms here. The two in the denominators are sums over randoms (or sums over lenses that can be written as randoms * boost), and these are already set up to calculate.
	denom_rand_close = sum_weights(z_l, z_close_low, z_close_high, pa.zsmin, pa.zsmax, z_ph_min_samp, z_ph_max_samp, z_ph_min_samp, z_ph_max_samp, erms, rp_bins_, rp_bin_c_, alpha_dN, zs_dN, sigz)
	denom_rand = sum_weights(z_l, pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, z_ph_min_samp, z_ph_max_samp, z_ph_min_samp, z_ph_max_samp, erms, rp_bins_, rp_bin_c_, alpha_dN, zs_dN, sigz)
	denom_excess = (boost - 1.) * denom_rand
	
	# The two in the numerator require summing over weights and Sigma_C. 
	
	#For the sum over rand-close in the numerator, this follows directly from the same type of expression as when summing weights:
	num_rand_close = sum_weights_SigC(z_l, z_close_low, z_close_high, pa.zsmin, pa.zsmax, z_ph_min_samp, z_ph_max_samp, z_ph_min_samp, z_ph_max_samp, erms, rp_bins_, rp_bin_c_, alpha_dN, zs_dN, sigz)
			
	# The other numerator sum is a term which represents the a sum over excess. We have to get the normalization indirectly so there are a bunch of terms here. See notes.
	# We assume all excess galaxies are at the lens redshift.
	
	# We first compute a sum over excess of weights and Sigma_C with arbitrary normalization:
	z_ph = scipy.linspace(z_ph_min_samp, z_ph_max_samp, 5000)
	exc_wSigC_arbnorm = scipy.integrate.simps(weights_times_SigC(erms, z_ph, z_l) * setup.p_z(z_ph, z_l, sigz), z_ph)
	
	# We do the same for a sum over excess of just weights with the same arbitrary normalization:
	exc_w_arbnorm = scipy.integrate.simps(weights(erms, z_ph, z_l) * setup.p_z(z_ph, z_l, sigz), z_ph)
	
	# We already have an appropriately normalized sum over excess weights, from above (denom_excess), via the relationship with the boost.
	# Put these components together to get the appropriately normalized sum over excess of weights and SigmaC:
	
	num_excess = exc_wSigC_arbnorm / exc_w_arbnorm * np.asarray(denom_excess)
	
	# Sigma_C_inv is in units of pc^2 / (h Msol) (comoving), so Sig_IA is in units of h Msol / pc^2 (comoving).
	Sig_IA = (np.asarray(num_excess) + np.asarray(num_rand_close)) / (np.asarray(denom_excess) + np.asarray(denom_rand_close)) 

	return Sig_IA  

def get_est_DeltaSig(z_l, rp_bins, rp_bins_c, boost, F, cz, SigIA, g_IA_fid):
	""" Returns the value of tilde Delta Sigma in bins"""
	
	# The first term is (1 + b_z) \Delta \Sigma ^{theory}, need theoretical Delta Sigma	
	DS_the = get_DeltaSig_theory(z_l, rp_bins, rp_bins_c)
		
	EstDeltaSig = np.asarray(DS_the) / cz + (boost-1.+ F) * SigIA * g_IA_fid
	
	#plt.figure()
	#plt.loglog(rp_bins_c, EstDeltaSig, 'g+', label='1-halo')
	#plt.xlim(0.05,20)
	#plt.ylim(0.3,200)
	#plt.xlabel('$r_p$, Mpc/h')
	#plt.ylabel('$\Delta \Sigma$, $h M_\odot / pc^2$')
	#plt.legend()
	#plt.savefig('./plots/test_EstDeltaSigmatot_b.png')
	#plt.close()

	return EstDeltaSig

def get_DeltaSig_theory(z_l, rp_bins, rp_bins_c):
	""" Returns the theoretical value of Delta Sigma in bin using projection over the NFW profile and over the 2-pt correlation function at larger scales, rather than using the lensing-related definition."""
	
	###### First get the term from halofit (valid at larger scales) ######
	# Import the correlation function as a function of R and Pi, obtained via getting P(k) from CAMB and then using FFT_log, Anze Slozar version. 
	# Note that since CAMB uses comoving distances, all distances here should be comoving. rpvec and Pivec are in Mpc/h.
	corr = np.loadtxt('./txtfiles/corr_2d_z='+str(z_l)+'_Singh2014params.txt')
	rpvec = np.loadtxt('./txtfiles/corr_rp_z='+str(z_l)+'.txt')
	Pivec = np.loadtxt('./txtfiles/corr_delta_z='+str(z_l)+'.txt') 
	
	# Get rho_m in comoving coordinates (independent of redshift)
	rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h (comoving distances)
	rho_m = pa.OmM * rho_crit # units of Msol h^2 / Mpc^3 (comoving distances)
	
	# Get Sigma(R) for the 2halo term.
	Sigma_HF = np.zeros(len(rpvec))
	for ri in range(0,len(rpvec)):
		# This will have units Msol h / Mpc^2 in comoving distances.
		Sigma_HF[ri] = rho_m * scipy.integrate.simps(corr[:, ri], Pivec)  
		
	# Now average Sigma_HF(R) over R to get the first averaged term in Delta Sigma
	barSigma_HF = np.zeros(len(rpvec))
	for ri in range(0,len(rpvec)):
		barSigma_HF[ri] = 2. / rpvec[ri]**2 * scipy.integrate.simps(rpvec[0:ri+1]**2*Sigma_HF[0:ri+1], np.log(rpvec[0:ri+1]))
	
	# Units Msol h / Mpc^2 (comoving distances).
	DeltaSigma_HF = barSigma_HF - Sigma_HF
	
	####### Now get the 1 halo term (valid at smaller scales) #######
	
	rvec = np.logspace(-7, 4, 10000)
	
	rho = ws.rho_NFW(rvec, pa.Mvir, z_l) # In units of Msol h^2 / Mpc^3 
	
	rho_interp = scipy.interpolate.interp1d(rvec, rho)

	rho_2D = np.zeros((len(rpvec), len(Pivec)))
	for ri in range(0, len(rpvec)):
		for pi in range(0, len(Pivec)):
			rho_2D[ri, pi] = rho_interp(np.sqrt(rpvec[ri]**2 + Pivec[pi]**2))
	
	Sigma_1h = np.zeros(len(rpvec))
	for ri in range(0,len(rpvec)):
		# Units Msol h / Mpc^2, comoving distances
		Sigma_1h[ri] = scipy.integrate.simps(rho_2D[ri, :], Pivec)
		
	#plt.figure()
	#plt.loglog(rpvec, Sigma_1h/ 10.**12, 'm+') # Plot in Msol h / pc^2.
	#plt.xlim(0.0003, 8)
	#plt.ylim(0.1,10**4)
	#plt.savefig('./plots/Sigma_1h.png')
	#plt.close()
	
	# Now average over R to get the first averaged term in Delta Sigma
	barSigma_1h = np.zeros(len(rpvec))
	for ri in range(0,len(rpvec)):
		barSigma_1h[ri] = 2. / rpvec[ri]**2 * scipy.integrate.simps(rpvec[0:ri+1]**2*Sigma_1h[0:ri+1], np.log(rpvec[0:ri+1]))
	
	# Units Msol h / Mpc^2, comoving distances.
	DeltaSigma_1h = barSigma_1h - Sigma_1h
	
	#plt.figure()
	#plt.loglog(rpvec, DeltaSigma_1h  / (10**12), 'g+', label='1-halo')
	#plt.hold(True)
	#plt.loglog(rpvec, DeltaSigma_HF  / (10**12), 'm+', label='halofit')
	#plt.hold(True)
	#plt.loglog(rpvec, (DeltaSigma_HF + DeltaSigma_1h)  / (10**12), 'k+', label='total')
	#plt.xlim(0.05,20)
	#plt.ylim(0.3,200)
	#plt.xlabel('$r_p$, Mpc/h')
	#plt.ylabel('$\Delta \Sigma$, $h M_\odot / pc^2$')
	#plt.legend()
	#plt.savefig('./plots/test_DeltaSigmatot_Feb28_Singh2014.png')
	#plt.close()
	
	# Interpolate and output at r_bins_c:
	ans_interp = scipy.interpolate.interp1d(rpvec, (DeltaSigma_1h + DeltaSigma_HF) / (10**12))
	ans = ans_interp(rp_bins_c)
	
	return ans # outputting as Msol h / pc^2 

##### ERRORS FOR FRACTIONAL ERROR CALCULATION #####

def shapenoise_cov(e_rms, z_p_l, z_p_h, B_samp, rp_c, rp, alpha_dN, zs_dN, sigz):
	""" Returns a diagonal covariance matrix in bins of projected radius for a measurement dominated by shape noise. Elements are 1 / (sum_{ls} w), carefully normalized, in each bin. """
	
	# Get the area of each projected radial bin in square arcminutes
	bin_areas       =       setup.get_areas(rp, pa.zeff)
	
	sum_denom =  pa.n_l * pa.Area_l * bin_areas * pa.n_s * np.asarray(sum_weights(pa.zeff, pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, z_p_l, z_p_h, pa.zphmin, pa.zphmax, e_rms, rp, rp_c, alpha_dN, zs_dN, sigz))
	
	cov = 1. / (B_samp * sum_denom)
	
	return cov
	
def boost_errors(rp_bins_c, filename):
	""" Imports a file with 2 columns, [rp (kpc/h), sigma(boost-1)]. Interpolates and returns the value of the error on the boost at the center of each bin. """
	
	(rp_kpc, boost_error_raw) = np.loadtxt(filename, unpack=True)
	
	# Convert the projected radius to Mpc/h
	rp_Mpc = rp_kpc / 1000.
	
	interpolate_boost_error = scipy.interpolate.interp1d(rp_Mpc, boost_error_raw)
	
	boost_error = interpolate_boost_error(rp_bins_c)
	
	return boost_error

def get_gammaIA_cov(rp_bins, rp_bins_c):
	""" Takes information about the uncertainty on constituent elements of gamma_{IA} and combines them to get the covariance matrix of gamma_{IA} in projected radial bins."""
	""" We are only interested right now in the diagonal elements of the covariance matrix, so we assume it is diagonal. """ 

	# Get the fiducial quantities required for the statistical error variance terms. #	
	# Boosts
	Boost_a = get_boost(rp_cent, pa.boost_close)
	Boost_b = get_boost(rp_cent, pa.boost_far)
	
	################ F's #################
	
	# F factors - first, fiducial
	F_a_fid = get_F(pa.e_rms_Bl_a, pa.zeff, pa.zeff+pa.delta_z, rp_bins, rp_bins_c, pa.alpha_fid, pa.zs_fid, pa.sigz_fid)
	F_b_fid = get_F(pa.e_rms_Bl_b, pa.zeff+pa.delta_z, pa.zphmax, rp_bins, rp_bins_c, pa.alpha_fid, pa.zs_fid, pa.sigz_fid)
	#F_a_fid = np.zeros(len(rp_bins_c))
	#F_b_fid = np.zeros(len(rp_bins_c))
	print "F_a_fid=", F_a_fid[0], "F_b_fid=", F_b_fid[0]
	
	
	# Now, the F for the systematic error associated with the dNdz
	F_a_dN = get_F(pa.e_rms_Bl_a, pa.zeff, pa.zeff+pa.delta_z, rp_bins, rp_bins_c, pa.alpha_sys, pa.zs_sys, pa.sigz_fid)
	F_b_dN = get_F(pa.e_rms_Bl_b, pa.zeff+pa.delta_z, pa.zphmax, rp_bins, rp_bins_c, pa.alpha_sys, pa.zs_sys, pa.sigz_fid)
	#F_a_dN = np.zeros(len(rp_bins_c))
	#F_b_dN = np.zeros(len(rp_bins_c))
	print "F_a_dN=", F_a_dN[0], "F_b_dN=", F_b_dN[0]
	
	# Now the F for systematic error associate with p_z
	F_a_pz = get_F(pa.e_rms_Bl_a, pa.zeff, pa.zeff+pa.delta_z, rp_bins, rp_bins_c, pa.alpha_fid, pa.zs_fid, pa.sigz_sys)
	F_b_pz = get_F(pa.e_rms_Bl_b, pa.zeff+pa.delta_z, pa.zphmax, rp_bins, rp_bins_c, pa.alpha_fid, pa.zs_fid, pa.sigz_sys)
	#F_a_pz = np.zeros(len(rp_bins_c))
	#F_b_pz = np.zeros(len(rp_bins_c))
	print "F_a_pz=", F_a_pz[0], "F_b_pz=", F_b_pz[0]
	
	############# Sig_IA's ##############
	
	# Sig IA - first, fiducial
	Sig_IA_a_fid = get_Sig_IA(pa.zeff, pa.zeff, pa.zeff+pa.delta_z, pa.e_rms_Bl_a, rp_bins, rp_bins_c, Boost_a, pa.alpha_fid, pa.zs_fid, pa.sigz_fid)
	Sig_IA_b_fid = get_Sig_IA(pa.zeff, pa.zeff+pa.delta_z, pa.zphmax, pa.e_rms_Bl_b, rp_bins, rp_bins_c, Boost_b, pa.alpha_fid, pa.zs_fid, pa.sigz_fid)
	print "Sig_IA_a_fid=", Sig_IA_a_fid
	print "Sig_IA_b_fid=", Sig_IA_b_fid
	
	# Sig IA - for systematic error associated with the dNdz
	Sig_IA_a_dN = get_Sig_IA(pa.zeff, pa.zeff, pa.zeff+pa.delta_z, pa.e_rms_Bl_a, rp_bins, rp_bins_c, Boost_a, pa.alpha_sys, pa.zs_sys, pa.sigz_fid)
	Sig_IA_b_dN = get_Sig_IA(pa.zeff, pa.zeff+pa.delta_z, pa.zphmax, pa.e_rms_Bl_b, rp_bins, rp_bins_c, Boost_b, pa.alpha_sys, pa.zs_sys, pa.sigz_fid)
	print "Sig_IA_a_dN=", Sig_IA_a_dN
	print "Sig_IA_b_dN=", Sig_IA_b_dN
	
	# Sig IA - for systematic error associated with the p_z
	Sig_IA_a_pz = get_Sig_IA(pa.zeff, pa.zeff, pa.zeff+pa.delta_z, pa.e_rms_Bl_a, rp_bins, rp_bins_c, Boost_a, pa.alpha_fid, pa.zs_fid, pa.sigz_sys)
	Sig_IA_b_pz = get_Sig_IA(pa.zeff, pa.zeff+pa.delta_z, pa.zphmax, pa.e_rms_Bl_b, rp_bins, rp_bins_c, Boost_b, pa.alpha_fid, pa.zs_fid, pa.sigz_sys)
	print "Sig_IA_a_pz=", Sig_IA_a_pz
	print "Sig_IA_b_pz=", Sig_IA_b_pz
	
	############ c_z's ##############
	
	# Photometric biases to estimated Delta Sigmas, fiducial
	cz_a_fid = get_cz(pa.zeff, pa.zeff, pa.zeff + pa.delta_z, pa.e_rms_Bl_a, rp_bins, rp_bins_c, pa.alpha_fid, pa.zs_fid, pa.sigz_fid)
	cz_b_fid = get_cz(pa.zeff, pa.zeff+pa.delta_z, pa.zphmax, pa.e_rms_Bl_b, rp_bins, rp_bins_c, pa.alpha_fid, pa.zs_fid, pa.sigz_fid)
	print "cz_a_fid =", cz_a_fid, "cz_b_fid=", cz_b_fid
	
	# Photometric biases for systematic errors associated with the dNdz
	cz_a_dN = get_cz(pa.zeff, pa.zeff, pa.zeff + pa.delta_z, pa.e_rms_Bl_a, rp_bins, rp_bins_c, pa.alpha_sys, pa.zs_sys, pa.sigz_fid)
	cz_b_dN = get_cz(pa.zeff, pa.zeff+pa.delta_z, pa.zphmax, pa.e_rms_Bl_b, rp_bins, rp_bins_c, pa.alpha_sys, pa.zs_sys, pa.sigz_fid)
	print "cz_a_dN =", cz_a_dN, "cz_b_dN=", cz_b_dN
	
	# Photometric biases for systematic errors associated with p_z
	cz_a_pz = get_cz(pa.zeff, pa.zeff, pa.zeff + pa.delta_z, pa.e_rms_Bl_a, rp_bins, rp_bins_c, pa.alpha_fid, pa.zs_fid, pa.sigz_sys)
	cz_b_pz = get_cz(pa.zeff, pa.zeff+pa.delta_z, pa.zphmax, pa.e_rms_Bl_b, rp_bins, rp_bins_c, pa.alpha_fid, pa.zs_fid, pa.sigz_sys)
	print "cz_a_pz =", cz_a_pz, "cz_b_pz=", cz_b_pz
	
	############ gamma_IA ###########
	
	# gamma_IA_fiducial, from model
	g_IA_fid = gamma_fid(rp_bins_c)
	
	# Estimated Delta Sigmas
	DeltaSig_est_a = get_est_DeltaSig(pa.zeff, rp_bins, rp_bins_c, Boost_a, F_a_fid, cz_a_fid, Sig_IA_a_fid, g_IA_fid)
	DeltaSig_est_b = get_est_DeltaSig(pa.zeff, rp_bins, rp_bins_c, Boost_b, F_b_fid, cz_b_fid, Sig_IA_b_fid, g_IA_fid)
	
	# gamma_IA from our quantities, for the fiducial case 
	#gamma_fid_from_q = gamma_fid_from_quants(rp_bins_c, Boost_a, Boost_b, F_a_fid, F_b_fid, Sig_IA_a_fid, Sig_IA_b_fid, cz_a_fid, cz_b_fid, DeltaSig_est_a, DeltaSig_est_b)
	
	############ Get statistical error ############
	
	# These are the shape-noise-dominated diagonal covariance matrices associated with each sample. Units: Msol^2 h / pc^2, comoving.
	DeltaCov_a = shapenoise_cov(pa.e_rms_Bl_a, pa.zeff, pa.zeff+pa.delta_z, Boost_a, rp_bins_c, rp_bins, pa.alpha_fid, pa.zs_fid, pa.sigz_fid)
	DeltaCov_b = shapenoise_cov(pa.e_rms_Bl_b, pa.zeff+pa.delta_z, pa.zphmax, Boost_b, rp_bins_c, rp_bins, pa.alpha_fid, pa.zs_fid, pa.sigz_fid)
	
	# Get the statistical errors on B-1 for each sample from file
	sigBa = boost_errors(rp_bins_c, pa.sigBF_a)
	sigBb = boost_errors(rp_bins_c, pa.sigBF_b)
	
	gammaIA_stat_cov = np.zeros((len(rp_bins_c), len(rp_bins_c))) 
	boost_contribution = np.zeros((len(rp_bins_c), len(rp_bins_c)))
	shape_contribution = np.zeros((len(rp_bins_c), len(rp_bins_c)))
	# Calculate the statistical error covariance
	for i in range(0,len(np.diag(rp_bins_c))):	 
		gammaIA_stat_cov[i,i] = g_IA_fid[i]**2 * ((cz_a_fid**2 *DeltaCov_a[i] + cz_b_fid**2 *DeltaCov_b[i]) / (cz_a_fid * DeltaSig_est_a[i] - cz_b_fid * DeltaSig_est_b[i])**2 + (cz_a_fid**2 * Sig_IA_a_fid[i]**2 * sigBa[i]**2 + cz_b_fid**2 * Sig_IA_b_fid[i]**2 * sigBb[i]**2) / (cz_a_fid * Sig_IA_a_fid[i] * (Boost_a[i] - 1. + F_a_fid[i]) - cz_b_fid * Sig_IA_b_fid[i] * (Boost_b[i]-1.+F_b_fid[i]))**2)
		
		#boost_contribution[i,i] = g_IA_fid[i]**2 * ((cz_a**2 * Sig_IA_a[i]**2 * sigBa[i]**2 + cz_b**2 * Sig_IA_b[i]**2 * sigBb[i]**2) / (cz_a * Sig_IA_a[i] * (Boost_a[i] - 1. + F_a[i]) - cz_b * Sig_IA_b[i] * (Boost_b[i]-1.+F_b[i]))**2)
		
		#shape_contribution[i,i] = g_IA_fid[i]**2 * ((cz_a**2 *DeltaCov_a[i] + cz_b**2 *DeltaCov_b[i]) / (cz_a * DeltaSig_est_a[i] - cz_b * DeltaSig_est_b[i])**2 )	
		
	# Save the fractional stat error
	save_variance = np.column_stack((rp_bins_c, np.sqrt(np.diag(gammaIA_stat_cov)) / g_IA_fid))
	np.savetxt('./txtfiles/frac_StatError_Blazek_LRG-shapes_7bins.txt', save_variance)
	
	############# Get systematic errors ############
	
	# gamma_IA in the case of systematic error from dNdz
	gamma_from_dNdz = gamma_fid_from_quants(rp_bins_c, Boost_a, Boost_b, F_a_dN, F_b_dN, Sig_IA_a_dN, Sig_IA_b_dN, cz_a_dN, cz_b_dN, DeltaSig_est_a, DeltaSig_est_b)
	
	# gamma_IA in the case of systematic error from pz
	gamma_from_pz = gamma_fid_from_quants(rp_bins_c, Boost_a, Boost_b, F_a_pz, F_b_pz, Sig_IA_a_pz, Sig_IA_b_pz, cz_a_pz, cz_b_pz, DeltaSig_est_a, DeltaSig_est_b)
	
	# gamma_IA in the case of systematic error from the Boost
	gamma_from_boost = gamma_fid_from_quants(rp_bins_c, Boost_a * (1.03), Boost_b * (1.03), F_a_fid, F_b_fid, Sig_IA_a_fid, Sig_IA_b_fid, cz_a_fid, cz_b_fid, DeltaSig_est_a, DeltaSig_est_b)
	
	"""plt.figure()
	plt.loglog(rp_bins_c,g_IA_fid, 'g+', label='fid')
	plt.hold(True)
	plt.loglog(rp_bins_c, gamma_from_dNdz, 'm+', label='dNdz sys')
	plt.hold(True)
	plt.loglog(rp_bins_c, gamma_from_pz, 'b+', label='pz sys')
	plt.hold(True)
	plt.loglog(rp_bins_c, gamma_from_boost, 'k+', label='boost sys')
	plt.legend()
	plt.xlabel('$r_p$')
	plt.ylabel('$\sigma(\gamma_{IA})$')
	plt.xlim(0.04, 20)
	plt.savefig('./plots/gammas_different_systematics_LRG-shapes.pdf')
	plt.close()"""
	
	# Systematic error from dNdz (only diagonal - this is wrong)
	print "SYSTEMATIC ERROR COVARIANCE IS ONLY DIAGONAL RIGHT NOW"
	sigma_dNdz = np.abs(gamma_from_dNdz - g_IA_fid)
	cov_dNdz_sys = sigma_dNdz**2 * np.diag(np.ones(len(rp_bins_c)))
	
	# Systematic error from pz (only diagonal - this is wrong
	sigma_pz = np.abs(gamma_from_pz - g_IA_fid)
	cov_pz_sys = sigma_pz**2 * np.diag(np.ones(len(rp_bins_c)))
	
	# Systematic error from the boost
	sigma_b = np.abs(gamma_from_boost - g_IA_fid)
	cov_B_sys = sigma_b**2 * np.diag(np.ones(len(rp_bins_c)))
	
	gamma_IA_sys_cov = cov_dNdz_sys + cov_pz_sys + cov_B_sys
	
	plt.figure()
	plt.loglog(rp_bins_c, np.sqrt(np.diag(gamma_IA_sys_cov)) / g_IA_fid, 'g+', label='sys total')
	plt.hold(True)
	plt.loglog(rp_bins_c, np.sqrt(np.diag(cov_dNdz_sys)) / g_IA_fid, 'm+', label='sys dNdz')
	plt.hold(True)
	plt.loglog(rp_bins_c, np.sqrt(np.diag(cov_pz_sys)) / g_IA_fid, 'b+', label='sys pz')
	plt.hold(True)
	plt.loglog(rp_bins_c, np.sqrt(np.diag(cov_B_sys)) / g_IA_fid, 'k+', label='sys Boost')
	plt.legend()
	plt.xlabel('$r_p$')
	plt.ylabel('$\sigma(\gamma_{IA})$')
	plt.xlim(0.04, 20)
	plt.savefig('./plots/frac_sys_errors_Blazek_LRG-shapes_7bins.pdf')
	plt.close()
	
	plt.figure()
	plt.loglog(rp_bins_c, np.sqrt(np.diag(gamma_IA_sys_cov)), 'g+', label='sys total')
	plt.hold(True)
	plt.loglog(rp_bins_c, np.sqrt(np.diag(cov_dNdz_sys)), 'm+', label='sys dNdz')
	plt.hold(True)
	plt.loglog(rp_bins_c, np.sqrt(np.diag(cov_pz_sys)), 'b+', label='sys pz')
	plt.hold(True)
	plt.loglog(rp_bins_c, np.sqrt(np.diag(cov_B_sys)), 'k+', label='sys Boost')
	plt.legend()
	plt.xlabel('$r_p$')
	plt.ylabel('$\sigma(\gamma_{IA})$')
	plt.xlim(0.04, 20)
	plt.savefig('./plots/sys_errors_Blazek_LRG-shapes.pdf')
	plt.close()

	# Save the fractional sys error
	save_variance = np.column_stack((rp_bins_c, np.sqrt(np.diag(gamma_IA_sys_cov)) / g_IA_fid))
	np.savetxt('./txtfiles/frac_SysError_Blazek_LRG-shapes_7bins.txt', save_variance)
	
	gammaIA_cov_total = gammaIA_stat_cov + gamma_IA_sys_cov
	
	# Save the fractional total error
	save_variance = np.column_stack((rp_bins_c, np.sqrt(np.diag(gammaIA_cov_total)) / g_IA_fid))
	np.savetxt('./txtfiles/frac_totalError_Blazek_LRG-shapes_7bins.txt', save_variance)
	
	plt.figure()
	plt.loglog(rp_bins_c,np.sqrt(np.diag(gammaIA_cov_total)), 'go', label='Both')
	plt.hold(True)
	plt.loglog(rp_bins_c, np.sqrt(np.diag(gamma_IA_sys_cov)), 'mo', label='sys total')
	plt.hold(True)
	plt.loglog(rp_bins_c, np.sqrt(np.diag(gammaIA_stat_cov)), 'bo', label='stat total')
	plt.legend()
	plt.title('Error, Blazek et al. method (not fractional)')
	plt.xlabel('$r_p$')
	plt.ylabel('$\sigma(\gamma_{IA})$')
	plt.xlim(0.04, 20)
	plt.savefig('./plots/absolute_error_Blazek_LRG-shapes_7bins.pdf')
	plt.close()
	
	"""plt.figure()
	plt.loglog(rp_bins_c,np.sqrt(np.diag(gammaIA_stat_cov)), 'go', label='stat')
	plt.hold(True)
	plt.loglog(rp_bins_c, np.sqrt(np.diag(gamma_IA_sys_cov)), 'mo', label='sys')
	plt.legend()
	plt.xlabel('$r_p$')
	plt.ylabel('$\sigma(\gamma_{IA})$')
	plt.xlim(0.04, 20)
	plt.savefig('./plots/stat_sys_error_Blazek_sigz='+str(pa.sigz_fid)+'_LRG.pdf')
	plt.close()"""

	return (gammaIA_cov_total, g_IA_fid)
	
def gamma_fid_from_quants(rp_bins_c, Boost_a, Boost_b, F_a, F_b, Sig_IA_a, Sig_IA_b, cz_a, cz_b, DeltaSig_est_a, DeltaSig_est_b):
	""" Returns gamma_IA as calculated from the things we put together (not from a fiducial model)."""
	
	gamm_fid = (cz_a * DeltaSig_est_a - cz_b*DeltaSig_est_b) / (((Boost_a -1. +F_a)*cz_a*Sig_IA_a) - ((Boost_b -1. +F_b)*cz_b*Sig_IA_b))
	
	#plt.figure()
	#plt.loglog(rp_bins_c, gamm_fid_check, 'go')
	#plt.xlim(0.05,20)
	#plt.savefig('./plots/checkgam.png')
	#plt.close()
	
	return gamm_fid
	
def gamma_fid(rp):
	""" Returns the fiducial gamma_IA from a combination of terms from different models which are valid at different scales """
	wgg_rp = ws.wgg_full(rp, pa.fsat_LRG, pa.fsky, pa.bd_Bl, pa.bs_Bl, './txtfiles/wgg_1h_Blazek_LRG-shapes_7bins.txt', './txtfiles/wgg_2h_Blazek_LRG_shapes_7bins.txt', './plots/wgg_full_Blazek_LRG-shapes_7bins.pdf')
	wgp_rp = ws.wgp_full(rp, pa.bd_Bl, pa.Ai_Bl, pa.ah_Bl, pa.q11_Bl, pa.q12_Bl, pa.q13_Bl, pa.q21_Bl, pa.q22_Bl, pa.q23_Bl, pa.q31_Bl, pa.q32_Bl, pa.q33_Bl, './txtfiles/wgp_1h_Blazek_LRG-shapes_7bins.txt','./txtfiles/wgp_2h_Blazek_LRG-shapes_7bins.txt', './plots/wgp_full_Blazek_LRG-shapes_7bins.pdf')
	
	gammaIA = wgp_rp / (wgg_rp + 2. * pa.close_cut) 
	
	plt.figure()
	plt.loglog(rp, gammaIA, 'go')
	plt.xlim(0.05,30)
	plt.ylabel('$\gamma_{IA}$')
	plt.xlabel('$r_p$')
	plt.title('Fiducial values of $\gamma_{IA}$')
	plt.savefig('./plots/gammaIA_Blazek_LRG-shapes_7bins.pdf')
	plt.close()
	
	return gammaIA

# FISHER FORECAST COMPUTATIONS #

def par_derivs(params, rp_mid):
	""" Computes the derivatives of gamma_IA wrt the parameters of the IA model we care about constraining.
	Returns a matrix of dimensions (# r_p bins, # parameters)."""

	n_bins = len(rp_mid)

	derivs = np.zeros((n_bins, len(params)))

	# This is for a power law model gamma_IA = A * rp ** beta

	derivs[:, pa.A] = rp_mid**(pa.beta_fid)

	derivs[:, pa.beta] = pa.beta_fid * pa.A_fid * rp_mid**(pa.beta_fid-1)	

	return derivs

def get_Fisher(p_derivs, dat_cov):	
	""" Constructs the Fisher matrix, given a matrix of derivatives wrt parameters in each r_p bin and the data covariance matrix."""

	inv_dat_cov = np.linalg.inv(dat_cov)

	Fish = np.zeros((len(p_derivs[0,:]), len(p_derivs[0, :])))
	for a in range(0,len(p_derivs[0,:])):
		for b in range(0,len(p_derivs[0,:])):
			Fish[a,b] = np.dot(p_derivs[:,a], np.dot(inv_dat_cov, p_derivs[:,b]))
	return Fish

def cut_Fisher(Fish, par_ignore ):
	""" Cuts the Fisher matrix to ignore any parameters we want to ignore. (par_ignore is a list of parameter names as defed in input file."""

	if (par_ignore!=None):
		Fish_cut = np.delete(np.delete(Fish, par_ignore, 0), par_ignore, 1)
	else:	
		Fish_cut = Fish	

	return Fish_cut

def get_par_Cov(Fish_, par_marg):
	""" Takes a Fisher matrix and returns a parameter covariance matrix, cutting out (after inversion) any parameters which will be marginalized over. Par_marg should either be None, if no parameters to be marginalised, or a list of parameters to be marginalised over by name from the input file."""
	
	par_Cov = np.linalg.inv(Fish_)

	if (par_marg != None):
		par_Cov_marg = np.delete(np.delete(par_Cov, par_marg, 0), par_marg, 1)
	else:
		par_Cov_marg = par_Cov

	return par_Cov_marg

def par_const_output(Fish_, par_Cov):
	""" Gunction to output  whatever information about parameters constaints we want given the parameter covariance matrix. The user should modify this function as desired.""" 

	# Put whatever you want to output here

	print "1-sigma constraint on A=", np.sqrt(par_Cov[pa.A, pa.A])

	print "1-sigma constraint on beta=", np.sqrt(par_Cov[pa.beta, pa.beta])


	return

# FOR EASE OF OUTPUT #
def plot_nofz(nofz_, z_, file):
	""" Plots the redshift distribution as a function of z (mostly for sanity check)"""

	plt.figure()
        plt.plot(z_, nofz_)
        plt.xlabel('$z$')
        plt.ylabel('$\\frac{dN}{dz}$')
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tick_params(axis='both', which='minor', labelsize=12)
        plt.tight_layout()
        plt.savefig(file)

	return

def plot_quant_vs_rp(quant, rp_cent, file):
	""" Plots any quantity vs the center of redshift bins"""

	plt.figure()
	plt.loglog(rp_cent, quant, 'ko')
	plt.xlabel('$r_p$')
	#plt.xlim(0.05,20)
	#plt.ylim(0.5,200)
	plt.tick_params(axis='both', which='major', labelsize=12)
	plt.tick_params(axis='both', which='minor', labelsize=12)
	plt.tight_layout()
	plt.savefig(file)

	return
	
def plot_variance(cov_1, fidvalues_1, bin_centers):
	""" Takes a covariance matrix, a vector of the fiducial values of the object in question, and the edges of the projected radial bins, and makes a plot showing the fiducial values and 1-sigma error bars from the diagonal of the covariance matrix. Outputs this plot to location 'filename'."""

	fig_sub=plt.subplot(111)
	plt.rc('font', family='serif', size=20)
	#fig_sub=fig.add_subplot(111) #, aspect='equal')
	fig_sub.errorbar(bin_centers,fidvalues_1, yerr = np.sqrt(np.diag(cov_1)), fmt='o')
	fig_sub.set_xscale("log")
	fig_sub.set_yscale("log") #, nonposy='clip')
	fig_sub.set_xlabel('$r_p$')
	fig_sub.set_ylabel('$\gamma_{IA}$')
	fig_sub.tick_params(axis='both', which='major', labelsize=12)
	fig_sub.tick_params(axis='both', which='minor', labelsize=12)
	fig_sub.set_title('Blazek et al. 2012 method')
	plt.tight_layout()
	plt.savefig('./plots/stat+sys_log_BlazekMethod_LRG-shapes_7bins.pdf')
	plt.close()
	
	fig_sub=plt.subplot(111)
	plt.rc('font', family='serif', size=20)
	#fig_sub=fig.add_subplot(111) #, aspect='equal')
	fig_sub.errorbar(bin_centers,fidvalues_1, yerr = np.sqrt(np.diag(cov_1)), fmt='o')
	fig_sub.set_xscale("log")
	#fig_sub.set_yscale("log") #, nonposy='clip')
	fig_sub.set_xlabel('$r_p$')
	fig_sub.set_ylabel('$\gamma_{IA}$')
	fig_sub.set_ylim(-0.015, 0.015)
	fig_sub.set_xlim(0.05, 20)
	fig_sub.set_title('Blazek et al. 2012 method')
	fig_sub.tick_params(axis='both', which='major', labelsize=12)
	fig_sub.tick_params(axis='both', which='minor', labelsize=12)
	plt.tight_layout()
	plt.savefig('./plots/stat+sys_notloBlazekMethod_LRG-shapes_7bins.pdf')
	plt.close()
	
	"""plt.figure()
	plt.loglog(bin_centers,np.sqrt(np.diag(cov_1)), 'go')
	plt.xlim(0.04, 20)
	plt.ylabel('$\sigma(\gamma_{IA}$')
	plt.xlabel('$r_p$')
	plt.savefig('./plots/stat+sys_alone_Blazek_sigz='+str(pa.sigz_fid)+'_LRG-shapes.pdf')
	plt.close()"""

	return  


######## MAIN CALLS ##########

# Set up projected bins
rp_bins 	= 	setup.setup_rp_bins(pa.rp_min, pa.rp_max, pa.N_bins)
rp_cent		=	setup.rp_bins_mid(rp_bins)

# Set up a function to get z as a function of comoving distance
z_of_com = setup.z_interpof_com()

# Get the redshift corresponding to the maximum separation from the effective lens redshift at which we assume IA may be present
# pa.close_cut is the separation in Mpc/h.
(z_close_high, z_close_low)	= 	setup.get_z_close(pa.zeff, pa.close_cut)

#gammaIA = gamma_fid(rp_cent)

# Get the statistical error on gammaIA
(Cov_gIA, fid_gIA) = get_gammaIA_cov(rp_bins, rp_cent)

# Get the fiducial value of gamma_IA in each projected radial bin
#fid_gIA		=	gamma_fid(rp_cent)

# Output a plot showing the 1-sigma error bars on gamma_IA in projected radial bins 
plot_variance(Cov_gIA, fid_gIA, rp_cent)

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
