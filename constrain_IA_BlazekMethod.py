# This is a script which predicts constraints on intrinsic alignments, using an updated version of the method of Blazek et al 2012 in which it is not assumed that only excess galaxies contribute to IA (rather it is assumed that source galaxies which are close to the lens along the line-of-sight can contribute.)

SURVEY = 'LSST_DESI'
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
	
	z_ph_vec = scipy.linspace(z_a_def_ph, z_b_def_ph, 500)
	z_ph_vec_norm = scipy.linspace(z_a_norm_ph, z_b_norm_ph, 500)
	
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
	
	# TO INCLUDE AN EXTENDED REDSHIFT FOR LENSES, THIS WILL BE A MATRIX
	SigC_t_inv = get_SigmaC_inv(z_, z_l_) 
	
	# THIS WILL BE MODIFIED TO BE AN ARRAY IN ZS, ZL
	weights = SigC_t_inv**2/(sigma_e(z_)**2 + e_rms**2 * np.ones(len(z_)))
	
	return weights # THIS WILL NEED TO RETURN AN ARRAY

#### THESE ARE OLD FUNCTIONS WHICH COMPUTE THE SHAPE-NOISE ONLY COVARIANCE IN REAL SPACE #####
# I'm just keeping these to be able to compare with their output

def shapenoise_cov(e_rms, z_p_l, z_p_h, B_samp, rp_c, rp, dNdzpar, pzpar, dNdztype, pztype):
	""" Returns a diagonal covariance matrix in bins of projected radius for a measurement dominated by shape noise. Elements are 1 / (sum_{ls} w), carefully normalized, in each bin. """
	
	# IF WE WANT TO CONTINUE TO USE THIS FOR A COMPARISON WITH AN EXTENDED LENS DISTRIBUTION, WE NEED TO FIGURE OUT HOW TO GET THESE AREAS - DO WE JUST USE THE MEAN REDSHIFT FOR THIS?
	# Get the area of each projected radial bin in square arcminutes
	bin_areas       =       setup.get_areas(rp, pa.zeff, SURVEY)
	
	#sum_denom =  pa.n_l * pa.Area_l * bin_areas * pa.n_s * np.asarray(sum_weights(pa.zeff, pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, z_p_l, z_p_h, pa.zphmin, pa.zphmax, e_rms, rp, rp_c, dNdzpar, pzpar, dNdztype, pztype))
	
	# IF WE WANT TO CONTINUE TO USE THIS FOR COMPARISON WITH AN EXTENDED LENS DISTRIBUTION, THE ARGUMENTS OF SUM_WEIGHTS WILL NEED TO BE MODIFIED AS INDICATED IN THAT FUNC.
	weighted_frac_sources = np.asarray(sum_weights(pa.zeff, pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, z_p_l, z_p_h, pa.zphmin, pa.zphmax, e_rms, rp, rp_c, dNdzpar, pzpar, dNdztype, pztype))[0] / np.asarray(sum_weights(pa.zeff, pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, e_rms, rp, rp_c, dNdzpar, pzpar, dNdztype, pztype))[0]
	
	# Get weighted SigmaC values
	#(zph, dNdzph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, z_p_l, z_p_h, pa.zphmin, pa.zphmax, dNdzpar, pzpar, dNdztype, pztype)
	#SigC_inv = get_SigmaC_inv(zph, pa.zeff)
	# IF WE WANT TO CONTINUE TO USE THIS FOR COMPARISON WITH AN EXTENDED LENS DISTRIBUTION, THE ARGUMENTS OF SUM_WEIGHTS_SIGC WILL NEED TO BE MODIFIED AS INDICATED IN THAT FUNC.
	SigC_avg = sum_weights_SigC(pa.zeff, pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, z_p_l, z_p_h, pa.zphmin, pa.zphmax, e_rms, rp, rp_c, dNdzpar, pzpar, dNdztype, pztype)[0] / sum_weights(pa.zeff, pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, z_p_l, z_p_h, pa.zphmin, pa.zphmax, e_rms, rp, rp_c, dNdzpar, pzpar, dNdztype, pztype)[0]
	
	cov = e_rms**2 * SigC_avg**2 / ( pa.n_l * pa.Area_l * bin_areas * pa.n_s * weighted_frac_sources)
	
	return cov

################### THEORETICAL VALUES FOR FRACTIONAL ERROR CALCULATION ########################333
		
def weights_times_SigC(e_rms, z_, z_l_):
	""" Returns the inverse variance weights as a function of redshift. """
	
	# TO INCLUDE AN EXTENDED REDSHIFT DISTRIBUTION IN LENSES, THIS WILL BE AN ARRAY IN ZS ZL
	SigC_t_inv = get_SigmaC_inv(z_, z_l_) 
	
	weights = SigC_t_inv/(sigma_e(z_)**2 + e_rms**2 * np.ones(len(z_))) # THIS WILL BE AN ARRAY IN ZS ZL
	
	return weights

# TO INCLUDE AN EXTENDED REDSHIFT THIS FUNCTION MUST BE MODIFIED TO TAKE 'A' OR 'B' OR 'FULL' AS AN ARGUMENT INDICATING THE SAMPLE AND TO INTEGRATE OVER A LENS Z DIST.
# ADDITIONALLY, THERE SHOULD BE AN ARGUMENT WHICH INDICATES 'CLOSE' OR 'ALL' FOR SPEC-Z, SEE NOTES JULY 31 / 26; August 10., SEE NOTES JULY 31 / 26; August 10.
# SEE NOTEBOOK, JULY 26.
def sum_weights(z_l, z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, erms, rp_bins_, rp_bin_c_, dNdz_par, pz_par, dNdztype, pztype):
	""" Returns the sum over rand-source pairs of the estimated weights, in each projected radial bin. Pass different z_min_s and z_max_s to get rand-close, rand-far, and all-rand cases."""
	
	(z_ph, dNdz_ph) = N_of_zph(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, dNdz_par, pz_par, dNdztype, pztype)
	
	norm = scipy.integrate.simps(dNdz_ph, z_ph)
	
	sum_ans = [0]*(len(rp_bins_)-1)
	for i in range(0,len(rp_bins_)-1):
		Integrand = dNdz_ph* weights(erms, z_ph, z_l)
		sum_ans[i] = scipy.integrate.simps(Integrand, z_ph)

	return sum_ans
	
# TO INCLUDE AN EXTENDED REDSHIFT THIS FUNCTION MUST BE MODIFIED TO TAKE 'A' OR 'B' AS AN ARGUMENT INDICATING THE SAMPLE AND TO INTEGRATE OVER A LENS Z DIST.
# SEE NOTEBOOK, JULY 26.	
# ADDITIONALLY, THERE SHOULD BE AN ARGUMENT WHICH INDICATES 'CLOSE' OR 'ALL' FOR WHAT WE ARE SUMMING OVER, SEE NOTES JULY 31 / 26; August 10.
def sum_weights_SigC(z_l, z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, erms, rp_bins_, rp_bin_c_, dNdz_par, pz_par, dNdztype, pztype):
	""" Returns the sum over rand-source pairs of the estimated weights multiplied by estimated SigC, in each projected radial bin. Pass different z_min_s and z_max_s to get rand-close, rand-far, and all-rand cases."""
	
	(z_ph, dNdz_ph) = N_of_zph(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, dNdz_par, pz_par, dNdztype, pztype)

	sum_ans = [0]*(len(rp_bins_)-1)
	for i in range(0,len(rp_bins_)-1):
		Integrand = dNdz_ph* weights_times_SigC(erms, z_ph, z_l) 
		sum_ans[i] = scipy.integrate.simps(Integrand, z_ph)

	return sum_ans

# TO INCORPORATE EXTENDED REDSHIFT DISTRIBUTION FOR LENSES, THIS FUNCTION WILL TAKE A SAMPLE = A OR B LABEL. 
def get_bSigW(z_p_min, z_p_max, e_rms, pzpar, pztype):
	""" Returns an interpolating function for tilde(w)tilde(SigC)SigC^{-1} as a function of source photo-z. Used in computing photo-z bias c_z / 1+b_z."""
	
	# TO INCORPORATE EXTENDED LENS REDSHIFT DISTRIBUTION: GET ZPH, DNDZPH IN HERE RATHER THAN IN THE WRAPPING GET CZ.
	# GET Z_L, DNDZL
	
	# Define a vector of photometric redshifts on the range of source redshifts we care about for this sample:
	z_p_vec = np.logspace(np.log10(z_p_min), np.log10(z_p_max),100)
	
	# Get the angular diameter distances to these photo-z's of sources, and to the lens z.
	Ds_photo = com_of_z(z_p_vec) / (1. + z_p_vec)
	Dl = com_of_z(pa.zeff) / (1. + pa.zeff)
	
	# TO INCORPORATE EXTENDED LENS REDSHIFT DISTRIBUTION, THIS WILL NEED TO BE LOOPED OVER ZL AND ZPH. THE SAMPLING OF ZS AROUND ZPH CAN BE DONE ONCE FOR EVERY ZPH AND THE SAME FOR EVERY ZL, SO ZPH SHOULD BE THE OUTSIDE LOOP. SEE FURTHER NOTES, NOTEBOOK, JULY 27.
	bsw = np.zeros(len(z_p_vec))
	for zi in range(0,len(z_p_vec)):
		
		if (pztype == 'Gaussian'):
			# Draw a set of points from a normal dist with mean zspec and variance pa.sigz^2. These are the spec-z's for this photo-z.
			zsvec = np.random.normal(z_p_vec[zi], pzpar[0], 5000000)
			# Set points below 0 to zero, it doesn't matter because they will always be lower redshift than the lens, just prevents errors in getting Ds:
			for i in range(0, len(zsvec)):
				if (zsvec[i]<0.):
					zsvec[i] = 0.0001
			print "WE SHOULD BE USING SIGZ(1+Z) HERE FOR OUR VARIANCE."
		else:
			print "Photo-z type "+str(pztype)+" is not supported. Exiting."
			exit()
		# Get the components of the terms we care about which depend on the spec-z's.
		Ds_spec = com_of_z(zsvec) / (1. + zsvec)
		Dls = np.zeros(len(zsvec))  # TO INCORPORATE EXTENDED LENS DISTRIBUTION, THIS WILL BE TWO-D IN ZS AND ZL.
		for zsi in range(0,len(zsvec)):
			if (zsvec[zsi]>pa.zeff):
				Dls[zsi] = Ds_spec[zsi] - Dl
			else:
				Dls[zsi] = 0.
		# Find the mean bsigma at this zphoto
		bsw[zi] = (4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc)**2   / (e_rms**2 + sigma_e(z_p_vec[zi])**2) * (1. +pa.zeff)**4 * Dl**2 * (Ds_photo[zi]-Dl) / Ds_photo[zi] * np.mean(Dls / Ds_spec)
		
	# TO INCORPORATE EXTENDED LENS DISTRIBUTION, INCLUDE INTEGRATION OVER ZPH (FOR EACH ZL AS APPROPRIATE) HERE, AND INTEGRATION OVER ZL
	# SEE NOTES, NOTEBOOK, JULY 27 FOR MORE DETAILS.
	
	# Interpolate the mean bsigmas such that we can report at any zspec in the range:
	bsig_interp = scipy.interpolate.interp1d(z_p_vec, bsw)

	return bsig_interp

def get_SigmaC_inv(z_s_, z_l_):
	""" Returns the theoretical value of 1/Sigma_c, (Sigma_c = the critcial surface mass density) """

	# TO INCORPORATE AN EXTENDED REDSHIFT DISTRIBUTION, THIS MUST BE MODIFIED TO RETURN AN ARRAY IN ZS, ZL.

	com_s = com_of_z(z_s_) 
	com_l = com_of_z(z_l_) 

	# Get scale factors for converting between angular-diameter and comoving distances.
	a_l = 1. / (z_l_ + 1.)
	a_s = 1. / (z_s_ + 1.)
	
	D_s = a_s * com_s # Angular diameter source distance.
	D_l = a_l * com_l # Angular diameter lens distance
	D_ls = (D_s - D_l) # CAREFUL HERE FOR INCLUDING AN EXTENDED Z_L DIST - THIS WILL BE A MATRIX IN ZS, ZL NOW.
	
	# Units are pc^2 / (h Msun), comoving
	Sigma_c_inv = 4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc *   D_l * D_ls * (1 + z_l_)**2 / D_s
	
	if hasattr(z_s_, "__len__"):
		for i in range(0,len(z_s_)):
			if(z_s_[i]<=z_l_):
				Sigma_c_inv[i] = 0.
	else:
		if (z_s_<=z_l_):
			Sigam_c_inv = 0.

	return Sigma_c_inv # THIS MUST BE RETURNED AS A MATRIX.

def get_boost(rp_cents_, propfact):
	"""Returns the boost factor in radial bins. propfact is a tunable parameter giving the proportionality constant by which boost goes like projected correlation function (= value at 1 Mpc/h). """

	Boost = propfact *(rp_cents_)**(-0.8) + np.ones((len(rp_cents_))) # Empirical power law fit to the boost, derived from the fact that the boost goes like projected correlation function.

	return Boost

# CHANGE THE ARGUMENTS OF THIS TO PASS IN A LABLE OF SAMPLE A OR B AND NOT THE ZPH_MIN AND ZPH_MAX STUFF, FOR AN EXTENDED LENS REDSHIFT DISTRIBUTION.
def get_F(erms, zph_min_samp, zph_max_samp, rp_bins_, rp_bin_c, dNdz_num_par, pz_num_par, dNdz_denom_par, pz_denom_par, dNdztype, pztype):
	""" Returns F (the weighted fraction of lens-source pairs from the smooth dNdz which are contributing to IA) """

	# Sum over `rand-close'
	# ARGUMENTS OF SUM_WEIGHTS WILL CHANGE HERE TO TAKE A LABEL OF WHICH SAMPLE, FOR EXTENDED REDSHIFT DIST FOR LENSES.
	numerator = sum_weights(pa.zeff, z_close_low, z_close_high, pa.zsmin, pa.zsmax, zph_min_samp, zph_max_samp, zph_min_samp, zph_max_samp, erms, rp_bins_, rp_bin_c, dNdz_num_par, pz_num_par, dNdztype, pztype)

	#Sum over all `rand'
	# ARGUMENTS OF SUM_WEIGHTS WILL CHANGE HERE TO TAKE A LABEL OF WHICH SAMPLE, FOR EXTENDED REDSHIFT DIST FOR LENSES.
	denominator = sum_weights(pa.zeff, pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zph_min_samp, zph_max_samp, zph_min_samp, zph_max_samp, erms, rp_bins_, rp_bin_c,  dNdz_denom_par, pz_denom_par, dNdztype, pztype)

	F = np.asarray(numerator) / np.asarray(denominator)

	return F

# CHANGE THE ARGUMENTS OF THIS TO PASS IN A LABLE OF SAMPLE A OR B AND NOT THE ZPH_MIN AND ZPH_MAX STUFF, FOR AN EXTENDED LENS REDSHIFT DISTRIBUTION.
def get_cz(z_l, z_ph_min, z_ph_max, erms, rp_bins_, rp_bins_c, dNdzpar, pzpar, dNdztype, pztype):
	""" Returns the value of the photo-z bias parameter c_z"""

	# The denominator (of 1+bz) is just a sum over tilde(weights) of all random-source pairs
	# (This outputs a bunch of things that are all the same, one for each rp bin, so we just take the first one.)
	# ARGUMENTS OF SUM_WEIGHTS WILL CHANGE HERE TO TAKE A LABEL OF WHICH SAMPLE, FOR EXTENDED REDSHIFT DIST FOR LENSES.
	denominator = sum_weights(z_l, pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, z_ph_min, z_ph_max, z_ph_min, z_ph_max, erms, rp_bins_, rp_bins_c, dNdzpar, pzpar, dNdztype, pztype)[0] 
	
	# The numerator (of 1+bz) is a sum over tilde(weights) tilde(Sigma_c) Sigma_c^{-1} of all random-source pair.s
	
	# Get dNdzph
	# TO INCORPORATE AN EXTENDED LENS REDSHIFT DIST, INCORPORATE THIS IN GET_BSIGW.
	(zph, dNdzph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, z_ph_min, z_ph_max, z_ph_min, z_ph_max, dNdzpar, pzpar, dNdztype, pztype)
	
	# Now get a function of photo-z which gives the mean of tilde(weight) tilde(Sigma_c) Sigma_c^{-1} as a function of photo-z.
	# TO INCORPORATE AN EXTENDED LENS REDSHIFT DIST, THIS FUNCTION WILL RETURN A NUMBER RATHER THAN AN INTERPOLATING FUNCTION.
	bsigW_interp = get_bSigW(z_ph_min, z_ph_max, erms, pzpar, pztype)
	numerator = scipy.integrate.simps(bsigW_interp(zph) * dNdzph, zph)

	cz = denominator / numerator

	return cz

# FOR INCLUDING AN EXTENDED LENS REDSHIFT DISTRIBUTION: NEED TO INCLUDE A SAMPLE LABEL OF 'A' OR 'B' TO PASS TO SUM WEIGHTS, MAY NOT NEED OTHER Z ARGS.
def get_Sig_IA(z_l, z_ph_min_samp, z_ph_max_samp, erms, rp_bins_, rp_bin_c_, boost, dNdzpar_1, pzpar_1, dNdzpar_2, pzpar_2, dNdztype, pztype):
	""" Returns the value of <\Sigma_c>_{IA} in radial bins. Parameters labeled '2' are for the `rand-close' sums and '1' are for the `excess' sums. """
	
	# There are four terms here. The two in the denominators are sums over randoms (or sums over lenses that can be written as randoms * boost), and these are already set up to calculate.
	# THE ARGUMENTS OF SUM_WEIGHTS WILL CHANGE TO INCLUDE AN EXTENDED LENS REDSHIFT DISTRIBUTION. 
	denom_rand_close = sum_weights(z_l, z_close_low, z_close_high, pa.zsmin, pa.zsmax, z_ph_min_samp, z_ph_max_samp, z_ph_min_samp, z_ph_max_samp, erms, rp_bins_, rp_bin_c_, dNdzpar_2, pzpar_2, dNdztype, pztype)
	# THE ARGUMENTS OF SUM_WEIGHTS WILL CHANGE TO INCLUDE AN EXTENDED LENS REDSHIFT DISTRIBUTION. 
	denom_rand = sum_weights(z_l, pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, z_ph_min_samp, z_ph_max_samp, z_ph_min_samp, z_ph_max_samp, erms, rp_bins_, rp_bin_c_, dNdzpar_1, pzpar_1, dNdztype, pztype)
	denom_excess = (boost - 1.) * denom_rand
	
	# The two in the numerator require summing over weights and Sigma_C. 
	
	#For the sum over rand-close in the numerator, this follows directly from the same type of expression as when summing weights:
	# THE ARGUMENTS OF SUM_WEIGHTS_SIGC WILL CHANGE TO INCLUDE AN EXTENDED LENS REDSHIFT DISTRIBUTION. 
	num_rand_close = sum_weights_SigC(z_l, z_close_low, z_close_high, pa.zsmin, pa.zsmax, z_ph_min_samp, z_ph_max_samp, z_ph_min_samp, z_ph_max_samp, erms, rp_bins_, rp_bin_c_, dNdzpar_2, pzpar_2, dNdztype, pztype)
	
	
	# WHEN INCLUDING AN EXTENDED LENS REDSHIFT DISTRIBUTION THERE ARE A BUNCH OF MODIFICATIONS TO THIS BIT; SEE NOTEBOOK JULY 26.	
	###############################################	
	# The other numerator sum is a term which represents the a sum over excess. We have to get the normalization indirectly so there are a bunch of terms here. See notes.
	# We assume all excess galaxies are at the lens redshift.
	
	# We first compute a sum over excess of weights and Sigma_C with arbitrary normalization:
	z_ph = scipy.linspace(z_ph_min_samp, z_ph_max_samp, 5000)
	exc_wSigC_arbnorm = scipy.integrate.simps(weights_times_SigC(erms, z_ph, z_l) * setup.p_z(z_ph, z_l, pzpar_1, pztype), z_ph)
	
	# We do the same for a sum over excess of just weights with the same arbitrary normalization:
	exc_w_arbnorm = scipy.integrate.simps(weights(erms, z_ph, z_l) * setup.p_z(z_ph, z_l, pzpar_1, pztype), z_ph)
	################################################3
	
	
	# We already have an appropriately normalized sum over excess weights, from above (denom_excess), via the relationship with the boost.
	# Put these components together to get the appropriately normalized sum over excess of weights and SigmaC:
	num_excess = exc_wSigC_arbnorm / exc_w_arbnorm * np.asarray(denom_excess)
	
	# Sigma_C_inv is in units of pc^2 / (h Msol) (comoving), so Sig_IA is in units of h Msol / pc^2 (comoving).
	Sig_IA = (np.asarray(num_excess) + np.asarray(num_rand_close)) / (np.asarray(denom_excess) + np.asarray(denom_rand_close)) 

	return Sig_IA  

# TO INCLUDE AN EXTENDED LENS REDSHIFT DISTRIBUTION: WE ACTUALLY DON'T USE ZL HERE ANYMORE, SO JUST REMOVE THE DEPENDENCE ON Z_L
def get_est_DeltaSig(z_l, rp_bins, rp_bins_c, boost, F, cz, SigIA, g_IA_fid):
	""" Returns the value of tilde Delta Sigma in bins"""
	# The first term is (1 + b_z) \Delta \Sigma ^{theory}, need theoretical Delta Sigma	
	#DS_the = get_DeltaSig_theory(z_l, rp_bins, rp_bins_c)
		
	#EstDeltaSig = np.asarray(DS_the) / cz + (boost-1.+ F) * SigIA * g_IA_fid
	EstDeltaSig = np.asarray(DeltaSigma_theoretical) / cz + (boost-1.+ F) * SigIA * g_IA_fid
	
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

def get_Pkgm_1halo():
	""" Returns (and more usefully saves) the 1halo lens galaxies x dark matter power spectrum, for the calculation of Delta Sigma (theoretical) """
	
	#Define the full k vector over which we will do the Fourier transform to get the correlation function
	logkmin = -6; kpts =40000; logkmax = 5
	kvec_FT = np.logspace(logkmin, logkmax, kpts)
	# Define the downsampled k vector over which we will compute Pk_{gm}^{1h}
	kvec_short = np.logspace(np.log10(kvec_FT[0]), np.log10(kvec_FT[-1]), 40)
	
	Mhalo = np.logspace(7., 16., 30)
	
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = 2.1*10**(-9), n_s=0.96)
	cosmo = ccl.Cosmology(p)
	HMF= ccl.massfunction.massfunc( cosmo, Mhalo / (pa.HH0/100.), 1./ (1. + pa.zeff), odelta=200. )
	
	if (SURVEY=='SDSS'):
		Ncen_lens = ws.get_Ncen_Reid(Mhalo, SURVEY) # We use the LRG model for the lenses from Reid & Spergel 2008
		Nsat_lens = ws.get_Nsat_Reid(Mhalo, SURVEY) 
	elif (SURVEY=='LSST_DESI'):
		Ncen_lens = ws.get_Ncen(Mhalo, 'nonsense', SURVEY)
		Nsat_lens = ws.get_Nsat(Mhalo, 'nonsense', SURVEY)
		
		plt.figure()
		plt.loglog(Mhalo, Ncen_lens, 'go')
		plt.hold(True)
		plt.loglog(Mhalo, Nsat_lens, 'bo')
		plt.hold(True)
		plt.loglog(Mhalo, Ncen_lens + Nsat_lens, 'mo')
		plt.xlim(10**12, 2*10**15)
		plt.ylim(0.1, 10)
		plt.savefig('./plots/Ntot_LSST+DESI.pdf')
		plt.close()
	else:
		print "We don't have support for that survey yet!"
		exit()
		
	# Check total number of galaxies:
	tot_ng = scipy.integrate.simps( ( Ncen_lens + Nsat_lens) * HMF, np.log10(Mhalo / (pa.HH0/100.) ) ) / (pa.HH0 / 100.)**3
	# Because the number density comes out a little different than the actual case, especially for DESI, we are going to use this number to get the right normalization.
	print "tot=", tot_ng

	# Get the fourier space NFW profile equivalent
	y = ws.gety(Mhalo, kvec_short, SURVEY) 
	
	# Get the density of matter in comoving coordinates
	rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h (comoving distances)
	rho_m = pa.OmM * rho_crit # units of Msol h^2 / Mpc^3 (comoving distances)
	
	# Get Pk
	Pkgm = np.zeros(len(kvec_short))
	for ki in range(0,len(kvec_short)):
		Pkgm[ki] = scipy.integrate.simps( HMF * (Mhalo / rho_m) * (Ncen_lens * y[ki, :] + Nsat_lens * y[ki, :]**2), np.log10(Mhalo / (pa.HH0/ 100.))) / (tot_ng) / (pa.HH0 / 100.)**3
		#print "k=", kvec_short[ki], "Pkgg=", Pkgm[ki]
	
	plt.figure()
	plt.loglog(kvec_short, 4* np.pi * kvec_short**3 * Pkgm / (2* np.pi)**3, 'mo')
	plt.ylim(0.1, 100000)
	plt.xlim(0.05, 1000)
	plt.ylabel('$4\pi k^3 P_{gg}^{1h}(k)$, $(Mpc/h)^3$, com')
	plt.xlabel('$k$, h/Mpc, com')
	plt.savefig('./plots/Pkgm_1halo_survey='+SURVEY+'.pdf')
	plt.close()
	
	Pkgm_interp = scipy.interpolate.interp1d(np.log(kvec_short), np.log(Pkgm))
	logPkgm = Pkgm_interp(np.log(kvec_FT))
	Pkgm = np.exp(logPkgm)
	
	Pkgm_save = np.column_stack((kvec_FT, Pkgm))
	np.savetxt('./txtfiles/Pkgm_1h_dndM_survey='+SURVEY+'.txt', Pkgm_save)
	
	plt.figure()
	plt.loglog(kvec_FT, 4* np.pi * kvec_FT**3 * Pkgm / (2* np.pi)**3, 'mo')
	plt.ylim(0.001, 100000)
	plt.xlim(0.01, 10000)
	plt.ylabel('$4\pi k^3 P_{gg}^{1h}(k)$, $(Mpc/h)^3$, com')
	plt.xlabel('$k$, h/Mpc, com')
	plt.savefig('./plots/Pkgm_1halo_longerkvec_survey='+SURVEY+'.pdf')
	plt.close()
	
	return Pkgm

def get_DeltaSig_theory(z_l, rp_bins, rp_bins_c):
	""" Returns the theoretical value of Delta Sigma in bin using projection over the NFW profile and over the 2-pt correlation function at larger scales, rather than using the lensing-related definition."""
	
	###### First get the term from halofit (valid at larger scales) ######
	# Import the correlation function as a function of R and Pi, obtained via getting P(k) from CAMB and then using FFT_log, Anze Slozar version. 
	# Note that since CAMB uses comoving distances, all distances here should be comoving. rpvec and Pivec are in Mpc/h.
	
	# IF WE WANT TO INCLUDE AN EXTENDED LENS REDSHIFT DISTRIBUTION IN FIDUCIAL QUANTITIES:
	# NEED TO IMPORT CORR_2H AT A VECTOR OF LENS REDSHIFTS TO BE INTEGRATED OVER.
	if (pa.survey == 'SDSS'):
		corr_2h = np.loadtxt('./txtfiles/corr2d_forDS_z='+str(z_l)+'.txt')
	elif (pa.survey =='LSST_DESI'):
		corr_2h = np.loadtxt('./txtfiles/corr2d_forDS_z='+str(z_l)+'.txt')
	else:
		print "No support for that survey yet."
		exit()	
	rpvec = np.loadtxt('./txtfiles/corr_rp_z='+str(z_l)+'.txt')
	Pivec = np.loadtxt('./txtfiles/corr_delta_z='+str(z_l)+'.txt') 
	
	# Get rho_m in comoving coordinates (independent of redshift)
	rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h (comoving distances)
	rho_m = pa.OmM * rho_crit # units of Msol h^2 / Mpc^3 (comoving distances)
	
	# Get Sigma(R) for the 2halo term.
	Sigma_HF = np.zeros(len(rpvec))
	for ri in range(0,len(rpvec)):
		# This will have units Msol h / Mpc^2 in comoving distances.
		Sigma_HF[ri] = rho_m * scipy.integrate.simps(corr_2h[:, ri], Pivec) 
		
	#plt.figure()
	#plt.loglog(rpvec, Sigma_HF, 'g+')
	#plt.ylim(0.0, 10**5)
	#plt.xlim(10**(-2), 100)
	#plt.savefig('./plots/SigHF_z='+str(z_l)+'.pdf')
		
	# Now average Sigma_HF(R) over R to get the first averaged term in Delta Sigma
	barSigma_HF = np.zeros(len(rpvec))
	for ri in range(0,len(rpvec)):
		barSigma_HF[ri] = 2. / rpvec[ri]**2 * scipy.integrate.simps(rpvec[0:ri+1]**2*Sigma_HF[0:ri+1], np.log(rpvec[0:ri+1]))
	
	# Units Msol h / Mpc^2 (comoving distances).
	DeltaSigma_HF = pa.bd_Bl*(barSigma_HF - Sigma_HF)
	
	####### Now get the 1 halo term (valid at smaller scales) #######
	
	#rvec = np.logspace(-7, 4, 10000)
	#rho = ws.rho_NFW(rvec, pa.Mvir, z_l, SURVEY) # In units of Msol h^2 / Mpc^3 

	# Get the max R associated to our max M
	Rmax = ws.Rhalo(10**16, SURVEY)
	
	# IF WE WANT TO INCLUDE AN EXTENDED LENS REDSHIFT DISTRIBUTION IN FIDUCIAL QUANTITIES:
	# NEED TO IMPORT CORR_1H AT A VECTOR OF LENS REDSHIFTS TO BE INTEGRATED OVER.
	r_1h, corr_1h = np.loadtxt('./txtfiles/corr_gm_1h_survey='+SURVEY+'.txt', unpack=True)
	
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
	plt.savefig('./plots/test_DeltaSigmatot_dndM_survey='+SURVEY+'.pdf')
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
	
	# IF WE WANT TO INCLUDE AN EXTENDED DISTRIBUTION OF LENS REDSHIFTS IN OUR FIDUCIAL VALUE CALCULATIONS:
	# NEED TO USE PA.BOOST_CLOSE AND PA.BOOST_FAR VALUES OBTAINED FROM A MODIFIED SCRIPT get_boost_amplitudes_zLextended.ipynb ACCORDING TO PLANNED MODS
	# SEE NOTES JULY 22, NOTEBOOK.
	Boost_a = get_boost(rp_cent, pa.boost_close)
	Boost_b = get_boost(rp_cent, pa.boost_far)
	# Run F, cz, and SigIA, this takes a while so we don't always do it.
	if pa.run_quants == True:
	
		################ F's #################
	
		# F factors - first, fiducial
		# TO INCLUDE AN EXTENDED Z DIST FOR LENSES, THE ARGUMENTS OF THIS FUNCTION WILL CHANGE TO PASS A OR B AS THE SAMPLE LABEL, SEE NOTEBOOK JULY 26.
		F_a_fid = get_F(pa.e_rms_Bl_a, pa.zeff, pa.zeff+pa.delta_z, rp_bins, rp_bins_c,  pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
		F_b_fid = get_F(pa.e_rms_Bl_b, pa.zeff+pa.delta_z, pa.zphmax, rp_bins, rp_bins_c,  pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
		print "F_a_fid=", F_a_fid[0], "F_b_fid=", F_b_fid[0]
	
		save_F = np.column_stack((rp_bins_c, F_a_fid, F_b_fid))
		np.savetxt('./txtfiles/F_afid_bfid_atNAM_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'.txt', save_F)
	
		############# Sig_IA's ##############
	
		# Sig IA - first, fiducial
		# TO INCLUDE AN EXTENDED Z DIST FOR LENSES, THE ARGUMENTS OF THIS FUNCTION WILL CHANGE TO PASS A OR B AS THE SAMPLE LABEL, SEE NOTEBOOK JULY 26.
		Sig_IA_a_fid = get_Sig_IA(pa.zeff, pa.zeff, pa.zeff+pa.delta_z, pa.e_rms_Bl_a, rp_bins, rp_bins_c, Boost_a,   pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
		Sig_IA_b_fid = get_Sig_IA(pa.zeff, pa.zeff+pa.delta_z, pa.zphmax, pa.e_rms_Bl_b, rp_bins, rp_bins_c, Boost_b,   pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
		print "Sig_IA_a_fid=", Sig_IA_a_fid
		print "Sig_IA_b_fid=", Sig_IA_b_fid
	
		save_SigIA = np.column_stack((rp_bins_c, Sig_IA_a_fid, Sig_IA_b_fid))
		np.savetxt('./txtfiles/Sig_IA_afid_bfid_atNAM_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'.txt', save_SigIA)
	
		############ c_z's ##############
	
		# Photometric biases to estimated Delta Sigmas, fiducial
		# TO INCLUDE AN EXTENDED Z DIST FOR LENSES, THE ARGUMENTS OF THIS FUNCTION WILL CHANGE TO PASS A OR B AS THE SAMPLE LABEL, SEE NOTEBOOK JULY 26.
		cz_a_fid = get_cz(pa.zeff, pa.zeff, pa.zeff + pa.delta_z, pa.e_rms_Bl_a, rp_bins, rp_bins_c,   pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
		cz_b_fid = get_cz(pa.zeff, pa.zeff+pa.delta_z, pa.zphmax, pa.e_rms_Bl_b, rp_bins, rp_bins_c,   pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
		print "cz_a_fid =", cz_a_fid, "cz_b_fid=", cz_b_fid
		# INCLUDING AN EXTENDED REDSHIFT DISTRIBUTION MAY MAKE THIS CALCULATION OF CZ TOO LONG TO DO ON A LAPTOP - MAY NEED TO DECOUPLE BSIGW AND RUN ON CLUSTER.
		
		save_cz = np.column_stack(([cz_a_fid], [cz_b_fid]))
		np.savetxt('./txtfiles/cz_afid_bfid_atNAM_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'.txt', save_cz)

	
	############ gamma_IA ###########
	# gamma_IA_fiducial, from model
	g_IA_fid = gamma_fid(rp_bins_c)
	
	if pa.run_quants==False :
		# Load stuff if we haven't computed it this time around:
		#(rp_bins_c, F_a_fid, F_b_fid) = np.loadtxt('./txtfiles/F_afid_bfid_atNAM_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'.txt', unpack=True)
		#(rp_bins_c, Sig_IA_a_fid, Sig_IA_b_fid) = np.loadtxt('./txtfiles/Sig_IA_afid_bfid_atNAM_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'.txt', unpack=True)
		#(cz_a_fid, cz_b_fid) = np.loadtxt('./txtfiles/cz_afid_bfid_atNAM_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'.txt', unpack=True)
		(rp_bins_c, F_a_fid, F_b_fid) = np.loadtxt('./txtfiles/F_afid_bfid_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'.txt', unpack=True)
		(rp_bins_c, Sig_IA_a_fid, Sig_IA_b_fid) = np.loadtxt('./txtfiles/Sig_IA_afid_bfid_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'.txt', unpack=True)
		(cz_a_fid, cz_b_fid) = np.loadtxt('./txtfiles/cz_afid_bfid_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'.txt', unpack=True)
	
	# Estimated Delta Sigmas
	#TO INCLUDE AN EXTENDED REDSHIFT DISTRIBUTION FOR LENSES, JUST REMOVE THE DEPENDENCING ON ZEFF HERE (WE DON'T USE IT ANYMORE).
	DeltaSig_est_a = get_est_DeltaSig(pa.zeff, rp_bins, rp_bins_c, Boost_a, F_a_fid, cz_a_fid, Sig_IA_a_fid, g_IA_fid)
	DeltaSig_est_b = get_est_DeltaSig(pa.zeff, rp_bins, rp_bins_c, Boost_b, F_b_fid, cz_b_fid, Sig_IA_b_fid, g_IA_fid)
	
	############ Get statistical error ############
	
	# Get the real-space shape-noise-only covariance matrices for Delta Sigma for each sample if we want to compare against them.
	#DeltaCov_a = shapenoise_cov(pa.e_rms_Bl_a, pa.zeff, pa.zeff+pa.delta_z, Boost_a, rp_bins_c, rp_bins, pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
	#DeltaCov_b = shapenoise_cov(pa.e_rms_Bl_b, pa.zeff+pa.delta_z, pa.zphmax, Boost_b, rp_bins_c, rp_bins, pa.dNdzpar_fid, pa.pzpar_fid,  pa.dNdztype, pa.pztype)
	
	# Import the covariance matrix of Delta Sigma for each sample as calculated from Fourier space in a different script, w / cosmic variance terms
	#DeltaCov_a_import_CV = np.loadtxt('./txtfiles/cov_DelSig_withCosmicVar_'+SURVEY+'_sample=A_rpts2000_lpts100000_SNanalytic_deltaz='+str(pa.delta_z)+'.txt')
	#DeltaCov_b_import_CV = np.loadtxt('./txtfiles/cov_DelSig_withCosmicVar_'+SURVEY+'_sample=B_rpts2000_lpts100000_SNanalytic_deltaz='+str(pa.delta_z)+'.txt')
	DeltaCov_a_import_CV = np.loadtxt('./txtfiles/cov_DelSig_1h2h_'+SURVEY+'_sample=A_rpts2000_lpts100000_SNanalytic_deltaz=0.17.txt')
	DeltaCov_b_import_CV = np.loadtxt('./txtfiles/cov_DelSig_1h2h_'+SURVEY+'_sample=B_rpts2000_lpts100000_SNanalytic_deltaz=0.17.txt')
	
	#DeltaCov_a_import_CV = np.loadtxt('./txtfiles/cov_DelSig_withCosmicVar_'+SURVEY+'_sample=A_rpts2000_lpts100000_SNanalytic.txt')
	#DeltaCov_b_import_CV = np.loadtxt('./txtfiles/cov_DelSig_withCosmicVar_'+SURVEY+'_sample=B_rpts2000_lpts100000_SNanalytic.txt')
	
	#plt.figure()
	#plt.loglog(rp_bins_c, DeltaCov_a, 'mo', label='shape noise: real')
	#plt.hold(True)
	#plt.loglog(rp_bins_c, np.diag(DeltaCov_a_import_CV), 'go', label='with CV: Fourier')
	#plt.xlabel('r_p')
	#plt.ylabel('Variance')
	#plt.legend()
	#plt.savefig('./plots/check_DeltaSigma_var_SNanalytic_'+SURVEY+'_A_1h2h.pdf')
	#plt.close()
	
	#plt.figure()
	#plt.loglog(rp_bins_c, DeltaCov_b, 'mo', label='shape noise: real')
	#plt.hold(True)
	#plt.loglog(rp_bins_c, np.diag(DeltaCov_b_import_CV), 'bo', label='with CV: Fourier')
	#plt.xlabel('r_p')
	#plt.ylabel('Variance')
	#plt.legend()
	#plt.savefig('./plots/check_DeltaSigma_var_SNanalytic_'+SURVEY+'_B_1h2h.pdf')
	#plt.close()

	# Get the systematic error on boost-1. 
	boosterr_sq_a = ((pa.boost_sys - 1.)*(Boost_a-1.))**2
	boosterr_sq_b = ((pa.boost_sys - 1.)*(Boost_b-1.))**2
	
	#gammaIA_tot_cov = np.zeros((len(rp_bins_c), len(rp_bins_c))) 
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
			denom_term_stat_withF =( ( cz_a_fid * (Boost_a[i] -1. + F_a_fid[i]) * Sig_IA_a_fid[i]) -  ( cz_b_fid * (Boost_b[i] -1. + F_b_fid[i]) * Sig_IA_b_fid[i]) ) * ( ( cz_a_fid * (Boost_a[j] -1. + F_a_fid[j]) * Sig_IA_a_fid[j]) -  ( cz_b_fid * (Boost_b[j] -1. + F_b_fid[j]) * Sig_IA_b_fid[j]) )
			denom_term_stat_noF =( ( cz_a_fid * (Boost_a[i] -1.) * Sig_IA_a_fid[i]) -  ( cz_b_fid * (Boost_b[i] -1.) * Sig_IA_b_fid[i]) ) * ( ( cz_a_fid * (Boost_a[j] -1.) * Sig_IA_a_fid[j]) -  ( cz_b_fid * (Boost_b[j] -1.) * Sig_IA_b_fid[j]) )
			gammaIA_stat_cov_withF[i,j] = num_term_stat / denom_term_stat_withF
			gammaIA_stat_cov_noF[i,j] = num_term_stat / denom_term_stat_noF	
			
			if (i==j):
				
				# Systematic, related to redshifts:
				num_term_sysZ = ( cz_a_fid**2 * DeltaSig_est_a[i]**2 * fudgeczA**2 + cz_b_fid**2 * DeltaSig_est_b[i]**2  * fudgeczB**2 ) / (cz_a_fid * DeltaSig_est_a[i] - cz_b_fid * DeltaSig_est_b[i])**2
				denom_term_sysZ = ( ( cz_a_fid * (Boost_a[i] -1. + F_a_fid[i]) * Sig_IA_a_fid[i])**2 * ( fudgeczA**2 + (fudgeFA * F_a_fid[i])**2 / (Boost_a[i] -1. + F_a_fid[i])**2 + fudgeSigA**2 ) + ( cz_b_fid * (Boost_b[i] -1. + F_b_fid[i]) * Sig_IA_b_fid[i])**2 * ( fudgeczB**2 + (fudgeFB * F_b_fid[i])**2 / (Boost_b[i] -1. + F_b_fid[i])**2 + fudgeSigB**2 ) ) / ( ( cz_a_fid * (Boost_a[i] -1. + F_a_fid[i]) * Sig_IA_a_fid[i]) -  ( cz_b_fid * (Boost_b[i] -1. + F_b_fid[i]) * Sig_IA_b_fid[i]) )**2
				gammaIA_sysZ_cov[i,i] = g_IA_fid[i]**2 * (num_term_sysZ + denom_term_sysZ)
				
				# Systematic, related to boost
				gammaIA_sysB_cov_withF[i,j] = g_IA_fid[i]**2 * ( cz_a_fid**2 * Sig_IA_a_fid[i]**2 * boosterr_sq_a[i] + cz_b_fid**2 * Sig_IA_b_fid[i]**2 * boosterr_sq_b[i] ) / ( ( cz_a_fid * (Boost_a[i] -1. + F_a_fid[i]) * Sig_IA_a_fid[i]) -  ( cz_b_fid * (Boost_b[i] -1. + F_b_fid[i]) * Sig_IA_b_fid[i]) )**2
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
	
def gamma_fid_from_quants(rp_bins_c, Boost_a, Boost_b, F_a, F_b, Sig_IA_a, Sig_IA_b, cz_a, cz_b, DeltaSig_est_a, DeltaSig_est_b):
	""" Returns gamma_IA as calculated from the things we put together. This is a cross check. """
	
	gamm_fid = cz_a * DeltaSig_est_a / (((Boost_a -1. +F_a)*cz_a*Sig_IA_a) - ((Boost_b -1. +F_b)*cz_b*Sig_IA_b)) - cz_b * DeltaSig_est_b / (((Boost_a -1. +F_a)*cz_a*Sig_IA_a) - ((Boost_b -1. +F_b)*cz_b*Sig_IA_b))
	
	#plt.figure()
	#plt.loglog(rp_bins_c, gamm_fid_check, 'go')
	#plt.xlim(0.05,20)
	#plt.savefig('./plots/checkgam.png')
	#plt.close()
	
	return gamm_fid
	
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
	plt.savefig('./plots/stat+sys_log_BlazekMethod_LRG-shapes_NsatThresh_'+str((pa.dNdzpar_fid[0]- pa.dNdzpar_sys[i][0]) / pa.dNdzpar_fid[0])+'.pdf')
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
	plt.savefig('./plots/stat+sys_notloBlazekMethod_LRG-shapes_NsatThresh_'+str((pa.dNdzpar_fid[0]- pa.dNdzpar_sys[i][0]) / pa.dNdzpar_fid[0])+'.pdf')
	plt.close()
	
	"""plt.figure()
	plt.loglog(bin_centers,np.sqrt(np.diag(cov_1)), 'go')
	plt.xlim(0.04, 20)
	plt.ylabel('$\sigma(\gamma_{IA}$')
	plt.xlabel('$r_p$')
	plt.savefig('./plots/stat+sys_alone_Blazek_sigz='+str(pa.sigz_fid)+'_LRG-shapes.pdf')
	plt.close()"""

	return  

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

#check_covergence()
#exit()

# Set up projected bins
rp_bins 	= 	setup.setup_rp_bins(pa.rp_min, pa.rp_max, pa.N_bins)
rp_cent		=	setup.rp_bins_mid(rp_bins)

# Set up a function to get z as a function of comoving distance
(z_of_com, com_of_z) = setup.z_interpof_com(SURVEY) # NO DEPENDENCE ON Z_L 

# Get the redshift corresponding to the maximum separation from the effective lens redshift at which we assume IA may be present
# pa.close_cut is the separation in Mpc/h.

# PASS A VECTOR OF Z_LENS POINTS AND GET OUT OF A VECTOR OF Z_CLOSE_HIGH AND Z_CLOSE_LOW CORRESPONDING TO EACH POINT ON THE Z_L DISTRIBUTION.
(z_close_high, z_close_low)	= 	setup.get_z_close(pa.zeff, pa.close_cut, SURVEY)

# IF WE WANT TO INCLUDE AN EXTENDED LENS REDSHIFT DISTRIBUTION IN FIDUCIAL QUANTITIES:
# WITHIN DELTASIGMA_THEORETICAL, WE NEED TO IMPORT CORR_1H AND CORR_2H AT A VECTOR OF LENS REDSHIFTS TO BE INTEGRATED OVER.
DeltaSigma_theoretical = get_DeltaSig_theory(pa.zeff, rp_bins, rp_cent)

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
