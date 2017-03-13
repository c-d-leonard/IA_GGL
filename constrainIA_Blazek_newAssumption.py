# This is a script which predicts constraints on intrinsic alignments, using an updated version of the method of Blazek et al 2012 in which it is not assumed that only excess galaxies contribute to IA (rather it is assumed that source galaxies which are close to the lens along the line-of-sight can contribute.)

import numpy as np
import IA_params_Fisher as pa
import scipy
import scipy.integrate
import matplotlib.pyplot as plt
import scipy.interpolate
import subprocess
import shutil


########## FUNCTIONS ##########

# SET UP PROJECTED RADIUS, DISTANCE, AND REDSHIFT #

def setup_rp_bins(rmin, rmax, nbins):
	""" Sets up the bins of projected radius (in units Mpc/h) """

	bins = scipy.logspace(np.log10(rmin), np.log10(rmax), nbins+1)

	return bins

def rp_bins_mid(rp_edges):
	""" Gets the middle of each projected radius bin."""

	logedges=np.log10(rp_edges)
	bin_centers=np.zeros(len(rp_edges)-1)
	for ri in range(0,len(rp_edges)-1):
		bin_centers[ri]    =       10**((logedges[ri+1] - logedges[ri])/2. +logedges[ri])

	return bin_centers

def get_z_close(z_l, cut_MPc_h):
	""" Gets the z above z_l which is the highest z at which we expect IA to be present for that lens. cut_Mpc_h is that separation in Mpc/h."""

	com_l = com(z_l) # Comoving distance to z_l, in Mpc/h

	tot_com_high = com_l + cut_MPc_h
	tot_com_low = com_l - cut_MPc_h

	# Convert tot_com back to a redshift.

	z_cl_high = z_of_com(tot_com_high)
	z_cl_low = z_of_com(tot_com_low)

	return (z_cl_high, z_cl_low)

def com(z_):
	""" Gets the comoving distance in units of Mpc/h at a given redshift, z_ (assuming the cosmology defined in the params file. """

	OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
	def chi_int(z):
	 	return 1. / (pa.H0 * ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )**(0.5))

	if hasattr(z_, "__len__"):
		chi=np.zeros((len(z_)))
		for zi in range(0,len(z_)):
			chi[zi] = scipy.integrate.quad(chi_int,0,z_[zi])[0]
	else:
		chi = scipy.integrate.quad(chi_int, 0, z_)[0]

	return chi

def z_interpof_com():
	""" Returns an interpolating function which can give z as a function of comoving distance. """

	z_vec = scipy.linspace(0., 10., 10000) # This hardcodes that we don't care about anything over z=2100

	com_vec = com(z_vec)

	z_of_com = scipy.interpolate.interp1d(com_vec, z_vec)

	return	z_of_com

def average_in_bins(F_, R_, Rp_):
	""" This function takes a function F_ of projected radius evaluated at projected radial points R_ and outputs the averaged values in bins with edges Rp_""" 
	
	F_binned = np.zeros(len(Rp_)-1)
	for iR in range(0, len(Rp_)-1):
		indlow=next(j[0] for j in enumerate(R_) if j[1]>=(Rp_[iR]))
		indhigh=next(j[0] for j in enumerate(R_) if j[1]>=(Rp_[iR+1]))
		
		F_binned[iR] = scipy.integrate.simps(F_[indlow:indhigh], R_[indlow:indhigh]) / (R_[indhigh] - R_[indlow])
		
	return F_binned
	
############## THINGS TO DO WITH DNDZ ###############

def get_areas(bins, z_eff):
	""" Gets the area of each projected radial bin, in square arcminutes. z_eff = effective lens redshift. """	

	# Areas in units (Mpc/h)^2
	areas_mpch = np.zeros(len(bins)-1)
	for i in range(0, len(bins)-1):
		areas_mpch[i] = np.pi * (bins[i+1]**2 - bins[i]**2) 

	#Comoving distance out to effective lens redshift in Mpc/h
	chi_eff = com(z_eff)

	# Areas in square arcminutes (466560000 / pi = sqAM in a sphere)
	areas_sqAM = areas_mpch * (466560000. / np.pi) / (4 * np.pi * chi_eff**2)

	return areas_sqAM

def get_NofZ_unnormed(a, zs, z_min, z_max, zpts):
	""" Returns the dNdz of the sources as a function of photometric redshift, as well as the z points at which it is evaluated."""

	z = scipy.linspace(z_min+0.0001, z_max, zpts)
	nofz_ = (z / zs)**(a-1) * np.exp( -0.5 * (z / zs)**2)	

	return (z, nofz_)
	
def N_of_zph(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph):
	""" Returns dNdz_ph, the number density in terms of photometric redshift, defined over photo-z range (z_a_def_ph, z_b_def_ph), normalized over the photo-z range (z_a_norm_ph, z_b_norm_ph), normalized over the spec-z range (z_a_norm, z_b_norm), and defined on the spec-z range (z_a_def, z_b_def)"""
	
	(z, dNdZ) = get_NofZ_unnormed(pa.alpha, pa.zs, z_a_def_s, z_b_def_s, pa.zpts)
	(z_norm, dNdZ_norm) = get_NofZ_unnormed(pa.alpha, pa.zs, z_a_norm_s, z_b_norm_s, pa.zpts)
	
	z_ph_vec = scipy.linspace(z_a_def_ph, z_b_def_ph, 5000)
	z_ph_vec_norm = scipy.linspace(z_a_norm_ph, z_b_norm_ph, 5000)
	
	int_dzs = np.zeros(len(z_ph_vec))
	for i in range(0,len(z_ph_vec)):
		int_dzs[i] = scipy.integrate.simps(dNdZ*p_z(z_ph_vec[i], z, pa.sigz), z)
	
	int_dzs_norm = np.zeros(len(z_ph_vec_norm))
	for i in range(0,len(z_ph_vec_norm)):
		int_dzs_norm[i] = scipy.integrate.simps(dNdZ_norm*p_z(z_ph_vec_norm[i], z_norm, pa.sigz), z_norm)
		
	norm = scipy.integrate.simps(int_dzs_norm, z_ph_vec_norm)
	
	return (z_ph_vec, int_dzs / norm)

def get_Ns_z_frac(rp, z_p_l, z_p_h, B_samp):
	""" Gets the fraction of source galaxies between photometric redshifts z_p_l and z_p_h, including excess and random galaxies, in each rp bin."""

	# Integrating Sigma_C_inv^{2} over this alone would give the fraction just for randoms
	(z, dNdz) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, z_p_l, z_p_h, pa.zphmin, pa.zphmax)
	
	# Because in this case the observable is estimated Delta Sigma and the associated weights go like estimated Sigma_C_inv^2, this factor comes in (see notes).
	
	SigCinv = get_SigmaC_inv(z, pa.zeff)
	random = scipy.integrate.simps(dNdz * SigCinv**2, z)
	
	# Now account for excess galaxies. I'm not sure if this is quite right - it depends on the assumption that Ntot = Boost * Nrand.
	boost_samp = get_boost(rp, B_samp) # This will be different depending on if we are looking at sample a or b
	boost_tot = get_boost(rp, pa.boost_tot) # This is always the boost for the total sample so it is fixed.
	    
	frac = boost_samp / boost_tot * random

	return frac

def get_denom_shapenoise(rp_bins_, rp_c_, zeff_, ns_, nl_, zbinmin, zbinmax, A, bsamp):
	""" Gets the denominator of the variance of the estimated Delta-Sigma. This plays the equivalent role to the number of lens/source pairs in each bin of projected radius in the case where the observable is the tangential shear. """
	
	# First get the factor that accounts for the fact that we don't use the whole z range for which we have surface densities:
	(z, dNdz) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, z_p_l, z_p_h, pa.zphmin, pa.zphmax)
	
	SigCinv = get_SigmaC_inv(z, pa.zeff) # Because in the associated weights go like estimated Sigma_C_inv^2
	random = scipy.integrate.simps(dNdz * SigCinv**2, z) # This would be the z term if it was just randoms.
	
	# Now account for excess galaxies. I'm not sure if this is quite right - it depends on the assumption that Ntot = Boost * Nrand.
	boost_samp = get_boost(rp, B_samp) # This will be different depending on if we are looking at sample a or b
	boost_tot = get_boost(rp, pa.boost_tot) # This is always the boost for the total sample so it is fixed.
	zdepterm = boost_samp / boost_tot * random

	# Get the area of each projected bin in square arcminutes
	bin_areas       =       get_areas(rp_bins_, zeff_)

	# This is the denominator.
	N_ls_pbin = nl_ * A * ns_ * bin_areas * zdepterm

	return N_ls_pbin

def p_z(z_ph, z_sp, sigz):
	""" Returns the probability of finding a photometric redshift z_ph given that the true redshift is z_sp. """
	
	# I'm going to use a Gaussian probability distribution here, but you could change that.
	p_z_ = np.exp(-(z_ph - z_sp)**2 / (2.*(sigz*(1.+z_sp))**2)) / (np.sqrt(2.*np.pi)*(sigz*(1.+z_sp)))
	
	return p_z_

################### THEORETICAL VALUES FOR FRACTIONAL ERROR CALCULATION ########################333

def sigma_e(z_s_, s_to_n):
	""" Returns a value for the model for the per-galaxy noise as a function of source redshift"""

	# This is a dummy things for now
	sig_e = 2. / s_to_n * np.ones(len(z_s_))

	return sig_e

def weights(e_rms, z_, z_l_):
	
	""" Returns the inverse variance weights as a function of redshift. """
	
	SigC_t_inv = get_SigmaC_inv(z_, z_l_)
	
	weights = SigC_t_inv**2/(sigma_e(z_, pa.S_to_N)**2 + e_rms**2 * np.ones(len(z_)))
	
	return weights
	
def weights_specz(e_rms, z_s, z_l_):
	""" Returns the inverse variance weights as a function of redshift. """
	
	SigC_t_inv = get_SigmaC_inv(z_s, z_l_)
	
	bsig = get_bSigma(z_s)
	
	# bsig*SigC_t is equal to SigC_t_estimated (in terms of photo z's)
	
	weights = SigC_t_inv**2/(bsig**2*(sigma_e(z_s, pa.S_to_N)**2 + e_rms**2 * np.ones(len(z_s))))
	
	return weights	
	
def weights_times_SigC(e_rms, z_, z_l_):
	""" Returns the inverse variance weights as a function of redshift. """
	
	SigC_t_inv = get_SigmaC_inv(z_, z_l_)
	
	weights = SigC_t_inv/(sigma_e(z_, pa.S_to_N)**2 + e_rms**2 * np.ones(len(z_)))
	
	return weights

def sum_weights(z_l, z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, erms, rp_bins_, rp_bin_c_):
	""" Returns the sum over rand-source pairs of the estimated weights, in each projected radial bin. Pass different z_min_s and z_max_s to get rand-close, rand-far, and all-rand cases."""
	
	(z_ph, dNdz_ph) = N_of_zph(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph)

	sum_ans = [0]*(len(rp_bins_)-1)
	for i in range(0,len(rp_bins_)-1):
		Integrand = dNdz_ph* weights(erms, z_ph, z_l)
		sum_ans[i] = scipy.integrate.simps(Integrand, z_ph)

	return sum_ans
	
def sum_weights_SigC(z_l, z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, erms, rp_bins_, rp_bin_c_):
	""" Returns the sum over rand-source pairs of the estimated weights multiplied by estimated SigC, in each projected radial bin. Pass different z_min_s and z_max_s to get rand-close, rand-far, and all-rand cases."""
	
	(z_ph, dNdz_ph) = N_of_zph(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph)

	sum_ans = [0]*(len(rp_bins_)-1)
	for i in range(0,len(rp_bins_)-1):
		Integrand = dNdz_ph* weights_times_SigC(erms, z_ph, z_l) 
		sum_ans[i] = scipy.integrate.simps(Integrand, z_ph)

	return sum_ans

def get_bSigma(z_s_):
	
	""" Returns the photo-z bias to the estimated critical surface density. In principle this is a model fit from the spectroscopic subsample of data. """

	# This is a dummy return value for now
	print "BSIGMA NEEDS TO BE CALIBRATED"
	b_Sig = 1. * np.ones(len(z_s_))

	return b_Sig

def get_SigmaC_inv(z_s_, z_l_):
	""" Returns the theoretical value of 1/Sigma_c, (Sigma_c = the critcial surface mass density) """

	com_s = com(z_s_) 
	com_l = com(z_l_) 

	# Get scale factors for converting to comoving distances.
	a_l = 1. / (z_l_ + 1.)
	a_s = 1. / (z_s_ + 1.)
	
	# Units are pc^2 / (h Msun), comoving
	Sigma_c_inv = 4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc * ((a_s*com_s - a_l * com_l) * com_l) / (a_l * a_s * com_s)
	
	if hasattr(z_s_, "__len__"):
		for i in range(0,len(z_s_)):
			if(z_s_[i]<=z_l_):
				Sigma_c_inv[i] = 0.
	else:
		if (z_s_<=z_l_):
			Sigam_c_inv = 0.

	return Sigma_c_inv

def get_boost(rp_cents_, propfact):
	""" Returns the boost factor in radial bins. propfact is a tunable parameter giving the proportionality constant by which boost goes like projected correlation function (= value at 1 Mpc/h). """

	Boost = propfact *(rp_cents_)**(-0.8) + np.ones((len(rp_cents_))) # Empirical power law fit to the boost, derived from the fact that the boost goes like projected correlation function.

	return Boost

def get_F(erms, zph_min_samp, zph_max_samp, rp_bins_, rp_bin_c):
	""" Returns F (the weighted fraction of lens-source pairs from the smooth dNdz which are contributing to IA) """

	# Sum over `rand-close'
	numerator = sum_weights(pa.zeff, z_close_low, z_close_high, pa.zsmin, pa.zsmax, zph_min_samp, zph_max_samp, zph_min_samp, zph_max_samp, erms, rp_bins_, rp_bin_c)

	#Sum over all `rand'
	denominator = sum_weights(pa.zeff, pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zph_min_samp, zph_max_samp, zph_min_samp, zph_max_samp, erms, rp_bins_, rp_bin_c)

	F = np.asarray(numerator) / np.asarray(denominator)

	return F

def get_cz(z_l, z_ph_min, z_ph_max, erms, rp_bins_, rp_bins_c):
	""" Returns the value of the photo-z bias parameter c_z"""

	# The denominator (of 1+bz) is just a sum over weights of all random-source pairs
	denominator = sum_weights(z_l, pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, z_ph_min, z_ph_max, z_ph_min, z_ph_max, erms, rp_bins_, rp_bins_c)

	# Numerator of (1+bz): we integrate over spec-z and account explicitly for the bias to Sigma_c
	(z_s, unnormed_nofz) = get_NofZ_unnormed(pa.alpha, pa.zs, z_ph_min, z_ph_max, pa.zpts) # Note we use the ph limits not the spec ones.
	norm = scipy.integrate.simps(unnormed_nofz, z_s)
	normed_nofz = unnormed_nofz / norm

	numerator = scipy.integrate.simps(normed_nofz * weights_specz(erms, z_s, z_l) * get_bSigma(z_s), z_s)

	# cz = 1 / (1+bz)
	cz = np.asarray(denominator) / np.asarray(numerator)

	return cz

def get_Sig_IA(z_l, z_ph_min_samp, z_ph_max_samp, erms, rp_bins_, rp_bin_c_, boost):
	""" Returns the value of <\Sigma_c>_{IA} in radial bins """
	
	# There are four terms here. The two in the denominators are sums over randoms (or sums over lenses that can be written as randoms * boost), and these are already set up to calculate.
	denom_rand_close = sum_weights(z_l, z_close_low, z_close_high, pa.zsmin, pa.zsmax, z_ph_min_samp, z_ph_max_samp, z_ph_min_samp, z_ph_max_samp, erms, rp_bins_, rp_bin_c_)
	denom_rand = sum_weights(z_l, pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, z_ph_min_samp, z_ph_max_samp, z_ph_min_samp, z_ph_max_samp, erms, rp_bins_, rp_bin_c_)
	denom_excess = (boost - 1.) * denom_rand
	
	# The two in the numerator require summing over weights and Sigma_C. 
	
	#For the sum over rand-close in the numerator, this follows directly from the same type of expression as when summing weights:
	num_rand_close = sum_weights_SigC(z_l, z_close_low, z_close_high, pa.zsmin, pa.zsmax, z_ph_min_samp, z_ph_max_samp, z_ph_min_samp, z_ph_max_samp, erms, rp_bins_, rp_bin_c_)
			
	# The other numerator sum is a term which represents the a sum over excess. Product of a sum over randoms and a boost factor. (Not sure if this is exactly right).
	# We take the randoms as being all at the lens, which simplifies the sum. Do it directly here:
	z_ph = scipy.linspace(z_ph_min_samp, z_ph_max_samp, 5000)

	rand_for_excess = scipy.integrate.simps(weights_times_SigC(erms, z_ph, z_l) * p_z(z_ph, z_l, pa.sigz), z_ph)
	norm_rand_for_excess = scipy.integrate.simps(p_z(z_ph, z_l, pa.sigz), z_ph)
	num_excess = (boost-1.) * rand_for_excess / norm_rand_for_excess
	
	# Sigma_C_inv is in units of pc^2 / (h Msol) (comoving), so Sig_IA is in units of h Msol / pc^2 (comoving).
	Sig_IA = (num_excess + num_rand_close) / (denom_excess + denom_rand_close) 

	return Sig_IA  

def get_est_DeltaSig(z_l, z_ph_min, z_ph_max, erms, rp_bins, rp_bins_c, boost, F, cz, SigIA, g_IA_fid):
	""" Returns the value of tilde Delta Sigma in bins"""
	
	# The first term is (1 + b_z) \Delta \Sigma ^{theory}, need theoretical Delta Sigma	
	DS_the = get_DeltaSig_theory(z_l, rp_bins, rp_bins_c)
		
	EstDeltaSig = np.asarray(DS_the) / np.asarray(cz) + (boost-1.+ F) * SigIA * g_IA_fid
	
	plt.figure()
	plt.loglog(rp_bins_c, EstDeltaSig, 'g+', label='1-halo')
	plt.xlim(0.05,20)
	plt.ylim(0.3,200)
	plt.xlabel('$r_p$, Mpc/h')
	plt.ylabel('$\Delta \Sigma$, $h M_\odot / pc^2$')
	plt.legend()
	plt.savefig('./plots/test_EstDeltaSigmatot_new_Feb21_b.png')
	plt.close()

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
	
	rho = rho_NFW(rvec, pa.Mvir, z_l) # In units of Msol h^2 / Mpc^3 
	
	rho_interp = scipy.interpolate.interp1d(rvec, rho)

	rho_2D = np.zeros((len(rpvec), len(Pivec)))
	for ri in range(0, len(rpvec)):
		for pi in range(0, len(Pivec)):
			rho_2D[ri, pi] = rho_interp(np.sqrt(rpvec[ri]**2 + Pivec[pi]**2))
	
	Sigma_1h = np.zeros(len(rpvec))
	for ri in range(0,len(rpvec)):
		# Units Msol h / Mpc^2, comoving distances
		Sigma_1h[ri] = scipy.integrate.simps(rho_2D[ri, :], Pivec)
		
	plt.figure()
	plt.loglog(rpvec, Sigma_1h/ 10.**12, 'm+') # Plot in Msol h / pc^2.
	plt.xlim(0.0003, 8)
	plt.ylim(0.1,10**4)
	plt.savefig('./plots/Sigma_1h.png')
	plt.close()
	
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
	plt.savefig('./plots/test_DeltaSigmatot_Feb28_Singh2014.png')
	plt.close()
	
	# Interpolate and output at r_bins_c:
	ans_interp = scipy.interpolate.interp1d(rpvec, (DeltaSigma_1h + DeltaSigma_HF) / (10**12))
	ans = ans_interp(rp_bins_c)
	
	return ans # outputting as Msol h / pc^2 

############# QUANTITIES TO GET FIDUCIAL GAMMA_IA ##############

def get_pi(q1, q2, q3, z_):
	""" Returns the pi functions requires for the 1 halo term in wg+, at z_ """
	
	pi = q1 * np.exp(q2 * z_**q3)
	
	return pi

def get_P1haloIA(z, k):
	""" Returns the power spectrum required for the wg+ 1 halo term, at z and k_perpendicular ( = k) """
	
	p1 = get_pi(pa.q11, pa.q12, pa.q13, z)
	p2 = get_pi(pa.q21, pa.q22, pa.q23, z)
	p3 = get_pi(pa.q31, pa.q32, pa.q33, z)

	
	P1halo = np.zeros((len(k), len(z)))
	for ki in range(0,len(k)):
		for zi in range(0,len(z)):
			P1halo[ki, zi]  = pa.ah * ( k[ki] / p1[zi] )**2 / ( 1. + ( k[ki] /p2[zi] )** p3[zi])
	
	return P1halo

def window(tag):
	""" Get window function for projected correlations in linear alignment model. Tag = '+g' or 'gg', determines dndz's."""
	
	sigz = pa.sigz_gwin
	z = scipy.linspace(pa.zeff-5.*sigz, pa.zeff+5.*sigz, 50)
	dNdz_1 = 1. / np.sqrt(2. * np.pi) / sigz * np.exp(-(z-pa.zeff)**2 / (2. *sigz**2))
	
	chi = com(z)
	OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
	dzdchi = pa.H0 * ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )**(0.5)
	
	if (tag =='gg'):
		dNdz_2 = dNdz_1
	elif ((tag=='+g') or (tag=='g+')):
		(z, dNdz_2) = get_NofZ_unnormed(pa.alpha, pa.zs, pa.zeff-5.*sigz, pa.zeff+5.*sigz, 50)
		
	norm = scipy.integrate.simps(dNdz_1*dNdz_2 / chi**2 * dzdchi, z)
	
	win = dNdz_1*dNdz_2 / chi**2 * dzdchi / norm
	
	return (z, win )

def get_Pk(z_, tag):
	""" Calls camb and returns the nonlinear power spectrum (from camb) for the current cosmological parameters and at the redshifts of interest. Returns a 2D array in k and z."""
	
	# The redshifts have to be in a certain order to make camb happy:
	z_list = list(z_)
	z_list.reverse()
	z_rev = np.asarray(z_list)
	
	cambfolderpath = '/home/danielle/Documents/CMU/camb'
	if (tag=='gg'):
		param_string='output_root=IA_shapes_gg\nombh2='+str(pa.OmB * (pa.HH0/100.)**2)+'\nomch2='+str(pa.OmC* (pa.HH0/100.)**2)+'\nhubble='+str(pa.HH0)+'\ntransfer_num_redshifts='+str(len(z_))
	elif ((tag=='gp') or (tag=='pg')):
		param_string='output_root=IA_shapes_gp\nombh2='+str(pa.OmB * (pa.HH0/100.)**2)+'\nomch2='+str(pa.OmC* (pa.HH0/100.)**2)+'\nhubble='+str(pa.HH0)+'\ntransfer_num_redshifts='+str(len(z_))
			
	for zi in range(0, len(z_rev)):
		param_string += '\ntransfer_redshift('+str(zi+1)+') = '+str(z_rev[zi])+ '\ntransfer_matterpower('+str(zi+1)+') = matterpower_z='+str(z_rev[zi])+'.dat'	
		
	
	tempfile=open(cambfolderpath+'/params_string.dat','w')
	tempfile.write(param_string)
	tempfile.close()

	paramsnewfilename='/params_IA_fid.ini'
	
	params_new=open(cambfolderpath+paramsnewfilename, 'w')
	shutil.copyfileobj(open(cambfolderpath+'/params_base.dat', 'r'), params_new)
	shutil.copyfileobj(open(cambfolderpath+'/params_string.dat', 'r'), params_new)
	params_new.close()
	
	#Now write the script to run camb for this set of parameters:
	
	temp_camb=open('./runcamb_temp.sh', 'w')
	temp_camb.write('./camb params_IA_fid.ini')
	temp_camb.close()
	run_camb_now=open('./runcambnow.sh', 'w')
	shutil.copyfileobj(open('./runcamb_base.sh', 'r'), run_camb_now)
	shutil.copyfileobj(open('./runcamb_temp.sh', 'r'), run_camb_now)
	run_camb_now.close()
	
	subprocess.call('./fix_permission.sh')
	subprocess.call('./runcambnow.sh')
	
	# Load one to get the length in k:
	if (tag=='gg'):
		(k, P) = np.loadtxt(cambfolderpath+'/IA_shapes_gg_matterpower_z='+str(z_[0])+'.dat',unpack=True)
	elif ((tag=='gp') or (tag=='pg')):
		(k, P) = np.loadtxt(cambfolderpath+'/IA_shapes_gp_matterpower_z='+str(z_[0])+'.dat',unpack=True)
	
	Pofkz = np.zeros((len(z_),len(k))) 
	for zi in range(0,len(z_)):
		if (tag=='gg'):
			k, Pofkz[zi, :] = np.loadtxt(cambfolderpath+'/IA_shapes_gg_matterpower_z='+str(z_[zi])+'.dat', unpack=True)
		elif ((tag=='gp') or (tag=='pg')):
			k, Pofkz[zi, :] = np.loadtxt(cambfolderpath+'/IA_shapes_gp_matterpower_z='+str(z_[zi])+'.dat', unpack=True)
	
	return (k, Pofkz)

def growth(z_):
	""" Returns the growth factor, normalized to 1 at z=0"""
	
	def int_(z):
		OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
		return (1.+z) / ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )**(1.5)
	
	norm = scipy.integrate.quad(int_, 0, 1000.)[0]
	
	ans = np.zeros(len(z_))
	for zi in range(0,len(z_)):
		ans[zi] = scipy.integrate.quad(int_, z_[zi], 1000)[0]
	
	D = ans / norm
	
	return D

def wgp_1halo(rp_c_):
	""" Returns the 1 halo term of wg+(rp) """
	
	(z, w) = window('g+') # Same window function as for the NLA term.
	
	# Set up a k vector to integrate over:
	k = np.logspace(-5., 4., 100000)
	
	# Get the `power spectrum' term
	P1h = get_P1haloIA(z, k)
	
	# First do the integral over z:
	zint = np.zeros(len(k))
	for ki in range(0,len(k)):
		zint[ki] = scipy.integrate.simps(P1h[ki, :] * w , z)
		
	#plot_quant_vs_quant(k, zint, './zint.png')
		
	# Now do the integral in k
	ans = np.zeros(len(rp_c_))
	for rpi in range(0,len(rp_c_)):
		integrand = k * zint * scipy.special.j0(rp_c_[rpi] * k)
		ans[rpi] = scipy.integrate.simps(k * zint * scipy.special.j0(rp_c_[rpi] * k), k)
		
	wgp1h = ans / (2. * np.pi)
		
	return wgp1h

def wgp_2halo(rp_cents_):
	""" Returns wgp from the nonlinear alignment model (2-halo term only). """
	
	# Get the redshift window function
	z_gp, win_gp = window('g+')
	
	# Get the required matter power spectrum from camb 
	(k_gp, P_gp) = get_Pk(z_gp, 'gp')
	
	# Get the growth factor
	D_gp = growth(z_gp)
	
	# First do the integral over z. Don't yet interpolate in k.
	zint_gp = np.zeros(len(k_gp))
	for ki in range(0,len(k_gp)):
		zint_gp[ki] = scipy.integrate.simps(win_gp * P_gp[:, ki] / D_gp, z_gp)
		
	# Define vectors of kp (kperpendicual) and kz. 
	kp_gp = np.logspace(np.log10(k_gp[0]), np.log10(k_gp[-1]/ np.sqrt(2.01)), pa.kpts_wgp)
	kz_gp = np.logspace(np.log10(k_gp[0]), np.log10(k_gp[-1]/ np.sqrt(2.01)), pa.kpts_wgp)
	
	# Interpolate the answers to the z integral in k to get it in terms of kperp and kz
	kinterp_gp = scipy.interpolate.interp1d(k_gp, zint_gp)
	
	# Get the result of the z integral in terms of kperp and kz
	kpkz_gp = np.zeros((len(kp_gp), len(kz_gp)))
	for kpi in range(0,len(kp_gp)):
		for kzi in range(0, len(kz_gp)):
			kpkz_gp[kpi, kzi] = kinterp_gp(np.sqrt(kp_gp[kpi]**2 + kz_gp[kzi]**2))
			
	# g+: integral in kz	
	kz_int_gp = np.zeros(len(kp_gp))
	for kpi in range(0,len(kp_gp)):
		kz_int_gp[kpi] = scipy.integrate.simps(kpkz_gp[kpi,:] * kp_gp[kpi]**3 / ( (kp_gp[kpi]**2 + kz_gp**2)*kz_gp) * np.sin(kz_gp*pa.ProjMax), kz_gp)
			
	# Finally, do the integrals in kperpendicular
	kp_int_gp = np.zeros(len(rp_cents_))
	for rpi in range(0,len(rp_cents_)):
		kp_int_gp[rpi] = scipy.integrate.simps(scipy.special.jv(2, rp_cents_[rpi]* kp_gp) * kz_int_gp, kp_gp)
		
	wgp_NLA = kp_int_gp * pa.Ai * pa.bd * pa.C1rho * (pa.OmC + pa.OmB) / np.pi**2
	
	wgp_stack = np.column_stack((rp_cents_, wgp_NLA))
	np.savetxt('./txtfiles/wgp_2halo_'+str(pa.kpts_wgp)+'pts.txt', wgp_stack)
	
	return wgp_NLA

def Rhalo(M_insol):
	""" Get the radius of a halo in COMOVING Mpc/h given its mass."""
	
	#E_ofz = ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )**(0.5)#the dimensionless part of the hubble parameter
	#E_ofz = ( (pa.OmM)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )**(0.5)
	#rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun * (pa. HH0 / 100.)) # Msol h^3 / Mpc^3, for use with M in Msol.
	rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h
	#rho_crit = 4.126 * 10** 11 * E_ofz**2 # This is rho_crit in units of Msol h^3 / Mpc^3
	#OmM = (pa.OmC+pa.OmB)*(1+z)**3 / ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )
	rho_m = rho_crit * pa.OmM
	
	Rvir = ( 3. * M_insol / (4. * np.pi * rho_m * 180.))**(1./3.)
	print "Rvir=", Rvir # in Mpc/h
	
	return Rvir

def cvir(M_insol, z):
	""" Returns the concentration parameter of the NFW profile, c_{vir}. """
	
	#cvi = pa.c14 / (1. + z) * (M_insol / 10**14)**(-0.11)
	#cvi = 8.4
	#cvi = 5.6
	#cvi = 7.
	cvi = 5. * (M_insol / 10**14)**(-0.1)
	
	return cvi

def rho_s(cvi, Rvi, M_insol):
	""" Returns rho_s, the NFW parameter representing the density at the `scale radius', Rvir / cvir. Units: Mvir units * ( 1 / (Rvir units)**3), usualy Msol * h^3 / Mpc^3 with comoving distances. Sometimes also Msol h^2 / Mpc^3 (when Mvir is in Msol / h). """
	
	rhos = M_insol / (4. * np.pi) * ( cvi / Rvi)**3 * (np.log(1. + cvi) - (cvi / (1. + cvi)))**(-1)
	
	return rhos

def rho_NFW(r_, M_insol, z):
	""" Returns the density for an NFW profile in real space at distance r from the center. Units = units of rhos. (Usually Msol * h^2 / Mpc^3 in comoving distances). r_ MUST be in the same units as Rv; usually Mpc / h."""

	Rv = Rhalo(M_insol)
	cv = cvir(M_insol, z)
	rhos = rho_s(cv, Rv, M_insol)
	
	rho_nfw = rhos  / ( (cv * r_ / Rv) * (1. + cv * r_ / Rv)**2) 
	
	#plt.figure()
	#plt.loglog(r_ *1000, rho_nfw *  (pa.HH0/100.)**3 / (10**6)**3 )
	#plt.ylim(10**(-6), 10**4)
	#plt.xlim(0.01, 2000.)
	#plt.xlabel('$r$, comoving')
	#plt.savefig('./plots/nfw_vandeVen_lowmass.png')
	#plt.close()
	
	#plt.figure()
	#plt.loglog(r_ *1000 / (pa.HH0 /100.), rho_nfw *  (pa.HH0/100.)**3 / (10**6)**3 )
	#plt.ylim(10**(-6), 10**4)
	#plt.xlim(0.03, 2000.)
	#plt.savefig('./plots/nfw_Mandelbaun2006.png')
	#plt.close()
	
	plt.figure()
	plt.loglog(r_ , rho_nfw  )
	plt.ylim(10**(11), 2.*10**18)
	plt.xlim(0.001, 3)
	plt.savefig('./plots/nfw_Singh2014_z=0.28.png')
	plt.close()
	
	return rho_nfw

def wgg_1halo(rp_cents_, z_l):
	""" Returns the 1-halo term of wgg by integrating over rho_NFW"""
	
	# Define the r (unprojected) vector and Pi (LOS distance) vector at which to get the NFW profile
	rvec = np.logspace(-7, 4, 10000)
	rpvec = np.logspace(-7, np.log10(1000./np.sqrt(2.01)), 1000)
	Pivec = np.logspace(-7, np.log10(1000./np.sqrt(2.01)), 1000)
	
	rho = rho_NFW(rvec, pa.Mvir, z_l) # Units Msol h^3 / Mpc^3 or Msol h^2 / Mpc^3.
	rho_interp = scipy.interpolate.interp1d(rvec, rho)
	
	rho_2D = np.zeros((len(rp_cents_), len(Pivec)))
	for ri in range(0, len(rp_cents_)):
		for pi in range(0, len(Pivec)):
			rho_2D[ri, pi] = rho_interp(np.sqrt(rp_cents_[ri]**2 + Pivec[pi]**2))

	#plt.figure()
	#plt.loglog(rvec, rho, 'go')
	#plt.hold(True)
	#plt.loglog(r_check, rho_check, 'm+')
	#plt.xlim(0.0002, 7.)
	#plt.ylim(0.1, 20000.)
	#plt.savefig('./plots/rho_NFW_checkinterp.png')
	#plt.close()		
			
	#Sigma = np.zeros(len(rpvec))
	#for ri in range(0,len(rpvec)):
	#	Sigma[ri] = scipy.integrate.simps(rho_2D[ri, :], Pivec)	
	
	#plt.figure()
	#plt.loglog(rpvec, Sigma / (10**6)**2, 'go')
	#plt.xlim(0.0002, 7.)
	#plt.ylim(0.1, 20000.)
	#plt.savefig('./plots/Sigma_Mandelbaum2006.png')
	#plt.close()
	
			
	# Get rho_m (comoving coordinates)
	#rho_crit = 4.126 * 10** 11 * E_ofz**2 # This is rho_crit in units of Msol h^3 / Mpc^3
	rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun) # Units Msol h^2 / Mpc^3.
	rho_m = pa. OmM * rho_crit 
	
	# Only integrate out to the virial radius
	Rvir = Rhalo(pa.Mvir)
	indvir = next(j[0] for j in enumerate(Pivec) if j[1]>=(Rvir))
	
	wgg_1h = np.zeros(len(rp_cents_))
	for ri in range(0,len(rp_cents_)):
		wgg_1h[ri] = scipy.integrate.simps(rho_2D[ri, :][0:indvir] / rho_m, Pivec[0:indvir])
		
	plt.figure()
	plt.loglog(rp_cents_, wgg_1h, 'go')
	plt.xlim(0.05, 20.)
	plt.ylim(10**(-3), 10**(4))
	plt.ylabel('$w_{gg}$, Mpc/h com')
	plt.savefig('./plots/wgg_Singh2014_outtovir.png')
	plt.close()

	return wgg_1h

def wgg_1halo_Four(rp_cents_):
	""" Gets the 1halo term of wgg via Fourier space, to account for central-satelite pairs and satelite-satelite pairs. """
	
	Rvir = Rhalo(pa.Mvir)
	
	kvec_FT = np.logspace(-7, 4, 5000000)
	rvec_NFW = np.logspace(-7, np.log10(Rvir), 1000000)
	rvec_xi = np.logspace(-4, 4, 1000)
	Pivec = np.logspace(-7, np.log10(1000./np.sqrt(2.01)), 1000)
	rpvec = np.logspace(-7, np.log10(1000./np.sqrt(2.01)), 1000)
	
	Pk = get_Pkgg_1halo(rvec_NFW, kvec_FT) # Gets the 1halo galaxy power spectrum including c-s and s-s terms. Pass rvec because need to get rho_NFW in here.
	
	xi_gg_1h = get_xi_1h(rvec_xi, kvec_FT, Pk) # Function that gets the 1halo galaxy-galaxy correlation function term.
	xi_interp = scipy.interpolate.interp1d(rvec_xi, xi_gg_1h)
	
	xi_2D = np.zeros((len(rp_cents_), len(Pivec)))
	for ri in range(0, len(rp_cents_)):
		for pi in range(0, len(Pivec)):
			xi_2D[ri, pi] = xi_interp(np.sqrt(rp_cents_[ri]**2 + Pivec[pi]**2)) 
	
	# Only integrate out to the virial radius
	Rvir = Rhalo(pa.Mvir)
	indvir = next(j[0] for j in enumerate(Pivec) if j[1]>=(Rvir))
	
	wgg_1h = np.zeros(len(rp_cents_))
	for ri in range(0,len(rp_cents_)):
		wgg_1h[ri] = scipy.integrate.simps(xi_2D[ri, :][0:indvir], Pivec[0:indvir])
		
	plt.figure()
	plt.loglog(rp_cents_, wgg_1h, 'go')
	plt.xlim(0.05, 20.)
	plt.ylim(10**(-3), 10**(4))
	plt.ylabel('$w_{gg}$, Mpc/h com')
	plt.savefig('./plots/wgg_1h_CS-SS.png')
	plt.close()
	
	plt.figure()
	plt.semilogx(rp_cents_, rp_cents_*wgg_1h, 'go')
	plt.xlim(0.1, 200.)
	plt.ylim(0, 300)
	plt.ylabel('$r_p w_{gg}$, Mpc^2/h^2 com')
	plt.xlabel('$r_p$, Mpc/h com')
	plt.savefig('./plots/Rwgg_1h_CS-SS.png')
	plt.close()
	
	return wgg_1h
	
def get_xi_1h(r, kvec, Pofk):
	""" Returns the 1 halo galaxy correlation function including cen-sat and sat-sat terms, from the power spectrum via Fourier transform."""
	
	xi = np.zeros(len(r))
	for ri in range(0,len(r)):
		xi[ri] = scipy.integrate.simps(Pofk * kvec**2 * np.sin(kvec*r[ri]) / (kvec * r[ri]) / 2. / np.pi**2 , kvec)
	
	plt.figure()
	plt.loglog(r, xi, 'go')
	plt.savefig('./plots/xigg_1h_fourier.png')
	plt.close()
	
	return xi
	
def get_Pkgg_1halo(rvec_nfw, kvec_ft):
	""" Returns the 1halo galaxy power spectrum with c-s and s-s terms."""
	
	
	# Get ingredients we need here:
	y = gety(rvec_nfw, pa.Mvir, pa.zeff, kvec_ft) # Mass-averaged Fourier transform of the density profile

	alpha = get_alpha(pa.Mvir) # The number which accounts for deviation from Poisson statistics
	Ncenavg = get_Nc(pa.Mvir) # The mean number of central galaxies in the halo (between 0 and 1).
	Nsatavg = get_Ns(pa.Mvir) # The mean number of satellite galaxies in the halo
	print "alpha=", alpha
	print "Ncen=", Ncenavg
	print "Nsatavg=", Nsatavg

	NcNs = NcenNsat(alpha, Ncenavg, Nsatavg) # The average number of central-satelite pairs in a halo of mass M
	NsNs = NsatNsat(alpha, Nsatavg) # The average number of satelite-satelite pairs in a halo of mass M
	
	print "NcNs=", NcNs
	print "NsNs=", NsNs
	
	#Pkgg = 1. / (2. * np.pi)**3 / (pa.ng**2) * pa.nh * (NcNs * y + NsNs * y **2)
	Pkgg = 1. / (pa.ng**2) * pa.nh * (NcNs * y + NsNs * y **2)


	plt.figure()
	plt.loglog(kvec_ft, Pkgg, 'b+')
	plt.savefig('./plots/Pkgg_1halo.png')
	plt.close()

	return Pkgg
	
def gety(rvec, M, z, kvec):
	""" Fourier transforms the density profile to get the power spectrum. """
	
	# Get the nfw density profile at the correct mass and redshift and at a variety of r
	rho = rho_NFW(rvec, M, z)  # Units Msol h^2 / Mpc^3, comoving.
	print "in gety - got rho"
	
	# Use a downsampled kvec to speed computation
	kvec_gety = np.logspace(np.log10(kvec[0]), np.log10(kvec[-1]), 1000) 
	print "in gety - got downsampled kvec"
	
	u_ = np.zeros(len(kvec_gety))
	for ki in range(0,len(kvec_gety)):
		u_[ki] = 4. * np.pi / M * scipy.integrate.simps( rvec * np.sin(kvec_gety[ki]*rvec)/ kvec_gety[ki] * rho, rvec) # unitless / dimensionless.
		
	print "in gety - got u"
		
	# Interpolate in k and use the higher-sampled k to output
	u_interp = scipy.interpolate.interp1d(kvec_gety, u_)
	u_morepoints = u_interp(kvec) 
		
	plt.figure()
	plt.loglog(kvec, u_morepoints, 'b+')
	#plt.ylim(0.01, 2)
	#plt.xlim(0.002, 100000)
	plt.savefig('./plots/u.png')
	plt.close()
	
	return u_morepoints
	
def get_alpha(M):
	""" Gets the parameter that accounts for the deviation from Poisson statistics. M is in Msol / h. """
	
	if (M<10**11):
		alpha = np.sqrt(np.log(M / 10**11))
	else:
		alpha = 1.
		
	return alpha
	
def get_Nc(M):
	""" Gets the average number of central galaxies per halo. M is in Msol/h, as in pa.Mmin."""
	
	Ncen = 0.5 * (1. + scipy.special.erf((np.log(M) -pa.logMmin) / pa.siglogM))
	
	return Ncen 
	
def get_Ns(M):
	""" Gets the average number of satelite galaxies per halo. M is in Msol/h, as are Mcut and M1."""
	
	Nsat = (M / pa.M1)**(pa.sat_exp) * np.exp(-pa.Mcut / M)

	return Nsat 
	
def NcenNsat(alpha, Ncen, Nsat):
	""" Returns the average number of pairs of central and satelite galaxies per halo of mass M. """
	
	NcNs = alpha**2 * Ncen * Nsat
	
	return NcNs
	
def NsatNsat(alpha, Nsat):
	""" Returns the average number of pairs of satelite galaxies per halo. """
	
	NsNs = alpha**2 * Nsat**2
	
	return NsNs
	
	
def wgg_2halo(rp_cents_):
	""" Returns wgg for the 2-halo term only."""
	
	# Get the redshift window functions
	z_gg, win_gg = window('gg')
	
	# Get the NL DM power spectrum from camb (this is only for the 2-halo term)
	(k_gg, P_gg) = get_Pk(z_gg, 'gg')
	
	# First do the integral over z. Don't yet interpolate in k.
	zint_gg = np.zeros(len(k_gg))
	for ki in range(0,len(k_gg)):
		zint_gg[ki] = scipy.integrate.simps(win_gg * P_gg[:, ki], z_gg)
		
	# Define vectors of kp (kperpendicual) and kz. Must have sufficiently high sampling to get the right answer, especially at large scales.
	kp_gg = np.logspace(np.log10(k_gg[0]), np.log10(k_gg[-1]/ np.sqrt(2.01)), pa.kpts_wgg)
	kz_gg = np.logspace(np.log10(k_gg[0]), np.log10(k_gg[-1]/ np.sqrt(2.01)), pa.kpts_wgg)
	
	# Interpolate in terms of kperp and kz
	kinterp_gg = scipy.interpolate.interp1d(k_gg, zint_gg)
	
	# Get the result of the z integral in terms of kperp and kz
	kpkz_gg = np.zeros((len(kp_gg), len(kz_gg)))
	for kpi in range(0,len(kp_gg)):
		for kzi in range(0, len(kz_gg)):
			kpkz_gg[kpi, kzi] = kinterp_gg(np.sqrt(kp_gg[kpi]**2 + kz_gg[kzi]**2))
			
	# Do the integrals in kz
	kz_int_gg = np.zeros(len(kp_gg))
	for kpi in range(0,len(kp_gg)):
		kz_int_gg[kpi] = scipy.integrate.simps(kpkz_gg[kpi,:] * kp_gg[kpi] / kz_gg * np.sin(kz_gg*pa.ProjMax), kz_gg)
		
	# Do the integral in kperp
	kp_int_gg = np.zeros(len(rp_cents_))
	for rpi in range(0,len(rp_cents_)):
		kp_int_gg[rpi] = scipy.integrate.simps(scipy.special.j0(rp_cents_[rpi]* kp_gg) * kz_int_gg, kp_gg)
		
	wgg_2h = kp_int_gg * pa.bs * pa. bd / np.pi**2
	wgg_stack = np.column_stack((rp_cents_, wgg_2h))
	np.savetxt('./txtfiles/wgg_2halo_'+str(pa.kpts_wgg)+'pts.txt', wgg_stack)
	
	return wgg_2h

def wgg_full(rp_c):
	""" Combine 1 and 2 halo terms of wgg """
	
	#wgg_1h = wgg_1halo(rp_c, pa.zeff)
	print "GETTING 1HALO WGG"
	wgg_1h = wgg_1halo_Four(rp_c)
	#print "DONE WITH 1HALO WGG, GETTING 2HALO"
	#wgg_2h = wgg_2halo(rp_c)
	#print "DONE WITH 2HALO WGG"
	(rp_cen, wgg_2h) = np.loadtxt('./txtfiles/wgg_2halo_'+str(pa.kpts_wgg)+'pts.txt', unpack=True)
	
	wgg_tot = wgg_1h + wgg_2h 
	
	plt.figure()
	plt.semilogx(rp_c, rp_c* wgg_tot, 'go')
	plt.xlim(0.01, 200.)
	plt.ylim(0., 300.)
	plt.savefig('./plots/Rwgg_tot_incSatSat.png')
	plt.close()
	
	plt.figure()
	plt.loglog(rp_c, wgg_tot, 'go')
	plt.xlim(0.01, 200.)
	#plt.ylim(0., 300.)
	plt.savefig('./plots/wgg_tot_incSatSat.png')
	plt.close()
	
	return wgg_tot

def wgp_full(rp_c):
	""" Combine 1 and 2 halo terms of wgg """
	
	print "GETTING WGP_1H"
	wgp_1h = wgp_1halo(rp_c)
	#print "DONE WITH WGP_1H, ON TO WGP 2H"
	#wgp_2h = wgp_2halo(rp_c)
	#print "DONE WITH 2 HALO WGP"
	(rp_cen, wgp_2h) = np.loadtxt('./txtfiles/wgp_2halo_'+str(pa.kpts_wgp)+'pts.txt', unpack=True)
	
	wgp_tot = wgp_1h + wgp_2h 
	
	plt.figure()
	plt.loglog(rp_c, wgp_tot, 'go')
	plt.xlim(0.01, 200.)
	plt.ylim(0.01, 20)
	plt.savefig('./plots/wgp_tot.png')
	plt.close()
	
	return wgp_tot
	
##### ERRORS FOR FRACTIONAL ERROR CALCULATION #####

def shapenoise_cov(e_rms, z_p_l, z_p_h, B_samp, rp_c, rp):
	""" Returns a diagonal covariance matrix in bins of projected radius for a measurement dominated by shape noise. Elements are e_{rms}^2 / (sum_{ls} Sigmac_inv^2) in each bin. """
	
	# First get the factor that accounts for the fact that we don't use the whole source z range for which we have the source surface density.:
	(z, dNdz) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, z_p_l, z_p_h, pa.zphmin, pa.zphmax)
	SigCinv = get_SigmaC_inv(z, pa.zeff) # Because in the associated weights go like estimated Sigma_C_inv^2
	random = scipy.integrate.simps(dNdz * SigCinv**2, z) # This would be the z term if it was just randoms.
	
	# Now account for excess galaxies in this factor. I'm not sure if this is quite right - it depends on the assumption that Ntot = Boost * Nrand.
	boost_samp = get_boost(rp_c, B_samp) # This will be different depending on if we are looking at sample a or b
	boost_tot = get_boost(rp_c, pa.boost_tot) # This is always the boost for the total sample so it is fixed.
	zdepterm = boost_samp / boost_tot * random

	# Get the area of each projected radial bin in square arcminutes
	bin_areas       =       get_areas(rp, pa.zeff)
	
	# Sum over (Sigma_c_inv^2) normalized to the correct number of lens-source pairs for each bin.
	denom = pa.n_l * pa.Area * pa.n_s * bin_areas * zdepterm
	
	cov = np.diag(e_rms**2 / denom)
	return cov
	
def boost_errors(rp_bins_c, filename):
	""" Imports a file with 2 columns, [rp (kpc/h), sigma(boost-1)]. Interpolates and returns the value of the error on the boost at the center of each bin. """
	
	(rp_kpc, boost_error_raw) = np.loadtxt(filename, unpack=True)
	
	# Convert the projected radius to Mpc/h
	rp_Mpc = rp_kpc / 1000.
	
	interpolate_boost_error = scipy.interpolate.interp1d(rp_Mpc, boost_error_raw)
	
	boost_error = interpolate_boost_error(rp_bins_c)
	
	return boost_error

def get_gammaIA_cov_stat(rp_bins, rp_bins_c):
	""" Takes information about the uncertainty on constituent elements of gamma_{IA} and combines them to get the covariance matrix of gamma_{IA} in projected radial bins."""
	""" We are only interested right now in the diagonal elements of the covariance matrix, so we assume it is diagonal. """ 

	# Get the fiducial quantities required for the statistical error variance terms. #	
	# Boosts
	Boost_a = get_boost(rp_cent, pa.boost_close)
	Boost_b = get_boost(rp_cent, pa.boost_far)
	
	# F factors
	F_a = get_F(pa.e_rms_a, pa.zeff, pa.zeff+pa.delta_z, rp_bins, rp_bins_c)
	F_b = get_F(pa.e_rms_b, pa.zeff+pa.delta_z, pa.zphmax, rp_bins, rp_bins_c)
	
	# Sig IA 
	Sig_IA_a = get_Sig_IA(pa.zeff, pa.zeff, pa.zeff+pa.delta_z, pa.e_rms_a, rp_bins, rp_bins_c, Boost_a)
	Sig_IA_b = get_Sig_IA(pa.zeff, pa.zeff+pa.delta_z, pa.zphmax, pa.e_rms_b, rp_bins, rp_bins_c, Boost_b)

	# Photometric biases to estimated Delta Sigmas
	#cz_a = get_cz(pa.zeff, pa.zeff, pa.zeff + pa.delta_z, pa.e_rms_a, rp_bins, rp_bins_c)
	#cz_b = get_cz(pa.zeff, pa.zeff+pa.delta_z, pa.zphmax, pa.e_rms_b, rp_bins, rp_bins_c)        
	# bSigma not yet estimated so setting both to unity for now:
	print "BSIGMA NEEDS TO BE CALIBRATED AND INCLUDED"
	cz_a = np.ones(len(rp_bins_c)) 
	cz_b = np.ones(len(rp_bins_c))
	
	# Fiducial gamma IA:
	g_IA_fid = gamma_fid(rp_bins_c)

	# Estimated Delta Sigmas
	DeltaSig_est_a = get_est_DeltaSig(pa.zeff, pa.zeff, pa.zeff+pa.delta_z, pa.e_rms_a, rp_bins, rp_bins_c, Boost_a, F_a, cz_a, Sig_IA_a, g_IA_fid)
	DeltaSig_est_b = get_est_DeltaSig(pa.zeff, pa.zeff, pa.zeff+pa.delta_z, pa.e_rms_b, rp_bins, rp_bins_c, Boost_b, F_b, cz_b, Sig_IA_b, g_IA_fid)
	
	# These are the shape-noise-dominated diagonal covariance matrices associated with each sample. Units: Msol^2 h / pc^2, comoving.
	DeltaCov_a = shapenoise_cov(pa.e_rms_a, pa.zeff, pa.zeff+pa.delta_z, pa.boost_close, rp_bins_c, rp_bins)
	DeltaCov_b = shapenoise_cov(pa.e_rms_b, pa.zeff+pa.delta_z, pa.zphmax, pa.boost_far, rp_bins_c, rp_bins)
	
	# Get the errors on B-1 for each sample from file
	sigBa = boost_errors(rp_bins_c, pa.sigBF_a)
	sigBb = boost_errors(rp_bins_c, pa.sigBF_b)
	
	gammaIA_stat_cov = np.zeros((len(rp_bins_c), len(rp_bins_c))) 
	boost_contribution = np.zeros((len(rp_bins_c), len(rp_bins_c)))
	shape_contribution = np.zeros((len(rp_bins_c), len(rp_bins_c)))
	# Calculate the statistical error covariance
	for i in range(0,len(np.diag(rp_bins_c))):	 
		gammaIA_stat_cov[i,i] = g_IA_fid[i]**2 * ((cz_a[i]**2 *DeltaCov_a[i,i] + cz_b[i]**2 *DeltaCov_b[i,i]) / (cz_a[i] * DeltaSig_est_a[i] - cz_b[i] * DeltaSig_est_b[i])**2 + (cz_a[i]**2 * Sig_IA_a[i]**2 * sigBa[i]**2 + cz_b[i]**2 * Sig_IA_b[i]**2 * sigBb[i]**2) / (cz_a[i] * Sig_IA_a[i] * (Boost_a[i] - 1. + F_a[i]) - cz_b[i] * Sig_IA_b[i] * (Boost_b[i]-1.+F_b[i]))**2)
		
		boost_contribution[i,i] = g_IA_fid[i]**2 * ((cz_a[i]**2 * Sig_IA_a[i]**2 * sigBa[i]**2 + cz_b[i]**2 * Sig_IA_b[i]**2 * sigBb[i]**2) / (cz_a[i] * Sig_IA_a[i] * (Boost_a[i] - 1. + F_a[i]) - cz_b[i] * Sig_IA_b[i] * (Boost_b[i]-1.+F_b[i]))**2)
		
		shape_contribution[i,i] = g_IA_fid[i]**2 * ((cz_a[i]**2 *DeltaCov_a[i,i] + cz_b[i]**2 *DeltaCov_b[i,i]) / (cz_a[i] * DeltaSig_est_a[i] - cz_b[i] * DeltaSig_est_b[i])**2 )	
	
	plt.figure()
	plt.loglog(rp_bins_c,np.sqrt(np.diag(gammaIA_stat_cov)), 'go', label='Both')
	plt.hold(True)
	plt.loglog(rp_bins_c, np.sqrt(np.diag(boost_contribution)), 'mo', label='Boost')
	plt.hold(True)
	plt.loglog(rp_bins_c, np.sqrt(np.diag(shape_contribution)), 'bo', label='Shape')
	plt.legend()
	plt.xlabel('$r_p$')
	plt.ylabel('$\sigma(\gamma_{IA})$')
	plt.xlim(0.04, 20)
	plt.savefig('./plots/variance_alone_components_Blazek_incSatSat.pdf')
	plt.close()

	return (gammaIA_stat_cov, g_IA_fid)
	
def gamma_fid_check(rp_bins, rp_bins_c):
	""" Returns the fiducial gamma as calculated from the things we put together (not from a fiducial model), just as a check that it looks okay. """
	# Boosts
	Boost_a = get_boost(rp_cent, pa.boost_close)
	Boost_b = get_boost(rp_cent, pa.boost_far)
	
	# F factors
	F_a = get_F(pa.e_rms_a, pa.zeff, pa.zeff+pa.delta_z, rp_bins, rp_bins_c)
	F_b = get_F(pa.e_rms_b, pa.zeff+pa.delta_z, pa.zphmax, rp_bins, rp_bins_c)
	
	# Sig IA 
	Sig_IA_a = get_Sig_IA(pa.zeff, pa.zeff, pa.zeff+pa.delta_z, pa.e_rms_a, rp_bins, rp_bins_c, Boost_a)
	Sig_IA_b = get_Sig_IA(pa.zeff, pa.zeff+pa.delta_z, pa.zphmax, pa.e_rms_b, rp_bins, rp_bins_c, Boost_b)

	# Photometric biases to estimated Delta Sigmas
	#cz_a = get_cz(pa.zeff, pa.zeff, pa.zeff + pa.delta_z, pa.e_rms_a, rp_bins, rp_bins_c)
	#cz_b = get_cz(pa.zeff, pa.zeff+pa.delta_z, pa.zphmax, pa.e_rms_b, rp_bins, rp_bins_c)
	# bSigma not yet estimated so setting both to unity for now:
	cz_a = np.ones(len(rp_bins_c)) 
	cz_b = np.ones(len(rp_bins_c))
	
	# Fiducial gamma IA:
	g_IA_fid = gamma_fid(rp_bins_c)

	# Estimated Delta Sigmas
	DeltaSig_est_a = get_est_DeltaSig(pa.zeff, pa.zeff, pa.zeff+pa.delta_z, pa.e_rms_a, rp_bins, rp_bins_c, Boost_a, F_a, cz_a, Sig_IA_a, g_IA_fid)
	DeltaSig_est_b = get_est_DeltaSig(pa.zeff, pa.zeff, pa.zeff+pa.delta_z, pa.e_rms_b, rp_bins, rp_bins_c, Boost_b, F_b, cz_b, Sig_IA_b, g_IA_fid)
	
	gamm_fid_check = (cz_a * DeltaSig_est_a - cz_b*DeltaSig_est_b) / (((Boost_a -1. +F_a)*cz_a*Sig_IA_a) - ((Boost_b -1. +F_b)*cz_b*Sig_IA_b))
	
	plt.figure()
	plt.loglog(rp_bins_c, gamm_fid_check, 'go')
	plt.xlim(0.05,20)
	plt.savefig('./plots/checkgam.png')
	plt.close()
	
	return
	
def gamma_fid(rp):
	""" Returns the fiducial gamma_IA from a combination of terms from different models which are valid at different scales """
	
	wgg_rp = wgg_full(rp)
	wgp_rp = wgp_full(rp)
	
	gammaIA = wgp_rp / (wgg_rp + 2. * pa.close_cut) 
	
	print "FIDUCIAL VALUES OF GAM_IA SUSPICIOUSLY HIGH AT LOW RP - CHECK"
	
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
	plt.tight_layout()
	plt.savefig('./plots/variance_log_incSatSat.pdf')
	plt.close()
	
	fig_sub=plt.subplot(111)
	plt.rc('font', family='serif', size=20)
	#fig_sub=fig.add_subplot(111) #, aspect='equal')
	fig_sub.errorbar(bin_centers,fidvalues_1, yerr = np.sqrt(np.diag(cov_1)), fmt='o')
	fig_sub.set_xscale("log")
	#fig_sub.set_yscale("log") #, nonposy='clip')
	fig_sub.set_xlabel('$r_p$')
	fig_sub.set_ylabel('$\gamma_{IA}$')
	fig_sub.tick_params(axis='both', which='major', labelsize=12)
	fig_sub.tick_params(axis='both', which='minor', labelsize=12)
	plt.tight_layout()
	plt.savefig('./plots/variance_notlog_incSatSat.pdf')
	plt.close()
	
	plt.figure()
	plt.loglog(bin_centers,np.sqrt(np.diag(cov_1)), 'go')
	plt.xlim(0.04, 20)
	plt.savefig('./plots/variance_alone_Blazek_incSatSat.pdf')
	plt.close()

	return  


######## MAIN CALLS ##########

# Set up projected bins
rp_bins 	= 	setup_rp_bins(pa.rp_min, pa.rp_max, pa.N_bins)
rp_cent		=	rp_bins_mid(rp_bins)

# Set up a function to get z as a function of comoving distance
z_of_com = z_interpof_com()

# Get the redshift corresponding to the maximum separation from the effective lens redshift at which we assume IA may be present
# pa.close_cut is the separation in Mpc/h.
(z_close_high, z_close_low)	= 	get_z_close(pa.zeff, pa.close_cut)

#gammaIA = gamma_fid(rp_cent)

# Get the statistical error on gammaIA
(Cov_stat_gIA, fid_gIA) = get_gammaIA_cov_stat(rp_bins, rp_cent)

# Get the fiducial value of gamma_IA in each projected radial bin
#fid_gIA		=	gamma_fid(rp_cent)

# Output a plot showing the 1-sigma error bars on gamma_IA in projected radial bins 
plot_variance(Cov_stat_gIA, fid_gIA, rp_cent)

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
