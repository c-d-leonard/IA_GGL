# This is a script which forecasts constraints on IA using multiple shape measurement methods.

SURVEY = 'SDSS'  # Set the survey here; this tells everything which parameter file to import.
print "SURVEY=", SURVEY

import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import shared_functions_setup as setup
import shared_functions_wlp_wls as ws

########## GENERIC FUNCTIONS ##########
	
def sum_weights(photoz_sample, specz_cut, color_cut, dNdz_par, pz_par):
	""" Returns the sum over lens-source pairs of the estimated weights."""
	
	# Get the distribution of lenses
	zL = scipy.linspace(pa.zLmin, pa.zLmax, 100)
	dndzl = setup.get_dNdzL(zL, SURVEY)
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
	(zs_norm, dNdzs_norm) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 1000)
	zs_integral_norm = np.zeros(len(zph_norm))
	for zpi in range(0,len(zph_norm)):
		pz = setup.p_z(zph_norm[zpi], zs_norm, pa.pzpar_fid, pa.pztype)
		zs_integral_norm[zpi] = scipy.integrate.simps(pz * dNdzs_norm, zs_norm)
	norm = scipy.integrate.simps(zs_integral_norm, zph_norm)
		
	# Sum in zphoto at each lens redshift value
	sum_in_zph = np.zeros(len(zL))
	# Loop over lens redshift values
	for zi in range(0,len(zL)):
		
		if (color_cut=='all'):
			if (photoz_sample=='close'):
			
				if (specz_cut=='close'):
					(z_ph, dNdz_ph) = N_of_zph_unweighted(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, zminclose[zi], zmaxclose[zi], pa.zphmin, pa.zphmax, dNdz_par, pz_par)
				elif(specz_cut=='nocut'):
					(z_ph, dNdz_ph) = N_of_zph_unweighted(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zminclose[zi], zmaxclose[zi], pa.zphmin, pa.zphmax, dNdz_par, pz_par)
				else:
					print "We do not have support for that spec-z cut. Exiting."
					exit()
			elif (photoz_sample=='full'):
				if (specz_cut=='close'):
					(z_ph, dNdz_ph) = N_of_zph_unweighted(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par)
				elif(specz_cut=='nocut'):
					(z_ph, dNdz_ph) = N_of_zph_unweighted(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par)
				else:
					print "We do not have support for that spec-z cut. Exiting."
					exit()
			else:
				print "We do not have support for that photo-z sample. Exiting."
				exit()
				
			weight = weights(pa.e_rms_mean, z_ph)	
			sum_in_zph[zi] = scipy.integrate.simps(weight * dNdz_ph, z_ph)
			
		elif (color_cut=='red'):
			if(photoz_sample=='close'):
				if (specz_cut=='close'):
					z_ph = np.linspace(zminclose[zi], zmaxclose[zi], 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zminclose[zi], zmaxclose[zi], 500)
				elif(specz_cut=='nocut'):
					z_ph = np.linspace(zminclose[zi], zmaxclose[zi], 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype,pa.zsmin, pa.zsmax, 500)
				else:
					print "We do not have support for that spec-z cut. Exiting."
					exit()
			elif (photoz_sample=='full'):
				if (specz_cut=='close'):
					z_ph = np.linspace(pa.zphmin, pa.zphmax, 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zminclose[zi], zmaxclose[zi], 500)
				elif (specz_cut=='nocut'):
					z_ph = np.linspace(pa.zphmin, pa.zphmax, 500)
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 500)
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
					
			weight = weights(pa.e_rms_mean, z_ph)
			sum_in_zph[zi] = scipy.integrate.simps(weight * dNdz_fred, z_ph)
	
		else:
			print "We do not have support for that color cut, exiting."
			exit()
	
	# Now sum over all the lenses
	sum_ans = scipy.integrate.simps(sum_in_zph * dndzl, zL)
	
	return sum_ans

def get_ns_partial():
	""" Gets the fractional value of ns appropriate for this subsample."""
	
	# To do this, just get the fraction of dNdzph within the sample:
	frac = sum_weights('close', 'nocut', 'all', pa.dNdzpar_fid, pa.pzpar_fid) / sum_weights('full', 'nocut', 'all', pa.dNdzpar_fid, pa.pzpar_fid)
	
	print "frac=", frac
	
	return frac * ns_tot

def get_fred(photoz_samp):
	""" This function returns the zl- and zph- averaged red fraction for the given sample."""
	
	zL = np.linspace(pa.zLmin, pa.zLmax, 200)
	
	dndzl = setup.get_dNdzL(zL, SURVEY)
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
	
	(zs, dNdzs) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, 0.02, pa.zsmax, 500) # 0.02 is to stay within the range of interpolation for e- and k- corrections - this is so far below the photoz range it shouldn't matter much.
	
	fred_of_z = setup.get_fred_ofz(zs, SURVEY)
	
	ans2 = np.zeros(len(zL))
	norm = np.zeros(len(zL))
	for zi in range(0, len(zL)):
		
		if (photoz_samp=='close'):
			
			zph = np.linspace(zminclose[zi], zmaxclose[zi], 500)
		elif (photoz_sampe =='full'):
			zph = np.linspace(pa.zphmin, pa.zphmax, 500)
		else:
			print "That photo-z cut is not supported; exiting."
			exit()
		
		ans1 = np.zeros(len(zph))
		norm1 = np.zeros(len(zph))
		for zpi in range(0,len(zph)):
			pz = setup.p_z(zph[zpi], zs, pa.pzpar_fid, pa.pztype)
			ans1[zpi] = scipy.integrate.simps(pz * dNdzs * fred_of_z, zs)
			norm1[zpi] = scipy.integrate.simps(pz * dNdzs, zs)
		ans2[zi] = scipy.integrate.simps(ans1, zph)
		norm[zi] = scipy.integrate.simps(norm1, zph)
		
	dndzl = setup.get_dNdzL(zL, SURVEY)
	
	fred_avg = scipy.integrate.simps(dndzl * ans2 / norm, zL)
	return fred_avg
	
def gamma_fid(rp):
	""" Returns the fiducial gamma_IA from a combination of terms from different models which are valid at different scales """
	
	wgg_rp = ws.wgg_full(rp, pa.fsky, pa.bd, pa.bs, './txtfiles/wgg_1h_survey='+pa.survey+'.txt', './txtfiles/wgg_2h_survey='+pa.survey+'_kpts='+str(pa.kpts_wgg)+'.txt', './plots/wgg_full_Blazek_survey='+pa.survey+'.pdf', SURVEY)
	wgp_rp = ws.wgp_full(rp, pa.bd, pa.Ai, pa.ah, pa.q11, pa.q12, pa.q13, pa.q21, pa.q22, pa.q23, pa.q31, pa.q32, pa.q33, './txtfiles/wgp_1h_survey='+pa.survey+'.txt','./txtfiles/wgp_2hsurvey='+pa.survey+'.txt', './plots/wgp_full_Blazek_survey='+pa.survey+'.pdf', SURVEY)
	
	# Get the red fraction for the source sample
	f_red = get_fred('close')
	print "red fraction=", f_red
	
	gammaIA = (f_red * wgp_rp) / (wgg_rp + 2. * pa.close_cut) 
	
	plt.figure()
	plt.loglog(rp, gammaIA, 'go')
	plt.xlim(0.05,30)
	plt.ylabel('$\gamma_{IA}$')
	plt.xlabel('$r_p$')
	plt.title('Fiducial values of $\gamma_{IA}$')
	plt.savefig('./plots/gammaIA_shapes_survey='+pa.survey+'.pdf')
	plt.close()
	
	return gammaIA

def N_of_zph_unweighted(z_a_def, z_b_def, z_a_norm, z_b_norm, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, dNdz_par, pz_par):
	""" Returns dNdz_ph, the number density in terms of photometric redshift, defined and normalized over the photo-z range (z_a_norm_ph, z_b_norm_ph), normalized over the spec-z range (z_a_norm, z_b_norm), but defined on the spec-z range (z_a_def, z_b_def)"""
	
	(z, dNdZ) = setup.get_NofZ_unnormed(dNdz_par, pa.dNdztype, z_a_def, z_b_def, 200)
	(z_norm, dNdZ_norm) = setup.get_NofZ_unnormed(dNdz_par, pa.dNdztype, z_a_norm, z_b_norm, 200)
	
	z_ph_vec_def = scipy.linspace(z_a_def_ph, z_b_def_ph, 200)
	z_ph_vec_norm = scipy.linspace(z_a_norm_ph, z_b_norm_ph, 200)
	
	int_dzs = np.zeros(len(z_ph_vec_def))
	for i in range(0,len(z_ph_vec_def)):
		int_dzs[i] = scipy.integrate.simps(dNdZ*setup.p_z(z_ph_vec_def[i], z, pz_par, pa.pztype), z)
		
	int_dzs_norm = np.zeros(len(z_ph_vec_norm))
	for i in range(0,len(z_ph_vec_norm)):
		int_dzs_norm[i] = scipy.integrate.simps(dNdZ_norm*setup.p_z(z_ph_vec_norm[i], z_norm, pz_par, pa.pztype), z_norm)
		
	norm = scipy.integrate.simps(int_dzs_norm, z_ph_vec_norm)
	
	return (z_ph_vec_def, int_dzs / norm)

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
	
#### THESE FUNCTIONS GET SHAPE-NOISE-ONLY COVARIANCE IN REAL SPACE ####
# I'm not using these to get results now because I'm using the Fourier space method that includes Cosmic Variance, just keeping theses function for comparing purposes
 
def shapenoise_cov(e_rms, rp, dNdzpar, pzpar):
	""" Returns a diagonal covariance matrix in bins of projected radius for a measurement dominated by shape noise. Elements are 1 / (sum_{ls} w), carefully normalized, in each bin. """
	
	# Get the area of each projected radial bin in square arcminutes
	bin_areas       =       setup.get_areas(rp, pa.zeff, SURVEY)
	
	# The ns we want here is in gal / sq arcmin, not steradian:
	ns_sq = ns  / (3600.*3282.8)
	
	cov = e_rms**2 / ( pa.n_l * pa.Area_l * bin_areas * ns_sq)
	
	return cov

def get_cov_btw_methods(cov_a_, cov_b_, covperc):
	""" Get the covariance between the methods given their correlation """
	
	cov_btw_methods = covperc * np.sqrt(cov_a_) * np.sqrt(cov_b_)
	
	return cov_btw_methods

def subtract_var(var_1, var_2, covar):
	""" Takes the variance of two non-independent  Gaussian random variables, and their covariance, and returns the variance of their difference."""
	
	var_diff = var_1 + var_2 - 2 * covar

	return var_diff

####### GETTING ERRORS #########

def get_boost(rp_cents_, propfact):
	""" Returns the boost factor in radial bins. propfact is a tunable parameter giving the proportionality constant by which boost goes like projected correlation function (= value at 1 Mpc/h). """

	Boost = propfact *(rp_cents_)**(-0.8) + np.ones((len(rp_cents_)))# Empirical power law fit to the boost, derived from the fact that the boost goes like projected correlation function.

	return Boost

def f_red_rand():
	""" We compute the weighted fraction of non-excess pairs with red sources over total weighted pairs."""
	
	f = sum_weights('close', 'nocut', 'red', pa.dNdzpar_fid, pa.pzpar_fid) / sum_weights('close', 'nocut', 'all', pa.dNdzpar_fid, pa.pzpar_fid)

	return f

def N_corr(boost, dNdzpar, pzpar):
	""" Computes the correction factor which accounts for the fact that some of the galaxies in the photo-z defined source sample are actually higher-z and therefore not expected to be affected by IA. """
	
	sumW_insamp = sum_weights('close', 'close', 'all', dNdzpar, pzpar)
	sumW_intotal = sum_weights('close', 'nocut', 'all', dNdzpar, pzpar)
	
	Corr_fac = 1. - (1. / boost) * ( 1. - (sumW_insamp / sumW_intotal)) # fraction of the galaxies in the source sample which have spec-z in the photo-z range of interest.
	
	return Corr_fac

def N_corr_err(boost, dNdzpar, pzpar):
	""" Gets the error on N_corr from systemtic error on the boost. We have checked the statistical error on the boost is negligible. """
	
	sumW_insamp = sum_weights('close', 'close', 'all', dNdzpar, pzpar)
	sumW_intotal = sum_weights('close', 'nocut', 'all', dNdzpar, pzpar)
	
	boost_err_sys = (pa.boost_sys - 1.) * (boost - 1.)
	
	sig_Ncorr = (boost_err_sys / boost**2) * np.sqrt(1. - (sumW_insamp/ sumW_intotal))

	return sig_Ncorr
	
def get_combined_covariance(Clggterm, corr):
	""" Get the covariance matrix Cov(gammat(r) - gammat'(r), gamma(r') - gammat'(r'))"""
	# Clggterm is the halo-shapenoise terms. corr is the correlation coefficient between \gamma_rms for the methods
	
	chiL = com_of_z(pa.zeff) # NOTE THAT WE ARE NOT GOING TO CHANGE THIS TO INCORPORATE AN EXTENDED LENS DISTRIBUTION, BECAUSE WE HAVE OPTED NOT TO INTEGRATE OF CHIL IN THE ARGUMENTS OF THE BESSEL FUNCTIONS IN THE COVARIANCE EXPRESSIONS AND THAT IS WHERE THIS COMES FROM. 
	
	# This expression works because of the particular way we have output the covariance terms of the Fourier space calculation.
	cov_comb = np.zeros((pa.N_bins, pa.N_bins))
	for i in range(0,pa.N_bins):
		for j in range(0,pa.N_bins):
			if (i==j):
				cov_comb[i,j] = Clggterm[i,j] * (pa.e_rms_a**2 +pa.e_rms_b**2 - 2. * corr * pa.e_rms_a * pa.e_rms_b) + (pa.e_rms_a**2 +pa.e_rms_b**2 - 2. * corr * pa.e_rms_a * pa.e_rms_b) * chiL**2 / (4. * np.pi**2 * pa.fsky * (rp_bins[i+1]**2 - rp_bins[i]**2) * ns * nl)
			else:
				cov_comb[i,j] = Clggterm[i,j] * (pa.e_rms_a**2 +pa.e_rms_b**2 - 2. * corr * pa.e_rms_a * pa.e_rms_b)
	
	return cov_comb

def get_gammaIA_cov(rp_cents_, rp_bins_, gIA_fid, boost, Ncorr_fid, Ncorr_err, covperc, a_con, fudge_Ncorr):
	""" Takes the covariance matrices of the constituent elements of gamma_{IA} and combines them to get the covariance matrix of gamma_{IA} in projected radial bins."""
	
	# Import Clgg-related covariance terms from separate Fourier-space script
	Clggterm_1 = np.loadtxt('./txtfiles/cov_gamt_extl_'+SURVEY+'_method=same_rms_rpts2000_lpts90000_Clggterm_fixns.txt')
	
	# Get the combined covariance Cov(gammat(r) - gammat'(r), gamma(r') - gammat'(r')):
	cov_gam_diff = get_combined_covariance(Clggterm_1, covperc)
	
	# Uncomment this section to check covariance matrix against the version calculated in real space (shape-noise only)
	# This gets the shape-noise from real-space methods for comparison - this matrix is diagonal by definition so it is output only as a vector of the diagonal elements
	"""shear_cov_1 = shapenoise_cov(pa.e_rms_a, rp_bins_, pa.dNdzpar_fid, pa.pzpar_fid) 
	shear_cov_2 = shear_cov_1
	shear_covar = get_cov_btw_methods(shear_cov_1, shear_cov_2, covperc)
	cov_old=np.zeros((pa.N_bins, pa.N_bins))
	for i in range(0, pa.N_bins):
		cov_old[i,i] = subtract_var(shear_cov_1[i], shear_cov_2[i], shear_covar[i]) 
	
	# Compare real-space shape noise only answer with fourier space answer (including CV)
	plt.figure()
	plt.loglog(rp_cents_, np.diag(cov_old), 'mo', label='real, combined')
	plt.hold(True)
	plt.loglog(rp_cents_, np.diag(cov_gam_diff), 'go', label='fourier, combined (with CV)')
	plt.xlabel('r_p')
	plt.ylabel('Variance')
	plt.legend()
	plt.savefig('./plots/check_gammat_extl_'+SURVEY+'_fixns.pdf')
	plt.close()"""

	# Get statistical covariance matrix for (1-a) gamma_IA (with factors of Ncorr)
	cov_mat_stat = np.zeros((pa.N_bins, pa.N_bins))
	for i in range(0,pa.N_bins):
		for j in range(0,pa.N_bins):
			cov_mat_stat[i,j] = cov_gam_diff[i,j] / Ncorr_fid[i] / Ncorr_fid[j]  # f_red needs to be incorporated here. 
	
	# Get the covariance matrix terms due to the two sources of systematic error, redshifts and boost.
	# First get diagonal elements
	cov_mat_sysz_Ncorr = np.diag(np.zeros(pa.N_bins))
	cov_mat_sysB = np.diag(np.zeros(pa.N_bins))
	for i in range(0,pa.N_bins):	
		cov_mat_sysz_Ncorr[i,i] = (1.-a_con)**2 * gIA_fid[i]**2 * ( (fudge_Ncorr)**2 )
		cov_mat_sysB[i,i] = (1.-a_con)**2 * gIA_fid[i]**2 * ( Ncorr_err[i]**2 / Ncorr_fid[i]**2 )
		
	# Get off-diagonal elements by assuming full correlation for both	
	for i in range(0,len((rp_cents_))):	
		for j in range(0,len((rp_cents_))):
			if (i != j):
				cov_mat_sysz_Ncorr[i,j] = np.sqrt(cov_mat_sysz_Ncorr[i,i]) * np.sqrt(cov_mat_sysz_Ncorr[j,j])
				cov_mat_sysB[i,j] = np.sqrt(cov_mat_sysB[i,i]) * np.sqrt(cov_mat_sysB[j,j])
	
	# Get the covariance matrix for the case of stat + sys due to z
	cov_mat_stat_sysz = cov_mat_stat + cov_mat_sysz_Ncorr
	# And the covariance matrix for the case of stat + sys due to boost
	cov_mat_stat_sysB = cov_mat_stat + cov_mat_sysB
	
	"""# Output the per-bin StoN to compare with other methods for one choice of a and rho
	SN_stat_sysB = gIA_fid / (np.sqrt(np.diag(cov_mat_stat_sysB)))
	save_SN = np.column_stack((rp_cents_, SN_stat_sysB))
	np.savetxt('./txtfiles/StoN_sysB_stat_shapes_'+SURVEY+'_a0pt7_rho0pt3.txt', save_SN)"""
	
	# Compute associated signal to noise quantities
	
	# For statistical + sys from z
	Cov_inv_stat_sysz = np.linalg.inv(cov_mat_stat_sysz)
	StoNsq_stat_sysz = np.dot(gIA_fid* (1.-a_con), np.dot(Cov_inv_stat_sysz, gIA_fid * (1-a_con)))
	
	# For statistical only
	Cov_inv_stat = np.linalg.inv(cov_mat_stat)
	StoNsq_stat = np.dot(gIA_fid* (1.-a_con), np.dot(Cov_inv_stat, gIA_fid * (1-a_con)))
	
	# For sysz only
	NtoSsq_sysz = 1./StoNsq_stat_sysz - 1./StoNsq_stat
	StoNsq_sysz = 1. / NtoSsq_sysz

	# For statistical + sys from B
	Cov_inv_stat_sysB = np.linalg.inv(cov_mat_stat_sysB)
	StoNsq_stat_sysB = np.dot(gIA_fid* (1.-a_con), np.dot(Cov_inv_stat_sysB, gIA_fid * (1-a_con)))
	
	# Return the signal to noise quantities we want
	return (StoNsq_stat_sysB, StoNsq_sysz, StoNsq_stat)
	
def check_convergence():
	""" This short function checks the convergence of the Clggterm as calculated in Fourier space (by another script) wrt the number of rp points."""
	
	#rpts_1 = '2000'; rpts_2 = '2500'; 
	#Clggterm_rpts1 = np.loadtxt('./txtfiles/cov_gamt_1h2h_'+SURVEY+'_method=1_rpts'+rpts_1+'_lpts100000_Clggterm.txt')
	#Clggterm_rpts2 = np.loadtxt('./txtfiles/cov_gamt_1h2h_'+SURVEY+'_method=1_rpts'+rpts_2+'_lpts100000_Clggterm.txt')
	
	#fracdiff = np.abs(Clggterm_rpts2 - Clggterm_rpts1) / np.abs(Clggterm_rpts1)*100
	#print "max percentage difference=", np.amax(fracdiff), "%"
	
	#exit()
	
	lpts_1 = '100000'; lpts_2 = '500000'; 
	Clggterm_lpts1 = np.loadtxt('./txtfiles/cov_gamt_1h2h_'+SURVEY+'_method=1_rpts2000_lpts'+lpts_1+'_Clggterm.txt')
	Clggterm_lpts2 = np.loadtxt('./txtfiles/cov_gamt_1h2h_'+SURVEY+'_method=1_rpts2000_lpts'+lpts_2+'_Clggterm.txt')
	
	fracdiff = np.abs(Clggterm_lpts2 - Clggterm_lpts1) / np.abs(Clggterm_lpts1)*100
	print "max percentage difference=", np.amax(fracdiff), "%"
	
	exit()
	
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
	
# Get the surface densities we need in terms of steradians, and the weighted fractional one of the z-bin.
nl			=	pa.n_l * 3282.8 # n_l is in # / square degree, numerical factor converts to # / steradian	
ns_tot			=	pa.n_s * 3600.*3282.8 # n_s is in # / sqamin, numerical factor converts to / steraidan
	
# Uncomment the following lines if you want to check the covariance matrix has converged (you also need to edit the function for the specific case in question).
# check_convergence()
# exit()

# Set up projected radial bins
rp_bins 	= 	setup.setup_rp_bins(pa.rp_min, pa.rp_max, pa.N_bins) # Edges
rp_cents	=	setup.rp_bins_mid(rp_bins) # Centers

# Set up to get z as a function of comoving distance and vice-versa
z_of_com, com_of_z 	= 	setup.z_interpof_com(SURVEY)

ns = get_ns_partial()

# Get the fiducial value of gamma_IA in each projected radial bin 
fid_gIA		=	gamma_fid(rp_cents)

# Get the values of Ncorr, the boost, and the error on Ncorr - these take some time and don't depend on the stuff we loop over so just do it once.
boost_fid = get_boost(rp_cents, pa.boost_assoc)
Ncorr_fid = N_corr(boost_fid, pa.dNdzpar_fid, pa.pzpar_fid)

Ncorr_err = N_corr_err(boost_fid, pa.dNdzpar_fid, pa.pzpar_fid)

# Get the desired S-to_N ratios for the case of stat error plus sys error from the boost (sys error due to z subdominant), for the case of sys error due to z, and for the case of statistical only.
StoNsquared_stat_sysB = np.zeros((len(pa.a_con), len(pa.cov_perc)))
StoNsquared_stat= np.zeros((len(pa.a_con), len(pa.cov_perc)))
# These two desired quantities are independent under the fudge fractional error level so don't loop over that here, just pass a dummy value
for i in range(0,len(pa.a_con)):
	for j in range(0, len(pa.cov_perc)):
		print "Running, a #"+str(i+1)+" rho #"+str(j+1)
		(StoNsq_stat, nonsense, StoNsq_stat_sysB)	=	get_gammaIA_cov(rp_cents, rp_bins, fid_gIA, boost_fid, Ncorr_fid, Ncorr_err, pa.cov_perc[j], pa.a_con[i], 0.01)
		StoNsquared_stat[i,j] = StoNsq_stat
		StoNsquared_stat_sysB[i,j] = StoNsq_stat_sysB
		
"""# The level of StoN due to sys errors related to z is independent of rho, don't loop over this, just pass a dummy value		
StoNsquared_sysz = np.zeros((len(pa.a_con), len(pa.fudge_frac_level)))		
for i in range(0,len(pa.a_con)):
	for k in range(0,len(pa.fudge_frac_level)):	
		print "Running, a #"+str(i+1)+" frac sys err level #" + str(k+1)
		(nonsense, StoNsq_sysz, nonsense)	=	get_gammaIA_cov(rp_cents, rp_bins, fid_gIA, boost_fid, Ncorr_fid, Ncorr_err, 0.1, pa.a_con[i], pa.fudge_frac_level[k])
		StoNsquared_sysz[i,k] = StoNsq_sysz
		print "frac level=", pa.fudge_frac_level[k],"StoN=", 1./np.sqrt(StoNsq_sysz)"""
		
np.savetxt('./txtfiles/StoNsq_stat_shapes_survey='+SURVEY+'.txt', StoNsquared_stat)
np.savetxt('./txtfiles/StoNsq_stat_sysB_shapes_survey='+SURVEY+'.txt', StoNsquared_stat_sysB)
np.savetxt('./txtfiles/StoNsq_sysz_shapes_survey='+SURVEY+'.txt', StoNsquared_sysz)

np.savetxt('./txtfiles/a_survey='+SURVEY+'.txt', pa.a_con)	
np.savetxt('./txtfiles/rho_survey='+SURVEY+'.txt', pa.cov_perc)	
np.savetxt('./txtfiles/fudgelevels_survey='+SURVEY+'.txt', pa.fudge_frac_level)	
	
