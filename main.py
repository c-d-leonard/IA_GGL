# This is a script which forecasts constraints on IA using multiple shape measurement methods.

SURVEY = 'LSST_DESI'  # Set the survey here; this tells everything which parameter file to import.
print "SURVEY=", SURVEY
endfile = 'fixDls'

import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import shared_functions_setup as setup
import shared_functions_wlp_wls as ws
import os.path

np.set_printoptions(linewidth=240)

########## GENERIC FUNCTIONS ##########
	
def N_of_zph_unweighted(z_a_def, z_b_def, z_a_norm, z_b_norm, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, dNdz_par, pz_par):
	""" Returns dNdz_ph, the number density in terms of photometric redshift, defined and normalized over the photo-z range (z_a_norm_ph, z_b_norm_ph), normalized over the spec-z range (z_a_norm, z_b_norm), but defined on the spec-z range (z_a_def, z_b_def)"""
	
	# This function sets up the fiducial underlying redshift distribution in terms of spectroscopic redshifts. 
	#Since this is only for the sources, the LSST+DESI one can probably be used for our case.
	(z, dNdZ) = setup.get_NofZ_unnormed(dNdz_par, pa.dNdztype, z_a_def, z_b_def, 200, SURVEY)
	(z_norm, dNdZ_norm) = setup.get_NofZ_unnormed(dNdz_par, pa.dNdztype, z_a_norm, z_b_norm, 200, SURVEY)
	
	z_ph_vec_def = scipy.linspace(z_a_def_ph, z_b_def_ph, 200)
	z_ph_vec_norm = scipy.linspace(z_a_norm_ph, z_b_norm_ph, 200)
	
	# Convolve the spec-z distribution with the photo-z model. As long as the photo-z model is just a Gaussian we can use this.
	int_dzs = np.zeros(len(z_ph_vec_def))
	for i in range(0,len(z_ph_vec_def)):
		int_dzs[i] = scipy.integrate.simps(dNdZ*setup.p_z(z_ph_vec_def[i], z, pz_par, pa.pztype), z)
		
	int_dzs_norm = np.zeros(len(z_ph_vec_norm))
	for i in range(0,len(z_ph_vec_norm)):
		int_dzs_norm[i] = scipy.integrate.simps(dNdZ_norm*setup.p_z(z_ph_vec_norm[i], z_norm, pz_par, pa.pztype), z_norm)
		
	norm = scipy.integrate.simps(int_dzs_norm, z_ph_vec_norm)
	
	return (z_ph_vec_def, int_dzs / norm)

def sum_weights(photoz_sample, specz_cut, color_cut, dNdz_par, pz_par):
	""" Returns the sum over lens-source pairs of the estimated weights for a given source redshift cut and colour cut."""
	""" Charlie - Don't worry about the colour cuts, just use color_cut=all"""
	
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
	(zs_norm, dNdzs_norm) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 1000, SURVEY)
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
					zs, dNdzs = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zminclose[zi], zmaxclose[zi], 500, SURVEY)
				elif(specz_cut=='nocut'):
					z_ph = np.linspace(zminclose[zi], zmaxclose[zi], 500)
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


def estimator_with_mult_bias():
    """ This function should calculate from theory the expected spurious signal for (1-a)gammaIA, in the case when the two methods have different multiplicative biases."""
    
    # Start with a situation where we assume that the true gammaIA is 0.
    # Then, spurious signal = ((delta_m - delta m') gamma_t) / (B-1+F)

    # gamma_t and boost should be calcualted using a full 1halo term so needs a HOD for the lenses and for the sources.
    
    # need to get gamma_t from lensing only. In principle this will be the same for each estimator up to systematic issues (such as the mag bias difference).
    
    # We can probably get this directly from CCL by generating gammat but with a cosmo object which uses matter_power_spectrum = halo_model. 
    # We'll have to pick what we want for the other cosmo parameters mass_function and halo concentration (can use defaults for now but should think about this.)
    # Need to test to make sure this gives a different gammat than if we just use matter_power_spectrum = halofit.
    # This also annoyingly does not account for the halo occupation distribution of the galaxies, just the halo model for the dark matter.
    # How to do this? Would we possibly be okay with just perturbative bias expansion (fastPT)?
    
    # Probably we can use the function get_Pkgm_1halo (from shared_functions_wlp_wls.py). 
    # That function takes as input 'y' which is the 1halo power spectrum of matter with a given concentration model and an NFW profile. 
    # We can get something for this using the CCL function 'one_halo_matter_power'. 
    # We can then use CCL calculator mode with this as the power spectrum and the galaxy bias set to 1 to construct a tracer with suitable small scale bias. 
    # Can add 1halo and 2halo power spectra before constructing tracer in this way with calculator mode. Then CCL will do the FFT to gammat for us.
    
    # We also need the boost - this require an HOD for both lenses and sources. Still need to work out what to do here.
    
    # Also need F.
    
    # Then put them together with with the multiplicative bias offset values to get the estimated signal.
    
    return
    
def get_boost(rp_cents_, sample):
	""" Returns the boost factor in radial bins. propfact is a tunable parameter giving the proportionality constant by which boost goes like projected correlation function (= value at 1 Mpc/h). """
	
	# RE-RUN BOOSTS
	
	#propfact = np.loadtxt('./txtfiles/boosts/Boost_'+str(sample)+'_survey='+str(SURVEY)+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')

	#Boost = propfact *(rp_cents_)**(-0.8) + np.ones((len(rp_cents_)))# Empirical power law fit to the boost, derived from the fact that the boost goes like projected correlation function.
	
	Boost = np.loadtxt('./txtfiles/boosts/Boost_full_'+str(sample)+'_survey='+str(SURVEY)+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt') + np.ones((len(rp_cents_)))

	return Boost

def get_F(rp_bins_, dNdz_par, pz_par):
	""" Returns F (the weighted fraction of lens-source pairs from the smooth dNdz which are contributing to IA) """

	# Sum over `rand-close'
	numerator = sum_weights('close', 'close',  'all', dNdz_par, pz_par)

	#Sum over all `rand'
	denominator = sum_weights('close', 'nocut',  'all', dNdz_par, pz_par)

	F = np.asarray(numerator) / np.asarray(denominator)

	return F
	
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

####### GETTING COVARIANCE #########

def get_ns_partial():
	""" Gets the fractional value of ns appropriate for this subsample."""
	# To do this, just get the fraction of dNdzph within the sample:
	frac = sum_weights('close', 'nocut', 'all', pa.dNdzpar_fid, pa.pzpar_fid) / sum_weights('full', 'nocut', 'all', pa.dNdzpar_fid, pa.pzpar_fid)
	
	return frac * ns_tot
	
def get_combined_covariance(Clggterm, corr):
	""" Get the covariance matrix Cov(gammat(r) - gammat'(r), gamma(r') - gammat'(r'))"""
	# Clggterm is the halo-shapenoise terms. corr is the correlation coefficient between \gamma_rms for the methods
	
	# CHANGE TO CCL
	print("change this call to use ccl to get comoving distance.")
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

def get_gammaIA_cov(rp_cents_, rp_bins_, gIA_fid, covperc, a_con, fudge_F):
	""" Takes the covariance matrices of the constituent elements of gamma_{IA} and combines them to get the covariance matrix of gamma_{IA} in projected radial bins."""
	
	# Import Clgg-related covariance terms from separate Fourier-space script
	Clggterm_1 = np.loadtxt('./txtfiles/covmats/cov_gamt_extl_'+SURVEY+'_method=same_rms_rpts2000_lpts90000_Clggterm_'+endfile+'.txt')
	
	# Get the combined covariance Cov(gammat(r) - gammat'(r), gamma(r') - gammat'(r')):
	cov_gam_diff = get_combined_covariance(Clggterm_1, covperc)
	
	"""# Uncomment this section to check covariance matrix against the version calculated in real space (shape-noise only)
	# This gets the shape-noise from real-space methods for comparison - this matrix is diagonal by definition so it is output only as a vector of the diagonal elements
	shear_cov_1 = shapenoise_cov(pa.e_rms_a, rp_bins_, pa.dNdzpar_fid, pa.pzpar_fid) 
	shear_cov_2 = shear_cov_1
	shear_covar = get_cov_btw_methods(shear_cov_1, shear_cov_2, covperc)
	cov_old=np.zeros((pa.N_bins, pa.N_bins))
	for i in range(0, pa.N_bins):
		cov_old[i,i] = subtract_var(shear_cov_1[i], shear_cov_2[i], shear_covar[i])"""
	
	"""# Compare real-space shape noise only answer with fourier space answer (including CV)
	plt.figure()
	plt.loglog(rp_cents_, np.diag(cov_old), 'mo', label='real, combined')
	plt.hold(True)
	plt.loglog(rp_cents_, np.diag(cov_gam_diff), 'go', label='fourier, combined (with CV)')
	plt.xlabel('r_p')
	plt.ylabel('Variance')
	plt.ylim(10**(-11), 10**(-5))
	plt.legend()
	plt.savefig('./plots/check_gammat_extl_'+SURVEY+'.pdf')
	plt.close()
	
	cov_gam_diff = cov_old"""
	
	# Get statistical covariance matrix for (1-a) gamma_IA 
	cov_mat_stat = np.zeros((pa.N_bins, pa.N_bins))
	for i in range(0,pa.N_bins):
		for j in range(0,pa.N_bins):
			cov_mat_stat[i,j] = cov_gam_diff[i,j] /(boost[i] -1. +F_fid) / (boost[j]-1. +F_fid) 
	
	# Get the covariance matrix terms due to systematic error associated with the boost.
	# First get diagonal elements
	cov_mat_sysz_F = np.diag(np.zeros(pa.N_bins))
	cov_mat_sysB = np.diag(np.zeros(pa.N_bins))
	for i in range(0,pa.N_bins):	
		cov_mat_sysz_F[i,i] = (1.-a_con)**2 * gIA_fid[i]**2 * ( F_fid**2 * (fudge_F)**2 ) / (boost[i]-1. + F_fid)**2
		cov_mat_sysB[i,i] = (1.-a_con)**2 * gIA_fid[i]**2 * ( pa.boost_sys**2 / (boost[i]-1. + F_fid)**2)
		
	# Get off-diagonal elements by assuming full correlation 
	for i in range(0,len((rp_cents_))):	
		for j in range(0,len((rp_cents_))):
			if (i != j):
				cov_mat_sysz_F[i,j] = np.sqrt(cov_mat_sysz_F[i,i]) * np.sqrt(cov_mat_sysz_F[j,j])
				cov_mat_sysB[i,j] = np.sqrt(cov_mat_sysB[i,i]) * np.sqrt(cov_mat_sysB[j,j])
	
	# Get the covariance matrix for the case of stat + sys due to z
	cov_mat_stat_sysz = cov_mat_stat + cov_mat_sysz_F
	# And the covariance matrix for the case of stat + sys due to boost
	cov_mat_stat_sysB = cov_mat_stat + cov_mat_sysB
	
	# Save the stat + sysB signal to noise as a function of preojected radius at this a and rho
	SN_stat_sysB = gIA_fid * (1. - a_con) / (np.sqrt(np.diag(cov_mat_stat_sysB)))
	save_SN = np.column_stack((rp_cents_, SN_stat_sysB))
	np.savetxt('./txtfiles/StoN/StoN_sysB_stat_shapes_'+SURVEY+'_rlim='+str(pa.mlim)+'_a'+str(a_con)+'_rho'+str(covperc)+'_'+endfile+'.txt', save_SN)
	
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
	
	return
	

######## MAIN CALLS ##########

print("need to make a new parameter file for LSST sources + lenses and add this to the IF statement.")
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
# Should be able to get rid of this as everywhere this is called should be replaced with CCL call.
#z_of_com, com_of_z 	= 	setup.z_interpof_com(SURVEY)



# Get the values of the boost and F - we use these for the estimator as well as the covariance so just do them once.
boost = get_boost(rp_cents, 'close')
F_fid = get_F(rp_bins, pa.dNdzpar_fid, pa.pzpar_fid)

# Get the estimator given a level of multiplicative bias:
estimator_with_mult_bias() # need to add arguments and write this function.

ns = get_ns_partial()

# Get the desired S-to_N ratios for the case of stat error plus sys error from the boost (sys error due to z subdominant), for the case of sys error due to z, and for the case of statistical only.
StoNsquared_stat_sysB = np.zeros((len(pa.a_con), len(pa.cov_perc)))
StoNsquared_stat= np.zeros((len(pa.a_con), len(pa.cov_perc)))

# These two desired quantities are independent under the fudge fractional error level so don't loop over that here, just pass a dummy value
for i in range(0,len(pa.a_con)):
	for j in range(0, len(pa.cov_perc)):
		#print "Running, a #"+str(i+1)+" rho #"+str(j+1)
		(StoNsq_stat_sysB, nonsense, StoNsq_stat)	=	get_gammaIA_cov(rp_cents, rp_bins, fid_gIA, pa.cov_perc[j], pa.a_con[i], 0.01)
		StoNsquared_stat[i,j] = StoNsq_stat
		StoNsquared_stat_sysB[i,j] = StoNsq_stat_sysB
		
		
# The level of StoN due to sys errors related to z is independent of rho, don't loop over this, just pass a dummy value		
StoNsquared_sysz = np.zeros((len(pa.a_con), len(pa.fudge_frac_level)))		
for i in range(0,len(pa.a_con)):
	for k in range(0,len(pa.fudge_frac_level)):	
		#print "Running, a #"+str(i+1)+" frac sys err level #" + str(k+1)
		(nonsense, StoNsq_sysz, nonsense)	=	get_gammaIA_cov(rp_cents, rp_bins, fid_gIA, 0.5, pa.a_con[i], pa.fudge_frac_level[k])
		StoNsquared_sysz[i,k] = StoNsq_sysz
		print "frac level=", pa.fudge_frac_level[k],"StoN=", 1./np.sqrt(StoNsq_sysz)

if (SURVEY=='SDSS'):		
	np.savetxt('./txtfiles/StoN/StoNsq_stat_shapes_survey='+SURVEY+'_rlim='+str(pa.mlim)+'_fixB_'+endfile+'.txt', StoNsquared_stat)
	np.savetxt('./txtfiles/StoN/StoNsq_stat_sysB_shapes_survey='+SURVEY+'_rlim='+str(pa.mlim)+'_fixB_'+endfile+'.txt', StoNsquared_stat_sysB)
	np.savetxt('./txtfiles/StoN/StoNsq_sysz_shapes_survey='+SURVEY+'_rlim='+str(pa.mlim)+'_fixB_'+endfile+'.txt', StoNsquared_sysz)
elif (SURVEY=='LSST_DESI'):
	np.savetxt('./txtfiles/StoN/StoNsq_stat_shapes_survey='+SURVEY+'_rlim='+str(pa.mlim)+'_fixB_'+endfile+'.txt', StoNsquared_stat)
	np.savetxt('./txtfiles/StoN/StoNsq_stat_sysB_shapes_survey='+SURVEY+'_rlim='+str(pa.mlim)+'_fixB_'+endfile+'.txt', StoNsquared_stat_sysB)
	np.savetxt('./txtfiles/StoN/StoNsq_sysz_shapes_survey='+SURVEY+'_rlim='+str(pa.mlim)+'_fixB_'+endfile+'.txt', StoNsquared_sysz)
else:
	print "We don't have support for that survey yet. Exiting."
	exit()
	
np.savetxt('./txtfiles/a_survey='+SURVEY+'.txt', pa.a_con)	
np.savetxt('./txtfiles/rho_survey='+SURVEY+'.txt', pa.cov_perc)	
np.savetxt('./txtfiles/fudgelevels_survey='+SURVEY+'.txt', pa.fudge_frac_level)
	
