# This is a script which forecasts constraints on IA using multiple shape measurement methods.

SURVEY = 'SDSS'  # Set the survey here; this tells everything which parameter file to import.

import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import subprocess
import shutil
import shared_functions_setup as setup
import shared_functions_wlp_wls as ws

########## FUNCTIONS ##########


def sum_weights(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, erms, rp_bin_c_, dNdz_par, pz_par):
	""" Returns the sum over rand-source pairs of the estimated weights, in each projected radial bin. Pass different z_min_s and z_max_s to get rand-close, rand-far, and all-rand cases."""
	
	(z_ph, dNdz_ph) = N_of_zph_unweighted(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, dNdz_par, pz_par)
	
	frac = scipy.integrate.simps(dNdz_ph, z_ph)
	
	sum_ans = [0]*(len(rp_bin_c_))
	for i in range(0,len(rp_bin_c_)):
		Integrand = dNdz_ph* weights(erms, z_ph)
		sum_ans[i] = scipy.integrate.simps(Integrand, z_ph)

	return sum_ans


def shapenoise_cov(e_rms, z_p_l, z_p_h, B_samp, rp_c, rp, dNdzpar, pzpar):
	""" Returns a diagonal covariance matrix in bins of projected radius for a measurement dominated by shape noise. Elements are 1 / (sum_{ls} w), carefully normalized, in each bin. """
	
	# Get the area of each projected radial bin in square arcminutes
	bin_areas       =       setup.get_areas(rp, pa.zeff, SURVEY)
	
	sum_denom =  pa.n_l * pa.Area_l * bin_areas * pa.n_s * np.asarray(sum_weights(pa.zmin, pa.zmax, pa.zmin, pa.zmax, z_p_l, z_p_h, pa.zmin_ph, pa.zmax_ph, e_rms, rp_c, dNdzpar, pzpar))
	
	cov = 1. / (B_samp * sum_denom)
	
	#sum_denom =  np.asarray(sum_weights(pa.zmin, pa.zmax, pa.zmin, pa.zmax, z_p_l, z_p_h, pa.zmin_ph, pa.zmax_ph, e_rms, rp_c, dNdzpar, pzpar))
	
	#print "weights=", sum_weights(pa.zmin, pa.zmax, pa.zmin, pa.zmax, z_p_l, z_p_h, pa.zmin_ph, pa.zmax_ph, e_rms, rp_c, dNdzpar, pzpar)
	
	#cov = 1. / (sum_denom)
	
	#print "cov=", cov
	
	return cov

	
def gamma_fid(rp):
	""" Returns the fiducial gamma_IA from a combination of terms from different models which are valid at different scales """
	
	wgg_rp = ws.wgg_full(rp, pa.fsat_LRG, pa.fsky, pa.bd_shapes, pa.bs_shapes, './txtfiles/wgg_1h_survey='+pa.survey+'.txt', './txtfiles/wgg_2h_survey='+pa.survey+'_kpts='+str(pa.kpts_wgg)+'.txt', './plots/wgg_full_shapes_survey='+pa.survey+'.pdf', SURVEY)
	wgp_rp = ws.wgp_full(rp, pa.bd_shapes, pa.Ai_shapes, pa.ah_shapes, pa.q11_shapes, pa.q12_shapes, pa.q13_shapes, pa.q21_shapes, pa.q22_shapes, pa.q23_shapes, pa.q31_shapes, pa.q32_shapes, pa.q33_shapes, './txtfiles/wgp_1h_survey='+pa.survey+'.txt','./txtfiles/wgp_2h_survey='+pa.survey+'.txt', './plots/wgp_full_survey='+pa.survey+'.pdf', SURVEY)
	
	gammaIA = wgp_rp / (wgg_rp + 2. * pa.close_cut) 
	
	return gammaIA
	

####### GETTING ERRORS #########

def get_boost(rp_cents_, propfact):
	""" Returns the boost factor in radial bins. propfact is a tunable parameter giving the proportionality constant by which boost goes like projected correlation function (= value at 1 Mpc/h). """

	Boost = propfact *(rp_cents_)**(-0.8) + np.ones((len(rp_cents_)))# Empirical power law fit to the boost, derived from the fact that the boost goes like projected correlation function.

	return Boost

def N_of_zph_unweighted(z_a_def, z_b_def, z_a_norm, z_b_norm, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, dNdz_par, pz_par):
	""" Returns dNdz_ph, the number density in terms of photometric redshift, defined and normalized over the photo-z range (z_a_norm_ph, z_b_norm_ph), normalized over the spec-z range (z_a_norm, z_b_norm), but defined on the spec-z range (z_a_def, z_b_def)"""
	
	(z, dNdZ) = setup.get_NofZ_unnormed(dNdz_par, pa.dNdztype, z_a_def, z_b_def, 1000)
	(z_norm, dNdZ_norm) = setup.get_NofZ_unnormed(dNdz_par, pa.dNdztype, z_a_norm, z_b_norm, 1000)
	
	z_ph_vec_def = scipy.linspace(z_a_def_ph, z_b_def_ph, 5000)
	z_ph_vec_norm = scipy.linspace(z_a_norm_ph, z_b_norm_ph, 5000)
	
	int_dzs = np.zeros(len(z_ph_vec_def))
	for i in range(0,len(z_ph_vec_def)):
		int_dzs[i] = scipy.integrate.simps(dNdZ*setup.p_z(z_ph_vec_def[i], z, pz_par, pa.pztype), z)
		
	int_dzs_norm = np.zeros(len(z_ph_vec_norm))
	for i in range(0,len(z_ph_vec_norm)):
		int_dzs_norm[i] = scipy.integrate.simps(dNdZ_norm*setup.p_z(z_ph_vec_norm[i], z_norm, pz_par, pa.pztype), z_norm)
		
	norm = scipy.integrate.simps(int_dzs_norm, z_ph_vec_norm)
	
	return (z_ph_vec_def, int_dzs / norm)

def N_corr(rp_cent, dNdz_par_num, pzpar_num, dNdz_par_denom, pzpar_denom, boost):
	""" Computes the correction factor which accounts for the fact that some of the galaxies in the photo-z defined source sample are actually higher-z and therefore not expected to be affected by IA. We have two sets of alpha, zs, and sigz because one set is affected by sys errors and the other isn't."""
	
	sumW_insamp = sum_weights(z_close_low, z_close_high, pa.zmin, pa.zmax, z_close_low, z_close_high, pa.zmin_ph, pa.zmax_ph, pa.e_rms_mean, rp_cent, dNdz_par_num, pzpar_num)
	sumW_intotal = sum_weights(pa.zmin, pa.zmax, pa.zmin, pa.zmax, z_close_low, z_close_high, pa.zmin_ph, pa.zmax_ph, pa.e_rms_mean, rp_cent, dNdz_par_denom, pzpar_denom)
	
	Corr_fac = 1. - (1. / boost) * ( 1. - (np.asarray(sumW_insamp) / np.asarray(sumW_intotal))) # fraction of the galaxies in the source sample which have spec-z in the photo-z range of interest.
	
	return Corr_fac

def N_corr_err(rp_cents_, boost_error_file, dNdzpar, pzpar):
	""" Gets the error on N_corr from statistical error on the boost and systemtic error on the boost. 
	Does NOT include systematic error on dNdz/pz. """
	
	sumW_insamp = sum_weights(z_close_low, z_close_high, pa.zmin, pa.zmax, z_close_low, z_close_high, pa.zmin_ph, pa.zmax_ph, pa.e_rms_mean, rp_cents_, dNdzpar, pzpar)
	sumW_intotal = sum_weights(pa.zmin, pa.zmax, pa.zmin, pa.zmax, z_close_low, z_close_high, pa.zmin_ph, pa.zmax_ph, pa.e_rms_mean, rp_cents_, dNdzpar, pzpar)
	boost = get_boost(rp_cents_, pa.boost_assoc)
	
	# Get statistical errors on the boost
	boost_err_stat = boost_errors(rp_cents_, boost_error_file)
	print "boost err stat=", boost_err_stat
	# Get systematic errors on the boost
	boost_err_sys = (pa.boost_sys - 1.) * boost
	print "boost err sys=", boost_err_sys
	# Get total (add in quadrature)
	boost_err = np.sqrt(boost_err_sys**2 + boost_err_stat**2)
	
	sig_Ncorr = (boost_err / boost**2) * np.sqrt(1. - np.asarray(sumW_insamp)/ np.asarray(sumW_intotal))

	return sig_Ncorr

def boost_errors(rp_bins_c, filename):
	""" Imports a file with 2 columns, [rp (kpc/h), sigma(boost-1)]. Interpolates and returns the value of the error on the boost at the center of each bin. """
	
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

def weights(e_rms, z_):
	""" Returns the inverse variance weights as a function of redshift. """
	
	weights = (1./(sigma_e(z_)**2 + e_rms**2)) * np.ones(len(z_))
	
	return weights

def get_cov_btw_methods(cov_a_, cov_b_, covperc):
	""" Get the covariance between the methods given their correlation """
	
	cov_btw_methods = covperc * np.sqrt(cov_a_) * np.sqrt(cov_b_)
	
	return cov_btw_methods

def subtract_var(var_1, var_2, covar):
	""" Takes the variance of two non-independent  Gaussian random variables, and their covariance, and returns the variance of their difference."""
	
	var_diff = var_1 + var_2 - 2 * covar

	return var_diff

def get_gammaIA_cov(rp_cents_, rp_bins_, gIA_fid, covperc, a_con, fudge_Ncorr):
	""" Takes the covariance matrices of the constituent elements of gamma_{IA} and combines them to get the covariance matrix of gamma_{IA} in projected radial bins."""
	
	boost_fid = get_boost(rp_cents_, pa.boost_assoc)
	
	# These are actually the same so there's no real need to compute it twice.
	shear_cov_1 = shapenoise_cov(pa.e_rms_mean, z_close_low, z_close_high, boost_fid, rp_cents_, rp_bins_, pa.dNdzpar_fid, pa.pzpar_fid) 
	shear_cov_2 = shapenoise_cov(pa.e_rms_mean, z_close_low, z_close_high, boost_fid, rp_cents_, rp_bins_, pa.dNdzpar_fid, pa.pzpar_fid) 
	
	# Get the covariance between the shear of the two shape measurement methods in each bin:
	shear_covar = get_cov_btw_methods(shear_cov_1, shear_cov_2, covperc)
	
	# factor correcting for galaxies which have higher spec-z than the sample but which end up in the sample.
	corr_fac_fid = N_corr(rp_cents_, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdzpar_fid, pa.pzpar_fid, boost_fid)  

	if (pa.survey=='SDSS'):
		# statistical error from Ncorr and systematic error due to effect on the boost (not including sys err due to dNdz / pz effects).
		corr_fac_err = N_corr_err(rp_cents_, pa.sigBF_a, pa.dNdzpar_fid, pa.pzpar_fid)
	elif(pa.survey=='LSST_DESI'):
		corr_fac_err = N_corr_err(rp_cents_, 'nonsense', pa.dNdzpar_fid, pa.pzpar_fid)
	else:
		print "We don't have support for that survey yet."
		exit()

	#cov_mat_tot = np.diag(np.zeros(len(shear_cov_1)))
	cov_mat_stat = np.diag(np.zeros(len(shear_cov_1)))
	cov_mat_sys = np.diag(np.zeros(len(shear_cov_1)))
	#subvar = np.zeros(len(shear_cov_1))
	#denom = np.zeros(len(shear_cov_1))
	for i in range(0,len(shear_cov_1)):	
		#cov_mat_tot[i, i] = (1.-a_con)**2 * gIA_fid[i]**2 * (( corr_fac_err[i]**2 / corr_fac_fid[i]**2)  + (subtract_var(shear_cov_1[i], shear_cov_2[i], shear_covar[i]) / (corr_fac_fid[i]**2 * (1.-a_con)**2 * gIA_fid[i]**2)) + (pa.fudge_Ncorr)**2 )
		cov_mat_stat[i,i] = (1.-a_con)**2 * gIA_fid[i]**2 * (( corr_fac_err[i]**2 / corr_fac_fid[i]**2)  + (subtract_var(shear_cov_1[i], shear_cov_2[i], shear_covar[i]) / (corr_fac_fid[i]**2 * (1.-a_con)**2 * gIA_fid[i]**2)))
		cov_mat_sys[i,i] = (1.-a_con)**2 * gIA_fid[i]**2 * ( (fudge_Ncorr)**2 )
		
	# Save the parts we want to compare
	#save_comp = np.column_stack((subvar, denom))
	#np.savetxt('./txtfiles/compare_survey='+SURVEY+'_a='+str(a_con)+'_covperc='+str(covperc)+'_sig=NOTgIA.txt', save_comp)
		
	for i in range(0,len((rp_cents_))):	
		for j in range(0,len((rp_cents_))):
			if (i != j):
				cov_mat_sys[i,j] = np.sqrt(cov_mat_sys[i,i]) * np.sqrt(cov_mat_sys[j,j])
				
	cov_mat_tot = cov_mat_stat + cov_mat_sys
	
	#print "cov_mat_sys=", cov_mat_sys
	#print "cov_mat_stat=", cov_mat_stat
	#print "cov_mat_tot=", cov_mat_tot
	
	# Compute associated signal to noise	
	Cov_inv_tot = np.linalg.inv(cov_mat_tot)
	StoNsq_tot = np.dot(gIA_fid* (1.-a_con), np.dot(Cov_inv_tot, gIA_fid * (1-a_con)))
	
	Cov_inv_stat = np.linalg.inv(cov_mat_stat)
	StoNsq_stat = np.dot(gIA_fid* (1.-a_con), np.dot(Cov_inv_stat, gIA_fid * (1-a_con)))
	
	NtoSsq_sys = 1./StoNsq_tot - 1./StoNsq_stat
	StoNsq_sys = 1. / NtoSsq_sys
	
	"""plt.figure()
	plt.loglog(rp_cents, np.sqrt(np.diag(cov_mat)) / ((1. - a_con) * gIA_fid) , 'mo')
	plt.ylim(10**(-1), 10**2)
	plt.savefig('./plots/statcov_acon='+str(a_con)+'covperc='+str(covperc)+'.pdf')
	plt.close()
	
	save_variance = np.column_stack((rp_cents_, np.sqrt(np.diag(cov_mat)) / ((1.-a_con) * gIA_fid)))
	np.savetxt('./txtfiles/fractional_toterror_shapemethod_LRG-shapes_covperc='+str(covperc)+'_a='+str(a_con)+'_fudgeNcorrErr='+str(pa.fudge_Ncorr)+'.txt', save_variance)"""

	return StoNsq_sys, StoNsq_stat

####### PLOTTING / OUTPUT #######

def plot_variance(cov_1, fidvalues_1, bin_centers, covperc, a_con):
	""" Takes a covariance matrix, a vector of the fiducial values of the object in question, and the edges of the projected radial bins, and makes a plot showing the fiducial values and 1-sigma error bars from the diagonal of the covariance matrix. Outputs this plot to location 'filename'."""

	fig_sub=plt.subplot(111)
	plt.rc('font', family='serif', size=20)
	#fig_sub=fig.add_subplot(111) #, aspect='equal')
	fig_sub.set_xscale("log")
	fig_sub.set_yscale("log")
	fig_sub.errorbar(bin_centers,fidvalues_1*(1-a_con), yerr = np.sqrt(np.diag(cov_1)), fmt='mo')
	#fig_sub.errorbar(bin_centers,fidvalues_1, yerr = np.sqrt(np.diag(cov_1)), fmt='mo')
	fig_sub.set_xlabel('$r_p$')
	fig_sub.set_ylabel('$\gamma_{IA}(1-a)$')
	#fig_sub.set_ylabel('$\gamma_{IA}$')
	fig_sub.set_ylim(10**(-5), 0.05)
	#fig_sub.set_ylim(10**(-4), 0.1)
	fig_sub.tick_params(axis='both', which='major', labelsize=12)
	fig_sub.tick_params(axis='both', which='minor', labelsize=12)
	fig_sub.set_title('a='+str(a_con)+', cov='+str(covperc))
	plt.tight_layout()
	plt.savefig('./plots/errorplot_stat+sys_shapemethod_LRG+shapes_a='+str(a_con)+'covperc='+str(covperc)+'_fudgeNcorr='+str(pa.fudge_Ncorr)+'.pdf')
	plt.close()

	return  

def plot_quant_vs_rp(quant, rp_cent, file):
	""" Plots any quantity vs the center of redshift bins"""

	plt.figure()
	plt.loglog(rp_cent, quant, 'ko')
	plt.xlabel('$r_p$')
	plt.tick_params(axis='both', which='major', labelsize=12)
	plt.tick_params(axis='both', which='minor', labelsize=12)
	plt.tight_layout()
	plt.savefig(file)

	return
	
def plot_quant_vs_quant(quant1, quant2, file):
	""" Plots any quantity vs any other."""

	plt.figure()
	plt.loglog(quant1, quant2, 'ko')
	plt.tick_params(axis='both', which='major', labelsize=12)
	plt.tick_params(axis='both', which='minor', labelsize=12)
	plt.tight_layout()
	plt.savefig(file)

	return

######### FISHER STUFF ##########

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


######## MAIN CALLS ##########

# Import the parameter file:
if (SURVEY=='SDSS'):
	import params as pa
elif (SURVEY=='LSST_DESI'):
	import params_LSST_DESI as pa
else:
	print "We don't have support for that survey yet; exiting."
	exit()

# Set up projected radial bins
rp_bins 	= 	setup.setup_rp_bins(pa.rp_min, pa.rp_max, pa.N_bins) # Edges
rp_cents	=	setup.rp_bins_mid(rp_bins) # Centers

# Set up to get z as a function of comoving distance
z_of_com, com_of_z 	= 	setup.z_interpof_com(SURVEY)

# Get the redshift corresponding to the maximum separation from the effective lens redshift at which we assume IA may be present (pa.close_cut is the separation in comoving Mpc/h)
(z_close_high, z_close_low)	= 	setup.get_z_close(pa.zeff, pa.close_cut, SURVEY)

#boost_fid = get_boost(rp_cents, pa.boost_assoc)
#corr_fac_fid = N_corr(rp_cents, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdzpar_fid, pa.pzpar_fid, boost_fid)
#print "corr_fac_fid=", corr_fac_fid
#exit()

# Get the number of lens source pairs for the source sample in projected radial bins
#N_ls_pbin	=	get_perbin_N_ls(rp_bins, pa.zeff, pa.n_s, pa.n_l, pa.Area_l, rp_cents, pa.alpha_fid, pa.zs_fid, pa.sigz_fid)

# Get the fiducial value of gamma_IA in each projected radial bin (this takes a while so only do it once
fid_gIA		=	gamma_fid(rp_cents)

# Get the covariance matrix in projected radial bins of gamma_t for both shape measurement methods
#Cov_a		=	shapenoise_cov(pa.e_rms_a, z_close_low, z_close_high, , rp_c, rp, alpha_dN, zs_dN, sigz) #setup_shapenoise_cov(pa.e_rms_a, N_ls_pbin) 
#Cov_b		=	setup_shapenoise_cov(pa.e_rms_b, N_ls_pbin)

StoNsquared_stat = np.zeros((len(pa.a_con), len(pa.cov_perc)))
#Ratio = np.zeros(len(pa.fudge_Ncorr))
for i in range(0,len(pa.a_con)):
	for j in range(0, len(pa.cov_perc)):
		#for k in range(0,len(pa.fudge_Ncorr)):
		print "Running, a #"+str(i+1)+" cov % #"+str(j+1) #+" Ncorr fudge #"+str(k+1)
		# Combine the constituent covariance matrices to get the covariance matrix for gamma_IA in projected radial bins
		StoNsq_sys, StoNsq_stat	=	get_gammaIA_cov(rp_cents, rp_bins, fid_gIA, pa.cov_perc[j], pa.a_con[i], pa.fudge_Ncorr) 
		#Ratio[k] = np.sqrt(StoNsq_sys / StoNsq_stat)
		StoNsquared_stat[i,j] = StoNsq_stat
		#Cov_sys 	=	get_gammaIA_sys_cov(rp_cents, fid_gIA, pa.cov_perc[j], pa.a_con, pa.dNdzpar_sys[i], pa.pzpar_sys[i])
		#Cov_tot		=	get_gamma_tot_cov(Cov_sys, Cov_stat)
		
	#save_tot = np.column_stack((rp_cents,np.sqrt(np.diag(Cov_tot)) / ((1.-pa.a_con) * fid_gIA)))
	#np.savetxt('./txtfiles/fractional_toterror_shapemethod_LRG-shapes_covperc='+str(pa.cov_perc[j])+'_a='+str(pa.a_con)+'_%sys='+str((pa.dNdzpar_fid[0]- pa.dNdzpar_sys[i][0]) / a.dNdzpar_fid[0])+'_mod_dNdz_pz_pars.txt', save_tot)

	# Output a plot showing the 1-sigma error bars on gamma_IA in projected radial bins
	#plot_variance(Cov_tot, fid_gIA, rp_cents, pa.cov_perc[j], pa.a_con)
	
#save_Ncorr = np.column_stack((pa.fudge_Ncorr, Ratio))
#np.savetxt('./txtfiles/fudgeNcorr.txt', save_Ncorr)
	
np.savetxt('./txtfiles/StoNsq_shapes_survey='+SURVEY+'.txt', StoNsquared_stat)

np.savetxt('./txtfiles/a_StoNstat_survey='+SURVEY+'.txt', pa.a_con)	
np.savetxt('./txtfiles/cov_perc_StoNstat_survey='+SURVEY+'.txt', pa.cov_perc)	
	
exit() # Below is Fisher stuff, don't worry about this yet

# Get the parameter derivatives required to construct the Fisher matrix
ders            =       par_derivs(pa.par, rp_cents)

# Get the Fisher matrix
fish            =       get_Fisher(ders, Cov_gIA)

# If desired, cut parameters which you want to fix from Fisher matrix:
fish_cut        =       cut_Fisher(fish, None)

# Get the covariance matrix from either fish or fish_cut, and marginalise over any desired parameters
parCov          =       get_par_Cov(fish_cut, None)

# Output whatever we want to know about the parameters:
par_const_output(fish_cut, parCov)
