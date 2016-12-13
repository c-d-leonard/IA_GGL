# This is a script which constrains the (amplitude? parameters of a model?) of intrinsic alignments, using multiple shape measurement methods.

import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt

########## FUNCTIONS ##########

####### SET UP & BASICS #######

def setup_rp_bins(rmin, rmax, nbins):
	""" Sets up the edges of the bins of projected radius """
	
	bins = scipy.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
	
	return bins

def rp_bins_mid(rp_edges):
        """ Gets the middle of each projected radius bin."""

        logedges=np.log10(rp_edges)
        bin_centers=np.zeros(len(rp_edges)-1)
        for ri in range(0,len(rp_edges)-1):
                bin_centers[ri]    =       10**((logedges[ri+1] - logedges[ri])/2. +logedges[ri])

        return bin_centers

def com(z_):
	""" Gets the comoving distance in units of Mpc/h at a redshift or a set of redshifts. """

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

	z_vec = scipy.linspace(0., 10., 1000) # This hardcodes that we don't care about anything over z=10.

	com_vec = com(z_vec)

	z_of_com = scipy.interpolate.interp1d(com_vec, z_vec)

	return	z_of_com

def get_z_close(z_l, cut_MPc_h):
	""" Gets the z above z_l which is the highest z at which we expect IA to be present for that lens. cut_Mpc_h is that separation in Mpc/h."""

	com_l = com(z_l) # Comoving distance to z_l, in Mpc/h

	tot_com_high = com_l + cut_MPc_h
	tot_com_low = com_l - cut_MPc_h

	# Convert tot_com back to a redshift.

	z_cl_high = z_of_com(tot_com_high)
	z_cl_low = z_of_com(tot_com_low)

	return (z_cl_high, z_cl_low)

def get_areas(bins, z_eff):
        """ Gets the area of each projected radial bin, in square arcminutes. z_eff = effective lens redshift. """

        # Areas in units (Mpc/h)^2
        areas_mpch = np.zeros(len(bins)-1)
        for i in range(0, len(bins)-1):
                areas_mpch[i] =  np.pi * (bins[i+1]**2 - bins[i]**2)

        #Comoving distance out to effective lens redshift in Mpc/h
        chi_eff = com(z_eff)

        # Areas in square arcminutes (466560000 / pi = sqAM in a sphere)
        areas_sqAM = areas_mpch * (466560000. / np.pi) / (3 * np.pi * chi_eff**2)

        return areas_sqAM
	
def get_NofZ_unnormed(a, zs, z_min, z_max, zpts):
	""" Returns the dNdz of the sources as a function of photometric redshift, as well as the z points at which it is evaluated."""
	z = scipy.linspace(z_min, z_max, zpts)
	
	nofz_ = (z / zs)**(a-1) * np.exp(-0.5 * ( z / zs)**2)

	return (z, nofz_)

def get_z_frac(z_1, z_2):
        """ Gets the fraction of sources in a sample between z_1 and z_2, for dNdz given by a normalized nofz_1 computed at redshifts z_v"""
        
        # The normalization factor should be the norm over the whole range of z for which the number density of the survey is given, i.e. z min to zmax.       
        (z_ph_vec, N_of_p) = N_of_zph_unweighted(pa.zmin, pa.zmax, pa.zmin, pa.zmax, pa.zmin_ph, pa.zmax_ph)
        
        i_z1 = next(j[0] for j in enumerate(z_ph_vec) if j[1]>=z_1)
        i_z2 = next(j[0] for j in enumerate(z_ph_vec) if j[1]>=z_2)

        frac = scipy.integrate.simps(N_of_p[i_z1:i_z2], z_ph_vec[i_z1:i_z2])

        return frac

def get_perbin_N_ls(rp_bins_, zeff_, ns_, nl_, A):
	""" Gets the number of lens/source pairs relevant to each bin of projected radius """
	""" zeff_ is the effective redshift of the lenses. frac_ is the fraction of sources in the sample. ns_ is the number density of sources per square arcminute. nl_ is the number density of lenses per square degree. A is the survey area in square degrees."""
        
	frac_		=	get_z_frac(z_close_low, z_close_high)

	# Get the area of each projected bin in square arcminutes
	bin_areas       =       get_areas(rp_bins_, zeff_)

	N_ls_pbin = nl_ * A * ns_ * bin_areas * frac_

	return N_ls_pbin

####### GETTING ERRORS #########

def N_of_zph_unweighted(z_a_def, z_b_def, z_a_norm, z_b_norm, z_a_norm_ph, z_b_norm_ph):
	""" Returns dNdz_ph, the number density in terms of photometric redshift, defined and normalized over the photo-z range (z_a_norm_ph, z_b_norm_ph), normalized over the spec-z range (z_a_norm, z_b_norm), but defined on the spec-z range (z_a_def, z_b_def)"""
	
	(z, dNdZ) = get_NofZ_unnormed(pa.alpha, pa.zs, z_a_def, z_b_def, pa.zpts)
	(z_norm, dNdZ_norm) = get_NofZ_unnormed(pa.alpha, pa.zs, z_a_norm, z_b_norm, pa.zpts)
	
	z_ph_vec = scipy.linspace(z_a_norm_ph, z_b_norm_ph, 5000)
	
	int_dzs = np.zeros(len(z_ph_vec))
	int_dzs_norm = np.zeros(len(z_ph_vec))
	for i in range(0,len(z_ph_vec)):
		int_dzs[i] = scipy.integrate.simps(dNdZ*p_z(z_ph_vec[i], z, pa.sigz), z)
		int_dzs_norm[i] = scipy.integrate.simps(dNdZ_norm*p_z(z_ph_vec[i], z_norm, pa.sigz), z_norm)
		
	norm = scipy.integrate.simps(int_dzs_norm, z_ph_vec)
	
	return (z_ph_vec, int_dzs / norm)
	
def N_of_zph_weighted(z_a_def, z_b_def, z_a_norm, z_b_norm, z_a_norm_ph, z_b_norm_ph):
	""" Returns dNdz_ph, the number density in terms of photometric redshift, defined and normalized over the photo-z range (z_a_norm_ph, z_b_norm_ph), normalized over the spec-z range (z_a_norm, z_b_norm), but defined on the spec-z range (z_a_def, z_b_def). This version returns the weighted photo-z number density, primarily for getting the correction factor calculatd in N_corr."""
	
	(z, dNdZ) = get_NofZ_unnormed(pa.alpha, pa.zs, z_a_def, z_b_def, pa.zpts)
	(z_norm, dNdZ_norm) = get_NofZ_unnormed(pa.alpha, pa.zs, z_a_norm, z_b_norm, pa.zpts)
	
	z_ph_vec = scipy.linspace(z_a_norm_ph, z_b_norm_ph, 5000)
	
	weights_ = weights(pa.e_rms_mean, z_ph_vec)
	
	int_dzs = np.zeros(len(z_ph_vec))
	int_dzs_norm = np.zeros(len(z_ph_vec))
	for i in range(0,len(z_ph_vec)):
		int_dzs[i] = scipy.integrate.simps(dNdZ*p_z(z_ph_vec[i], z, pa.sigz), z)
		int_dzs_norm[i] = scipy.integrate.simps(dNdZ_norm*p_z(z_ph_vec[i], z_norm, pa.sigz), z_norm)
		
	norm = scipy.integrate.simps(int_dzs_norm*weights_, z_ph_vec)
	
	return (z_ph_vec, weights_*int_dzs / norm)

def N_in_samp(z_a, z_b, e_rms_weights):
	""" Number of galaxies in the photometric redshift range of the sample (assumed z_eff of lenses to z_close) from the SPECTROSCOPIC redshift range z_a to z_b """
	
	(z_ph, N_of_zp) = N_of_zph_weighted(z_a, z_b, pa.zmin, pa.zmax, z_close_low, z_close_high)

	answer = scipy.integrate.simps(N_of_zp, z_ph)
	
	return (answer)
	
def N_corr():
	""" Computes the correction factor which accounts for the fact that some of the galaxies in the photo-z defined source sample are actually higher-z and therefore not expected to be affected by IA"""
	
	N_tot = N_in_samp(pa.zmin, pa.zmax, pa.e_rms_mean)
	N_close = N_in_samp(z_close_low, z_close_high, pa.e_rms_mean)
	
	Corr_fac = N_close / N_tot # fraction of the galaxies in the source sample which have spec-z in the photo-z range of interest.
	
	return Corr_fac

def sigma_e(z_s_, s_to_n):
	""" Returns a value for the model for the per-galaxy noise as a function of source redshift"""

	sig_e = 2. / s_to_n * np.ones(len(z_s_))

	return sig_e

def weights(e_rms, z_):
	""" Returns the inverse variance weights as a function of redshift. """
	
	weights = (sigma_e(z_, pa.S_to_N)**2 + e_rms**2 * np.ones(len(z_)))
	
	return weights
	
def p_z(z_ph, z_sp, sigz):
	""" Returns the probability of finding a photometric redshift z_ph given that the true redshift is z_sp. """
	
	# I'm going to use a Gaussian probability distribution here for now
	p_z_ = np.exp(-(z_ph - z_sp)**2 / (2.*(sigz*(1.+z_sp))**2)) / (np.sqrt(2.*np.pi)*(sigz * (1. + z_sp)))
	
	return p_z_

def get_fid_gIA(rp_bins_c):
	""" This function computes the fiducial value of gamma_IA in each projected radial bin."""
	
	# This is a dummy thing for now.
	#fidvals = np.zeros(len(rp_bins_c))
	fidvals = pa.A_fid * np.asarray(rp_bins_c)**pa.beta_fid

	return fidvals

def setup_shapenoise_cov(e_rms, N_ls_pbin):
	""" Returns a diagonal covariance matrix in bins of projected radius for a measurement dominated by shape noise. Elements are e_{rms}^2 / (N_ls) where N_ls is the number of l/s pairs relevant to each projected radius bin.""" 

	cov = np.diag(e_rms**2 / N_ls_pbin)
	
	return cov

def subtract_var(var_1, var_2, covar):
	""" Takes the variance of two non-independent  Gaussian random variables, and their covariance, and returns the variance of their difference."""
	
	var_diff = var_1 + var_2 - 2 * covar

	return var_diff

def get_gammaIA_stat_cov(Cov_1, Cov_2, covar):
	""" Takes the covariance matrices of the constituent elements of gamma_{IA} and combines them to get the covariance matrix of gamma_{IA} in projected radial bins."""
	
	corr_fac = N_corr()  # factor correcting for galaxies which have higher spec-z than the sample but which end up in the sample.

	stat_mat = np.diag(np.zeros(len(np.diag(Cov_1))))

	for i in range(0,len(np.diag(Cov_1))):	
		stat_mat[i,i] = subtract_var(Cov_1[i,i], Cov_2[i,i], covar[i]) / corr_fac

	return stat_mat

def get_gammaIA_sys_cov(rp_cents_, sys):
	""" Takes the centers of rp_bins and a systematic error sources from dNdz_s uncertainty (assumed to affect each r_p bin in the same way) and adds them to each other in quadrature."""
	
	#Sys^2 = (sys_dNdz^2) * N_corr**2 * (1-a)**2 * gamma_IA**2
	
	sys_mat = np.zeros((len(rp_cents_), len(rp_cents_)))

	gIa_fid = get_fid_gIA(rp_cents_)
	
	corr_fac = N_corr()
	
	for i in range(0,len(rp_cents_)):
		for j in range(0,len(rp_cents_)):
			sys_mat[i,j] = sys**2 * (1-pa.a_con)**2 * corr_fac**2 * gIa_fid[i] * gIa_fid[j]

	return sys_mat
	
def get_gamma_tot_cov(sys_mat, stat_mat):
	""" Takes the covariance matrix from statistical error and systematic error, and adds them to get the total covariance matrix. Assumes stat and sys errors should be added in quadrature."""
	
	tot_cov = sys_mat+stat_mat
	
	return tot_cov

####### PLOTTING / OUTPUT #######

def plot_variance(cov_1, fidvalues_1, bin_centers, filename):
	""" Takes a covariance matrix, a vector of the fiducial values of the object in question, and the edges of the projected radial bins, and makes a plot showing the fiducial values and 1-sigma error bars from the diagonal of the covariance matrix. Outputs this plot to location 'filename'."""

	fig=plt.figure()
	plt.rc('font', family='serif', size=20)
	fig_sub=fig.add_subplot(111) #, aspect='equal')
	fig_sub.set_xscale("log")
	fig_sub.set_yscale("log")
	fig_sub.errorbar(bin_centers,fidvalues_1*(1-pa.a_con), yerr = np.sqrt(np.diag(cov_1)), fmt='o')
	fig_sub.set_xlabel('$r_p$')
	fig_sub.set_ylabel('$\gamma_{IA}(1-a)$')
	fig_sub.tick_params(axis='both', which='major', labelsize=12)
	fig_sub.tick_params(axis='both', which='minor', labelsize=12)
	plt.tight_layout()
	plt.savefig(filename)

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
	plt.plot(quant1, quant2, 'ko')
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
import IA_params_shapes as pa

# Set up projected radial bins
rp_bins 	= 	setup_rp_bins(pa.rp_min, pa.rp_max, pa.N_bins) # Edges
rp_cents	=	rp_bins_mid(rp_bins) # Centers

# Set up to get z as a function of comoving distance
z_of_com 	= 	z_interpof_com()

# Get the redshift corresponding to the maximum separation from the effective lens redshift at which we assume IA may be present (pa.close_cut is the separation in Mpc/h)
(z_close_high, z_close_low)	= 	get_z_close(pa.zeff, pa.close_cut)

# Get the number of lens source pairs for the source sample in projected radial bins
N_ls_pbin	=	get_perbin_N_ls(rp_bins, pa.zeff, pa.n_s, pa.n_l, pa.Area)

# Get the covariance matrix in projected radial bins of gamma_t for both shape measurement methods
Cov_a		=	setup_shapenoise_cov(pa.e_rms_a, N_ls_pbin)
Cov_b		=	setup_shapenoise_cov(pa.e_rms_b, N_ls_pbin)

# Combine the constituent covariance matrices to get the covariance matrix for gamma_IA in projected radial bins
Cov_stat	=	get_gammaIA_stat_cov(Cov_a, Cov_a, pa.covar_DSig)
Cov_sys 	=	get_gammaIA_sys_cov(rp_cents, pa.syslist)

Cov_tot		=	get_gamma_tot_cov(Cov_sys, Cov_stat)

# Get the fiducial value of gamma_IA in each projected radial bin
fid_gIA		=	get_fid_gIA(rp_cents)

# Output a plot showing the 1-sigma error bars on gamma_IA in projected radial bins
plot_variance(Cov_tot, fid_gIA, rp_cents, pa.plotfile)

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
