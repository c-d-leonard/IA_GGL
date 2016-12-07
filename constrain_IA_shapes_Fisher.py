# This is a script which constrains the (amplitude? parameters of a model?) of intrinsic alignments, using multiple shape measurement methods.

import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt

########## FUNCTIONS ##########

####### SET UP & BASICS #######

def setup_rp_bins(rmin, rmax, nbins):
	""" Sets up the bins of projected radius """

	# These are the *edges* of the bins.
	bins = scipy.logspace(np.log10(rmin), np.log10(rmax), nbins+1)

	return bins

def rp_bins_mid(rp_edges):
        """ Gets the middle of each projected radius bin."""

        # Get the midpoints of the projected radius bins
        logedges=np.log10(rp_edges)
        bin_centers=np.zeros(len(rp_edges)-1)
        for ri in range(0,len(rp_edges)-1):
                bin_centers[ri]    =       10**((logedges[ri+1] - logedges[ri])/2. +logedges[ri])

        return bin_centers

def com(z_):
	""" Gets the comoving distance in units of Mpc/h at a given redshift. """

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

	z_vec = scipy.linspace(0., 2100., 100000) # This hardcodes that we don't care about anything over z=2100

	com_vec = com(z_vec)

	z_of_com = scipy.interpolate.interp1d(com_vec, z_vec)

	return	z_of_com

def get_z_close(z_l, cut_MPc_h):
	""" Gets the z above z_l which is the highest z at which we expect IA to be present for that lens. cut_Mpc_h is that separation in Mpc/h."""

	com_l = com(z_l) # Comoving distance to z_l, in Mpc/h

	tot_com = com_l + cut_MPc_h

	# Convert tot_com back to a redshift.

	z_cl = z_of_com(tot_com)

	return z_cl

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

def get_NofZ(a, zs, z_min, z_max, zpts):
	""" Returns the dNdz of the sources as a function of photometric redshift, as well as the z points at which it is evaluated."""
	z = scipy.linspace(z_min, z_max, zpts)
	
	nofz_ = (z / zs)**(a-1) * np.exp(-0.5 * ( z / zs)**2)

	#Normalisation:
	norm = scipy.integrate.simps(nofz_, z)
	nofz = nofz_ / norm 

	return (z, nofz)

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

def get_z_frac(z_1, z_2, nofz_1, z_v):
        """ Gets the fraction of sources in a sample between z_1 and z_2, for dNdz given by a normalized nofz_1 computed at redshifts z_v"""

        # Get the index of the z vector closest to the limits of the integral
        i_z1 = next(j[0] for j in enumerate(z_v) if j[1]>=z_1)
        i_z2 = next(j[0] for j in enumerate(z_v) if j[1]>=z_2)

        frac = scipy.integrate.simps(nofz_1[i_z1:i_z2], z_v[i_z1:i_z2])

        return frac

def get_perbin_N_ls(rp_bins_, zeff_, frac_, ns_, nl_, A):
        """ Gets the number of lens/source pairs relevant to each bin of projected radius """
        """ zeff_ is the effective redshift of the lenses. frac_ is the fraction of sources in the sample. ns_ is the           number density of sources per square arcminute. nl_ is the number density of lenses per square degree. A is the survey area in square degrees."""

        # Get the area of each projected bin in square arcminutes
        bin_areas       =       get_areas(rp_bins_, zeff_)

        N_ls_pbin = nl_ * A * ns_ * bin_areas * frac_

        return N_ls_pbin

####### GETTING ERRORS #########

def sum_weights(z_l, z_min_s, z_max_s, erms, rp_bins_, rp_bin_c_):
	""" Returns the sum over rand-source pairs of the estimated weights, in each projected radial bin. Pass different z_min_s and z_max_s to get rand-close, rand-far, and all-rand cases."""
	
	(z_s, dNdz_s) = get_NofZ(pa.alpha, pa.zs, z_min_s, z_max_s, pa.zpts)
	
	SigC_t = get_SigmaC_theory(z_s,	z_l)

	b_Sig = get_bSigma(z_s, z_l)

	sig_e = sigma_e(z_s, pa.S_to_N)

	sum_ans = [0]*(len(rp_bins_)-1)
	for i in range(0,len(rp_bins_)-1):
		Integrand = dNdz_s  / (SigC_t**2 * b_Sig**2 * (erms**2*np.ones(len(z_s)) + sig_e**2))
		sum_ans[i] = scipy.integrate.simps(Integrand, z_s)

	return sum_ans

def sigma_e(z_s_, s_to_n):
	""" Returns a value for the model for the per-galaxy noise as a function of source redshift"""

	# This is a dummy things for now
	sig_e = 2. / s_to_n * np.ones(len(z_s_))

	return sig_e

def get_SigmaC_theory(z_s_, z_l_):
	""" Returns the theoretical value of Sigma_c, the critcial surface mass density """

	com_s = com(z_s_) 
	com_l = com(z_l_) 

	a_l = 1. / (z_l_ + 1.)

	a_s = 1. / (z_s_ + 1.)

	# This is missing a factor of c^2 / 4piG - I'm hoping this cancels everywhere? Check.
	Sigma_c = (a_l * a_s * com_s) / ((a_s*com_s - a_l * com_l) * com_l)

	return Sigma_c

def get_bSigma(z_s_, z_l_):
	
	""" Returns the photo-z bias to the estimated critical surface density. In principle this is a model fit from the spectroscopic subsample of data. """

	# This is a dummy return value for now
	b_Sig = 1. * np.ones(len(z_s_))

	return b_Sig

def p_z(z_ph, z_sp, sigz):
	""" Returns the probability of finding a photometric redshift z_ph given that the true redshift is z_sp. """
	
	# I'm going to use a Gaussian probability distribution here, but you could change that.
	p_z_ = np.exp(-(z_ph - z_sp)**2 / (2.*sigz**2)) / (np.sqrt(2.*np.pi)*sigz)
	
	return p_z_

def get_boost(rp_cents_, propfact):
	""" Returns the boost factor in radial bins. propfact is a tunable parameter giving the proportionality constant by which boost goes like projected correlation function (= value at 1 Mpc/h). """

	Boost = propfact *(rp_cents_)**(-0.8) + np.ones((len(rp_cents_))) # Empirical power law fit to the boost, derived from the fact that the boost goes like projected correlation function.

	plot_quant_vs_rp(Boost, rp_cents_, './boost.png')

	return Boost

def get_F(z_l, z_close_max, z_max_samp, erms, rp_bins_, rp_bin_c):
	""" Returns F (the weighted fraction of lens-source pairs from the smooth dNdz which are contributing to IA) """

	# Sum over `rand-close'
	numerator = sum_weights(z_l, z_l, z_close_max, erms, rp_bins_, rp_bin_c)

	#Sum over all `rand'
	denominator = sum_weights(z_l, z_l, z_max_samp, erms, rp_bins_, rp_bin_c)

	F = np.asarray(numerator) / np.asarray(denominator)
	
	return F

def get_Sig_IA(z_l, z_min_s, z_cut_IA, z_max_samp, erms, rp_bins_, rp_bin_c_, boost):
	""" Returns the value of <\Sigma_c>_{IA} in radial bins """
	
	# There are four terms here. The two in the denominators are sums over randoms (or sums over lenses that can be written as randoms * boost), and these are already set up to calculate.
	denom_rand_far = sum_weights(z_l, z_cut_IA, z_max_samp, erms, rp_bins_, rp_bin_c_)
	denom_rand = sum_weights(z_l, z_l, z_max_samp, erms, rp_bins_, rp_bin_c_)
	
	# The two in the numerator require summing over weights and Sigma_C. 
	
	#For the sum over rand-far, this follows directly from the same type of expression as when summing weights:
	(z_s, dNdz_s) = get_NofZ(pa.alpha, pa.zs, z_cut_IA, z_max_samp, pa.zpts)
	SigC_t = get_SigmaC_theory(z_s,	z_l)
	b_Sig = get_bSigma(z_s, z_l)
	sig_e = sigma_e(z_s, pa.S_to_N)
	
	rand_far_num = [0]*(len(rp_bins_)-1)
	for i in range(0,len(rp_bins_)-1):
		Integrand = dNdz_s  / (SigC_t * b_Sig * (erms**2*np.ones(len(z_s)) + sig_e**2))
		rand_far_num[i] = scipy.integrate.simps(Integrand, z_s)
			
	# The other numerator sum is itself the sum of (a sum over all randoms) and (a term which represents the a sum over excess). 
	
	# The rand part:
	(z_s_all, dNdz_s_all) = get_NofZ(pa.alpha, pa.zs, z_l, z_max_samp, pa.zpts)
	SigC_t_all = get_SigmaC_theory(z_s_all,	z_l)
	b_Sig_all = get_bSigma(z_s_all, z_l)
	sig_e_all = sigma_e(z_s_all, pa.S_to_N)
	
	rand_all_num = [0]*(len(rp_bins_)-1)
	for i in range(0,len(rp_bins_)-1):
		Integrand = dNdz_s_all  / (SigC_t_all * b_Sig_all * (erms**2*np.ones(len(z_s_all)) + sig_e_all**2))
		rand_all_num[i] = scipy.integrate.simps(Integrand, z_s_all)
	
	#The excess part:
	excess_num = [0]*(len(rp_bins_)-1)	
	for i in range(0, len(rp_bins_)-1):
			p_z_arr = p_z(z_s_all, z_l, pa.sigz)
			Integrand = p_z_arr / (SigC_t_all * (sig_e_all**2 + erms**2 * np.ones(len(z_s_all))))
			excess_num[i] = scipy.integrate.simps(Integrand, z_s_all)
		
	Sig_IA = ((np.asarray(boost)-1.)*np.asarray(excess_num) + np.asarray(rand_all_num) - np.asarray(rand_far_num)) / (np.asarray(boost)*np.asarray(denom_rand) - np.asarray(denom_rand_far))

	return Sig_IA

def get_Sig_all(z_l, z_min_s, z_max_samp, erms, rp_bins_, rp_bin_c_, boost):
	""" Returns the value of <\Sigma_c>_{all} in radial bins """
	
	# The denominator is a sums over lenses that can be written as randoms * boost.
	denom_rand = sum_weights(z_l, z_min_s, z_max_samp, erms, rp_bins_, rp_bin_c_)
	
	# The numerator requires summing over weights and Sigma_C, over randoms and excess. 
	
	# The rand part:
	(z_s, dNdz_s) = get_NofZ(pa.alpha, pa.zs, z_min_s, z_max_samp, pa.zpts)
	SigC_t = get_SigmaC_theory(z_s,	z_l)
	b_Sig = get_bSigma(z_s, z_l)
	sig_e = sigma_e(z_s, pa.S_to_N)
	
	rand_num = [0]*(len(rp_bins_)-1)
	for i in range(0,len(rp_bins_)-1):
		Integrand = dNdz_s  / (SigC_t * b_Sig * (erms**2*np.ones(len(z_s)) + sig_e**2))
		rand_num[i] = scipy.integrate.simps(Integrand, z_s)
	
	#The excess part:
	excess_num = [0]*(len(rp_bins_)-1)	
	for i in range(0, len(rp_bins_)-1):
			p_z_arr = p_z(z_s, z_l, pa.sigz)
			Integrand = p_z_arr / (SigC_t * (sig_e**2 + erms**2 * np.ones(len(z_s))))
			excess_num[i] = scipy.integrate.simps(Integrand, z_s)
		
	Sig_all = ((np.asarray(boost)-1.)*np.asarray(excess_num) + np.asarray(rand_num)) / (np.asarray(boost)*np.asarray(denom_rand))

	return Sig_all

def get_est_DeltaSig_diff(z_l, zSmax, z_close_, erms, rp_bins_, rp_cents_, a_):
	""" Returns the value of est(Delta Sigma, method 1) - est(Delta Sigma, method 2) """
	
	boost = get_boost(rp_cents_, pa.Boost_prop)
	F = get_F(z_l, z_close_, zSmax, erms, rp_bins_, rp_cents_)
	gIA = get_fid_gIA(rp_cents_) # Nonlinear alignments model not yet included
	SigIA = get_Sig_IA(z_l, zSmin, z_close_, zSmax, erms, rp_bins_, rp_cents_, boost)
	
	diff = gIA * (1-a) * SigIA * (boost -1 + F)
	
	return diff 

def get_fid_gIA(rp_bins_c):
	""" This function computes the fiducial value of gamma_IA in each projected radial bin."""
	
	# This needs to be updated to nonlinear alignment model.
	
	fidvals = pa.A_fid * np.asarray(rp_bins_c)**pa.beta_fid

	return fidvals

def setup_shapenoise_cov(e_rms, N_ls_pbin):
	""" Returns a diagonal covariance matrix in bins of projected radius for a measurement dominated by shape noise. Elements are e_{rms}^2 / (N_ls) where N_ls is the number of l/s pairs relevant to each projected radius bin.""" 

	cov = np.diag(e_rms**2 / N_ls_pbin)

	#print "cov=", cov
	
	return cov

def subtract_var(var_1, var_2, covar):
	""" Takes the variance of two non-independent  Gaussian random variables, and their covariance, and returns the variance of their difference."""
	
	var_diff = var_1 + var_2 - 2 * covar

	return var_diff

def get_gammaIA_diff_cov(Cov_1, Cov_2, sys1, covar):
	""" Takes the covariance matrices of the constituent elements of gamma_{IA} and combines them to get the covariance matrix of gamma_{IA} in projected radial bins."""

	gammaIA_diag = np.diag(np.zeros(len(np.diag(Cov_1))))

	for i in range(0,len(np.diag(Cov_1))):	
		gammaIA_diag[i,i] = subtract_var(Cov_1[i,i], Cov_2[i,i], covar[i]) 

	sys_mat = sys1**2 * np.ones((len(np.diag(Cov_1)), len(np.diag(Cov_1)))) # Assume the systematic error (here on the excess critical surface density) is fully correlated 
		
	gammaIA_cov = sys_mat + gammaIA_diag

	return gammaIA_cov

def get_fid_gIA(rp_bins_1):
	""" This function computes the fiducial value of gamma_IA in each projected radial bin."""

	# This is a dummy function for now.

	fidvals = np.zeros(len(rp_bins_1)-1)

	return fidvals

####### PLOTTING / OUTPUT #######

def plot_variance(cov_1, fidvalues_1, bin_centers, filename):
	""" Takes a covariance matrix, a vector of the fiducial values of the object in question, and the edges of the projected radial bins, and makes a plot showing the fiducial values and 1-sigma error bars from the diagonal of the covariance matrix. Outputs this plot to location 'filename'."""

	fig=plt.figure()
        plt.rc('font', family='serif', size=20)
        fig_sub=fig.add_subplot(111) #, aspect='equal')
	fig_sub.set_xscale("log")
        fig_sub.errorbar(bin_centers,fidvalues_1, yerr = np.sqrt(np.diag(cov_1)), fmt='o')
        fig_sub.set_xlabel('$r_p$')
        fig_sub.set_ylabel('$\gamma_{IA}$')
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
import IA_params_shapes as pa

# Set up projected bins
rp_bins 	= 	setup_rp_bins(pa.rp_min, pa.rp_max, pa.N_bins)
rp_cents	=	rp_bins_mid(rp_bins)

# Set up the redshift distribution of sources
(z, dNdz)	=	get_NofZ(pa.alpha,pa.zs, pa.zmin, pa.zmax, pa.zpts)

# Set up a function to get z as a function of comoving distance
z_of_com = z_interpof_com()

# Get the redshift corresponding to the maximum separation from the effective lens redshift at which we assume IA may be present (pa.close_cut is the separation in Mpc/h)
z_close = get_z_close(pa.zeff, pa.close_cut)

# Get the fraction of dndZ which is in the source sample
frac		=       get_z_frac(pa.zS_min, pa.zS_max, dNdz, z)

# Get the number of lens source pairs for the source sample in projected radial bins
N_ls_pbin	=	get_perbin_N_ls(rp_bins, pa.zeff, frac, pa.n_s, pa.n_l, pa.Area)

# Get the covariance matrix in projected radial bins of Delta Sigma for both shape measurement methods
Cov_a		=	setup_shapenoise_cov(pa.e_rms_a, N_ls_pbin)
Cov_b		=	setup_shapenoise_cov(pa.e_rms_b, N_ls_pbin)

# TESTING COMPONENTS THAT ARE BEING ADDED:
Boost = get_boost(rp_cents, pa.Boost_prop)
#F_a = get_F(pa.zeff, z_close, pa.zS_max, pa.e_rms_a, rp_bins, rp_cents)
#F_b = get_F(pa.zeff, z_close, pa.zS_max, pa.e_rms_b, rp_bins, rp_cents)
#b_Sig = get_bSigma(z, pa.zeff)
#p_ = p_z(z, pa.zeff, pa.sigz)
#sige = sigma_e(z, pa.S_to_N)
#Sigc_th = get_SigmaC_theory(z, pa.zeff)
#Sig_IA_a = get_Sig_IA(pa.zeff, pa.zS_min, z_close, pa.zS_max, pa.e_rms_a, rp_bins, rp_cents, Boost)
#Sig_IA_b = get_Sig_IA(pa.zeff, pa.zS_min, z_close, pa.zS_max, pa.e_rms_b, rp_bins, rp_cents, Boost)
#cov = get_gammaIA_diff_cov(Cov_a, Cov_b, pa.sys_sigc, pa.covar_DSig)
#Sig_all_a = get_Sig_all(pa.zeff, pa.zS_min, pa.zS_max, pa.e_rms_a, rp_bins, rp_cents, Boost)
#Sig_all_b = get_Sig_all(pa.zeff, pa.zS_min, pa.zS_max, pa.e_rms_b, rp_bins, rp_cents, Boost)
diffa = get_est_DeltaSig_diff(pa.zeff, pa.zSmax, z_close, pa.e_rms_a, rp_bins, rp_cents, pa.a)
diffb = get_est_DeltaSig_diff(pa.zeff, pa.zSmax, z_close, pa.e_rms_b, rp_bins, rp_cents, pa.a)
plot_quant_vs_rp(Sig_all_a, rp_cents, './Sigall_a.png')
plot_quant_vs_rp(Sig_all_b, rp_cents, './Sigall_b.png')
exit()


# Combine the constituent covariance matrices to get the covariance matrix for gamma_IA in projected radial bins
Cov_gIA		= 	get_gammaIA_cov(Cov_a, Cov_b, pa.sys_sigc, pa.covar_DSig)

# Get the fiducial value of gamma_IA in each projected radial bin
fid_gIA		=	get_fid_gIA(rp_bins)

# Output a plot showing the 1-sigma error bars on gamma_IA in projected radial bins
plot_variance(Cov_gIA, fid_gIA, rp_cents, pa.plotfile)

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
