# This is a script which constrains the (amplitude? parameters of a model?) of intrinsic alignments, using an updated version of the method of Blazek et al 2012 in which it is not assumed that only excess galaxies contribute to IA (rather than source galaxies which are close to the lens along the line-of-sight can contribute.)

import numpy as np
import IA_params_Fisher as pa
import scipy
import scipy.integrate
import matplotlib.pyplot as plt

########## FUNCTIONS ##########

def setup_rp_bins(rmin, rmax, nbins):
	""" Sets up the bins of projected radius (in units Mpc/h) """

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

def com(z_, H_, OmC_, OmB_, OmR_, OmN_):
	""" Gets the comoving distance in units of Mpc/h at a given redshift. """

	OmL_ = 1. - OmC_ - OmB_ - OmR_ - OmN_

	def chi_int(z):
	 	return 1. / (H_ * ( (OmC_+OmB_)*(1+z)**3 + OmL_ + (OmR_+OmN_) * (1+z)**4 )**(0.5))

	if hasattr(z_, "__len__"):
		chi=np.zeros((len(z_)))
		for zi in range(0,len(z_)):
			chi[zi] = scipy.integrate.quad(chi_int,0,z_[zi])[0]
	else:
		chi = scipy.integrate.quad(chi_int, 0, z_)[0]

	return chi

def get_areas(bins, z_eff):
	""" Gets the area of each projected radial bin, in square arcminutes. z_eff = effective lens redshift. """	

	# Areas in units (Mpc/h)^2
	areas_mpch = np.zeros(len(bins)-1)
	for i in range(0, len(bins)-1):
		areas_mpch[i] = np.pi * (bins[i+1]**2 - bins[i]**2) 

	#Comoving distance out to effective lens redshift in Mpc/h
	chi_eff = com(z_eff, pa.H0, pa.OmC, pa.OmB, pa.OmR, pa.OmN)

	# Areas in square arcminutes (466560000 / pi = sqAM in a sphere)
	areas_sqAM = areas_mpch * (466560000. / np.pi) / (4 * np.pi * chi_eff**2)

	return areas_sqAM

def get_NofZ(a, zs, z_min, z_max, zpts):
	""" Returns the dNdz of the sources as a function of photometric redshift, as well as the z points at which it is evaluated."""

	z = scipy.linspace(z_min, z_max, zpts)
	#nofz_ = z ** a * np.exp( - (z / (z_med / np.sqrt(2.))) ** b )
	nofz_ = (z / zs)**(a-1) * np.exp( -0.5 * (z / zs)**2)	

	#Normalisation:
	norm = scipy.integrate.simps(nofz_, z)
	nofz = nofz_ / norm 

	return (z, nofz)

def get_NofZ_excess(a, zs, z_min, z_max, zpts, z_l_, sig_ex_LOS, sig_ex_rp, rp_bin_centers_):
	""" Returns the dNdz of sources including excess galaxies (as opposed to just the smooth dNdz)"""
	
	# Get the smooth dNdz which is independent of projected radius and contains no "excess" galaxies.
	(z, nofz) = get_NofZ(a, zs, z_min, z_max, zpts)

	# Use a broad Gaussin in r_p for the amplitude of the line-of-sight Gaussian
	amplitude_excess= np.zeros(len(rp_bin_centers_))
	for ai in range(0,len(rp_bin_centers_)):
		amplitude_excess[ai] = 10000000.*np.exp(-(rp_bin_centers_[ai])**2) / (2. * sig_ex_rp**2) / (np.sqrt(2.*np.pi) * sig_ex_rp)
		print "amplitude excess=", amplitude_excess[ai]

	# We add a Gaussian in line-of-sight distance and projected radius to model the excess (this may or may not be a good model, just something to try for now).
	excess_model_LOS = [0]*len(rp_bin_centers_)
	for ai in range(0,len(rp_bin_centers_)):
		excess_model_LOS[ai] = amplitude_excess[ai]*np.exp(-(z-z_l_)**2 / (2*sig_ex_LOS**2)) / (np.sqrt(2.*np.pi) * sig_ex_LOS) 
	
	nofz_excess = [0]*len(rp_bin_centers_)
	for ri in range(0,len(rp_bin_centers_)):
		nofz_excess[ri] = nofz + excess_model_LOS[ri]

	return (z, nofz_excess)

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
	plt.semilogx(rp_cent, quant, 'ko')
	plt.xlabel('$r_p$')
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
	""" zeff_ is the effective redshift of the lenses. frac_ is the fraction of sources in the sample. ns_ is the number density of sources per square arcminute. nl_ is the number density of lenses per square degree. A is the survey area in square degrees."""

	# Get the area of each projected bin in square arcminutes
	bin_areas       =       get_areas(rp_bins_, zeff_)

	N_ls_pbin = nl_ * A * ns_ * bin_areas * frac_

	return N_ls_pbin

def setup_shapenoise_cov(e_rms, N_ls_pbin):
	""" Returns a diagonal covariance matrix in bins of projected radius for a measurement dominated by shape noise. Elements are e_{rms}^2 / (N_ls) where N_ls is the number of l/s pairs relevant to each projected radius bin.""" 

	cov = np.diag(e_rms**2 / N_ls_pbin)
	
	return cov

def sigma_e(z_s_):
	""" Returns a value for the model for the per-galaxy noise as a function of source redshift"""

	# This is a dummy things for now
	sig_e = 1.0 * np.ones(len(z_s_))

	return sig_e

def sum_weights(z_l, z_min_s, z_max_s, dNdz_tag, erms, rp_bins_, rp_bin_c_):
	""" Returns the sum over lens-source pairs of the estimated weights, in each projected radial bin. Generically applicable for different subsamples of both sources and lens-source pairs"""	
	""" dNdz_tag labels whether we should use dNdz with ('lens') or without ('rand') an added bump for excess galaxies (i.e. summing over all or over some part of rand """

	# At the moment this is just for a given bin in r_p.

	if (dNdz_tag =='rand'):
		dNdz_rp_list = [0]* (len(rp_bins_)-1)
		for i in range(0,len(rp_bins_)-1):
			# For the case where dNdz of sources is just the smooth source distribution, it is independent of projected radius
			(z_s, dNdz_rp_list[i]) = get_NofZ(pa.alpha, pa.zs, z_min_s, z_max_s, pa.zpts)

	if (dNdz_tag =='lens'):
		#dNdz_rp_list = [0]* (len(rp_bins_)-1)
		for i in range(0,len(rp_bins_)-1):
			(z_s, dNdz_rp_list) = get_NofZ_excess(pa.alpha, pa.zs, z_min_s, z_max_s, pa.zpts, z_l, 0.01, 3., rp_bin_c_)

	plot_nofz(dNdz_rp_list[0], z_s, './excess_bin1.pdf')
	plot_nofz(dNdz_rp_list[1], z_s, './excess_bin2.pdf')
	plot_nofz(dNdz_rp_list[2], z_s, './excess_bin3.pdf')
	plot_nofz(dNdz_rp_list[4], z_s, './excess_bin4.pdf')
	plot_nofz(dNdz_rp_list[5], z_s, './excess_bin5.pdf')
	plot_nofz(dNdz_rp_list[6], z_s, './excess_bin6.pdf')
		

	SigC_t = get_SigmaC_theory(z_s,	z_l, pa.H0, pa.OmC, pa.OmB, pa.OmR, pa.OmN)

	b_Sig = get_bSigma(z_s, z_l)

	sig_e = sigma_e(z_s)

	sum_ans = [0]*(len(rp_bins_)-1)
	for i in range(0,len(rp_bins_)-1):
		Integrand = dNdz_rp_list[i]  / (SigC_t**2 * b_Sig**2 * (erms* np.ones(len(z_s))**2 + sig_e**2))
		#print "integrand=", Integrand
		sum_ans[i] = scipy.integrate.simps(Integrand, z_s)

	return sum_ans

def sum_weights_Sigma():
	""" Returns the sum over lens-source pairs of the estimated weights multiplied by the critical surface density. Generically applicable for different subsamples of both sources and lens-source pairs"""

	return

def get_bSigma(z_s_, z_l_):
	""" Returns the photo-z bias to the estimated critical surface density. In principle this is a model fit from the spectroscopic subsample of data. """

	# This is a dummy return value for now
	b_Sig = 1. * np.ones(len(z_s_))

	return b_Sig

def get_boost(z_l_, zmin_, zmax_, erms_, rp_bin_, rp_cents_):
	""" Returns the boost factor in radial bins """

	numerator = sum_weights(z_l_, zmin_, zmax_, 'lens', erms_, rp_bin_, rp_cents_)
	denominator = sum_weights(z_l_, zmin_, zmax_, 'rand', erms_, rp_bin_, rp_cents_)

	Boost = np.asarray(numerator) / np.asarray(denominator)

	plot_quant_vs_rp(Boost, rp_cents_, './boost.png')

	return

def get_F():
	""" Returns F (the weighted fraction of lens-source pairs from the smooth dNdz which are contributing to IA) """

	return

def get_Sig_IA():
	""" Returns the value of <\Sigma_c>_{IA} in radial bins """

	return

def get_est_DeltaSig():
	""" Returns the value of tilde Delta Sigma in bins"""

	return

def get_DeltaSig_theory():
	""" Returns the theoretical value of Delta Sigma in bins """

	return

def get_SigmaC_theory(z_s_, z_l_, H_, OmC_, OmB_, OmR_, OmN_):
	""" Returns the theoretical value of Sigma_c, the critcial surface mass density """

	com_s = com(z_s_, H_, OmC_, OmB_, OmR_, OmN_) 
	com_l = com(z_l_, H_, OmC_, OmB_, OmR_, OmN_) 

	a_l = 1. / (z_l_ + 1.)

	a_s = 1. / (z_s_ + 1.)

	# This is missing a factor of c^2 / 4piG - I'm hoping this cancels everywhere? Check.
	Sigma_c = (a_l * a_s * com_s) / ((a_s*com_s - a_l * com_l) * com_l)

	return Sigma_c

def get_cz():
	""" Returns the value of the photo-z bias parameter c_z"""

	return

def subtract_var(var_1, var_2, covar):
	""" Takes the variance of two non-independent  Gaussian random variables, and their covariance, and returns the variance of their difference."""
	
	var_diff = var_1 + var_2 - 2 * covar

	return var_diff

def get_gammaIA_cov(Cov_1, Cov_2, sys_level, covar):
	""" Takes information about the uncertainty on constituent elements of gamma_{IA} and combines them to get the covariance matrix of gamma_{IA} in projected radial bins."""
	""" We assume that the statistical uncertainties affecting the estimated Delta Sigma and the boost-related factor B-1+F are uncorrelated between different projected radial bins, i.e., their covariance matrices in projected radial bins are diagnonal and elements can be taken individually."""

	# Set up the diagonal, statistical-error part of the matrix
	gammaIA_diag = np.diag(np.zeros(len(np.diag(Cov_1))))

	cz_1 = get_cz()
	cz_2 = get_cz()
	DeltaCov_1 = setup_shapenoise_cov()
	DeltaCov_2 = setup_shapenoise_cov()
	DeltaSig_1 = get_est_DeltaSig()
	DeltaSig_2 = get_est_DeltaSig()
	BCov_1 = get_Btermcov()
	BCov_2 = get_Btermcov()
	Sig_IA_1 = get_Sig_IA()
	Sig_IA_2 = get_Sig_IA()
	BFfac_1 = get_B() - 1. + get_F()
	BFfac_2 = get_B() - 1. + get_F()

	# Calculate same
	for i in range(0,len(np.diag(Cov_1))):	 
		gammaIA_diag[i,i] = gamIA_fid[i]**2 * (subtract_var(cz_1*DeltaCov_1[i,i], cz_2*DeltaCov_2[i,i], covar_Deta[i]) / (cz_1 * DeltaSig_1 - cz_2 * DeltaSig_2)**2 + subtract_var(cz_1 * Sig_IA_1 * BCov_1[i,i], cz_2 * Sig_IA_2 * BCov_2[i,i]) / (cz_1 * Sig_IA_1 * BFfac_1 - cz_2 * Sig_IA_2 * BFfac_2)**2)

	# Set up the systematics contribution
	#sys_level is a single value for each systematic error contribution
	sys_mat = np.zeros((len(np.diag(Cov_1)), len(np.diag(Cov_1))))
	# Get a single systematics matrix by adding systematic errors in quadrature to each other and then to statistical errors.
	for i in range(0,len(sys_level)):
		sys_mat = sys_level[i]**2 * np.ones((len(np.diag(Cov_1)), len(np.diag(Cov_1)))) + sys_mat
	
	gammaIA_cov = sys_mat + gammaIA_diag

	return gammaIA_cov

def get_fid_gIA(rp_bins_1):
	""" This function computes the fiducial value of gamma_IA in each projected radial bin."""

	# This is a dummy function for now.

	fidvals = 0.000*np.ones(len(rp_bins_1)-1)

	return fidvals

def plot_variance(cov_1, fidvalues_1, bin_centers, filename):
	""" Takes a covariance matrix, a vector of the fiducial values of the object in question, and the edges of the projected radial bins, and makes a plot showing the fiducial values and 1-sigma error bars from the diagonal of the covariance matrix. Outputs this plot to location 'filename'."""


	fig_sub=plt.subplot(111)
        plt.rc('font', family='serif', size=20)
        #fig_sub=fig.add_subplot(111) #, aspect='equal')
	fig_sub.set_xscale("log")
	#fig_sub.set_yscale("log", nonposy='clip')
        fig_sub.errorbar(bin_centers,fidvalues_1, yerr = np.sqrt(np.diag(cov_1)), fmt='o')
        fig_sub.set_xlabel('$r_p$')
        fig_sub.set_ylabel('$\gamma_{IA}$')
        fig_sub.tick_params(axis='both', which='major', labelsize=12)
        fig_sub.tick_params(axis='both', which='minor', labelsize=12)
        plt.tight_layout()
        plt.savefig(filename)

	return  

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

# Set up projected bins
rp_bins 	= 	setup_rp_bins(pa.rp_min, pa.rp_max, pa.N_bins)
rp_cent		=	rp_bins_mid(rp_bins)

# Set up the redshift distribution of sources
(z, dNdz)	=	get_NofZ(pa.alpha, pa.zs, pa.zmin, pa.zmax, pa.zpts)

# Testing for getting fiducial value of various quantities in order to include fractional statistical error on the boost-like factor
#sum_w = sum_weights(pa.zeff, pa.zeff, pa.zeff+pa.delta_z, 'lens', pa.e_rms_a, rp_bins, rp_cent)

get_boost(pa.zeff, pa.zeff, pa.zeff+pa.delta_z, pa.e_rms_a, rp_bins, rp_cent)

exit()

# Get the fraction of dndz covered by each source sample.
frac_a 	= 	get_z_frac(pa.zeff, pa.zeff+pa.delta_z, dNdz, z)
frac_b	=	get_z_frac(pa.zeff+pa.delta_z, pa.zmax, dNdz, z)

# Get the number of lens source pairs for each source sample in projected radial bins
N_ls_pbin_a	=	get_perbin_N_ls(rp_bins, pa.zeff, frac_a, pa.n_s, pa.n_l, pa.Area)
N_ls_pbin_b	=	get_perbin_N_ls(rp_bins, pa.zeff, frac_b, pa.n_s, pa.n_l, pa.Area)

# Get the covariance matrix in projected radial bins of Delta Sigma for samples a and b
Cov_a		=	setup_shapenoise_cov(pa.e_rms_a, N_ls_pbin_a)
Cov_b		=	setup_shapenoise_cov(pa.e_rms_b, N_ls_pbin_b)

# Combine the constituent covariance matrices to get the covariance matrix for gamma_IA in projected radial bins
Cov_gIA		= 	get_gammaIA_cov(Cov_a, Cov_b, pa.sys_sigc, np.zeros(len(np.diag(Cov_b))))

# Get the fiducial value of gamma_IA in each projected radial bin
fid_gIA		=	get_fid_gIA(rp_bins)

# Output a plot showing the 1-sigma error bars on gamma_IA in projected radial biplot_variance
plot_variance(Cov_gIA, fid_gIA, rp_cent, pa.plotfile)

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
