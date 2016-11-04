# This is a script which predicts constraints on the parameters of an intrinsic alignment model, using multiple shape measurement methods.

import numpy as np
import scipy
import scipy.integrate
import matplotlib.pyplot as plt

########## FUNCTIONS ##########

def setup_rp_bins(rmin, rmax, nbins):
	""" Sets up the bins of projected radius """

	# These are the *edges* of the bins.
	bins = scipy.logspace(np.log10(rmin), np.log10(rmax), nbins+1)

	return bins

def get_NofZ(a, b, z_med, z_min, z_max, zpts):
	""" Returns the dNdz of the sources as a function of photometric redshift, as well as the z points at which it is evaluated."""

	z = scipy.linspace(z_min, z_max, zpts)
	nofz_ = z ** a * np.exp( - (z / (z_med / np.sqrt(2.))) ** b )
	
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

def get_N_ls(z_1, z_2, n_l_tot, n_s_tot, nofz_1, z_v):
	""" Gets the number of lens/source pairs *total* in a sample between z_1 and z_2, for dNdz given by a normalized nofz_1 computed at redshifts z_v"""

	# Get the index of the z vector closest to the limits of the integral
	i_z1 = next(j[0] for j in enumerate(z_v) if j[1]>=z_1)
	i_z2 = next(j[0] for j in enumerate(z_v) if j[1]>=z_2)
	
	frac = scipy.integrate.simps(nofz_1[i_z1:i_z2], z_v[i_z1:i_z2])

	N_ls = frac*n_s_tot*n_l_tot

	return N_ls 

def get_perbin_N_ls(rp_bins_1, N_ls_1):
	""" Gets the number of lens/source pairs relevant to each bin of projected radius """

	# This is a dummy prescription for now.

	N_ls_pbin = np.ones(len(rp_bins_1)-1) * N_ls_1 / (len(rp_bins_1)-1)

	return N_ls_pbin

def setup_shapenoise_cov(e_rms, N_ls_pbin):
	""" Returns a diagonal covariance matrix in bins of projected radius for a measurement dominated by shape noise. Elements are e_{rms}^2 / (N_ls) where N_ls is the number of l/s pairs relevant to each projected radius bin.""" 

	cov = np.diag(e_rms**2 / N_ls_pbin)
	
	return cov

def subtract_var(var_1, var_2, covar):
	""" Takes the variance of two non-independent  Gaussian random variables, and their covariance, and returns the variance of their difference."""
	
	var_diff = var_1 + var_2 - 2 * covar

	return var_diff

def get_gammaIA_cov(Cov_1, Cov_2, covar):
	""" Takes the covariance matrices of the constituent elements of gamma_{IA} and combines them to get the covariance matrix of gamma_{IA} in projected radial bins."""

	gammaIA_cov = np.diag(np.zeros(len(np.diag(Cov_1))))

	for i in range(0,len(np.diag(Cov_1))):	
		gammaIA_cov[i,i] = subtract_var(Cov_1[i,i], Cov_2[i,i], covar[i])

	return gammaIA_cov

def get_fid_gIA(rp_bins_1):
	""" This function computes the fiducial value of gamma_IA in each projected radial bin."""

	# This is a dummy function for now.

	fidvals = np.zeros(len(rp_bins_1)-1)

	return fidvals

def plot_variance(cov_1, fidvalues_1, rp_edges, filename):
	""" Takes a covariance matrix, a vector of the fiducial values of the object in question, and the edges of the projected radial bins, and makes a plot showing the fiducial values and 1-sigma error bars from the diagonal of the covariance matrix. Outputs this plot to location 'filename'."""

	# Get the midpoints of the projected radius bins
	logedges=np.log10(rp_edges)
	bin_centers=np.zeros(len(rp_edges)-1)
        for ri in range(0,len(rp_edges)-1):
                bin_centers[ri]    =       10**((logedges[ri+1] - logedges[ri])/2. +logedges[ri])

	print "bincenters=", bin_centers

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


######## MAIN CALLS ##########

# Import the parameter file:
import IA_params_shapes as pa

# Set up projected bins
rp_bins 	= 	setup_rp_bins(pa.rp_min, pa.rp_max, pa.N_bins)

# Set up the redshift distribution of sources
(z, dNdz)	=	get_NofZ(pa.alpha, pa.beta, pa.zm, pa.zmin, pa.zmax, pa.zpts)

# Get the total number of lens source pairs for the source sample
N_ls_total 	= 	get_N_ls(pa.zS_min, pa.zS_max, pa.N_l, pa.N_s, dNdz, z)

# Get the number of lens source pairs for each source sample in projected radial bins
N_ls_pbin	=	get_perbin_N_ls(rp_bins, N_ls_total)

# Get the covariance matrix in projected radial bins of Delta Sigma for samples a and b
Cov_a		=	setup_shapenoise_cov(pa.e_rms_a, N_ls_pbin)
Cov_b		=	setup_shapenoise_cov(pa.e_rms_b, N_ls_pbin)

# Combine the constituent covariance matrices to get the covariance matrix for gamma_IA in projected radial bins
Cov_gIA		= 	get_gammaIA_cov(Cov_a, Cov_b, pa.covar_DSig)

# Get the fiducial value of gamma_IA in each projected radial bin
fid_gIA		=	get_fid_gIA(rp_bins)

# Output a plot showing the 1-sigma error bars on gamma_IA in projected radial bins
plot_variance(Cov_gIA, fid_gIA, rp_bins, pa.plotfile)
