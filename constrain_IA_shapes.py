# This is a script which constrains the (amplitude? parameters of a model?) of intrinsic alignments, using multiple shape measurement methods.

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

def com(z_, H_, OmC_, OmB_, OmR_, OmN_):
        """ Gets the comoving distance in units of Mpc/h at a given redshift. """

        OmL_ = 1. - OmC_ - OmB_ - OmR_ - OmN_

        def chi_int(z):
                return 1. / (H_ * ( (OmC_+OmB_)*(1+z)**3 + OmL_ + (OmR_+OmN_) * (1+z)**4 )**(0.5))

        chi = scipy.integrate.quad(chi_int, 0, z_)[0]

        return chi

def get_areas(bins, z_eff):
        """ Gets the area of each projected radial bin, in square arcminutes. z_eff = effective lens redshift. """

        # Areas in units (Mpc/h)^2
        areas_mpch = np.zeros(len(bins)-1)
        for i in range(0, len(bins)-1):
                areas_mpch[i] =  np.pi * (bins[i+1]**2 - bins[i]**2)

        #Comoving distance out to effective lens redshift in Mpc/h
        chi_eff = com(z_eff, pa.H0, pa.OmC, pa.OmB, pa.OmR, pa.OmN)

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

def setup_shapenoise_cov(e_rms, N_ls_pbin):
	""" Returns a diagonal covariance matrix in bins of projected radius for a measurement dominated by shape noise. Elements are e_{rms}^2 / (N_ls) where N_ls is the number of l/s pairs relevant to each projected radius bin.""" 

	cov = np.diag(e_rms**2 / N_ls_pbin)

	#print "cov=", cov
	
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
(z, dNdz)	=	get_NofZ(pa.alpha,pa.zs, pa.zmin, pa.zmax, pa.zpts)

# Get the fraction of dndZ which is in the source sample
frac		=       get_z_frac(pa.zS_min, pa.zS_max, dNdz, z)

# Get the number of lens source pairs for each source sample in projected radial bins
N_ls_pbin	=	get_perbin_N_ls(rp_bins, pa.zeff, frac, pa.n_s, pa.n_l, pa.Area)

# Get the covariance matrix in projected radial bins of Delta Sigma for samples a and b
Cov_a		=	setup_shapenoise_cov(pa.e_rms_a, N_ls_pbin)
Cov_b		=	setup_shapenoise_cov(pa.e_rms_b, N_ls_pbin)

# Combine the constituent covariance matrices to get the covariance matrix for gamma_IA in projected radial bins
Cov_gIA		= 	get_gammaIA_cov(Cov_a, Cov_b, pa.covar_DSig)

# Get the fiducial value of gamma_IA in each projected radial bin
fid_gIA		=	get_fid_gIA(rp_bins)

# Output a plot showing the 1-sigma error bars on gamma_IA in projected radial bins
plot_variance(Cov_gIA, fid_gIA, rp_bins, pa.plotfile)
