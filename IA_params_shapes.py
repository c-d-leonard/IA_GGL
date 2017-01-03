# This is an input parameter file for use in calculating forecast constraints on intrinsic alignments using two shape-measurement methods.

import numpy as np

# Cosmological parameters and constants (required to convert between z and comoving distance)
c		=		2.99792458*10**(8) # Speed of light in units of m/s
Nnu     =       3.046    # Massless neutrinos
HH0     =       67.26
OmR     =       2.47*10**(-5)/(HH0/100.)**2
OmN     =       Nnu*(7./8.)*(4./11.)**(4./3.)*OmR
OmB     =       0.02222/(HH0/100.)**2
OmC     =       0.1199/(HH0/100.)**2
H0      =       10**(5)/c

# Survey properties: 
e_rms_a = 	0.32 # The rms ellipticity under measurement method a.
e_rms_b	=	0.35 # The rms ellipticity under measurement method b.
n_l     =   8.7 # The number of lenses in the lens sample per square DEGREE
Area    =   7131 # Area associated with the lens sample in square DEGREES
n_s     =   1.2 # The number density of sources in the sample per square ARCMINUTE
a_con	=	1./1.4	# Fiducial constant offset, approximated from Singh 2016 assuming unprimed method isophotal and primed ReGaussianisation

#Parameters of the dNdz of sources, if using an analytic distribution.
alpha 	= 	2.338 
zs 		= 	0.303 
zpts	=	10000  # Number of points in the z vector at which we are evaluating dNdz
zeff 	= 	0.32   # The effective redshift of the lens sample

#Quantities necessary to get the weights:
S_to_N = 15. # The signal to noise - necessary for estimating sigma_e
e_rms_mean 	=	np.abs((e_rms_b+e_rms_a)/2.) # This is the e_rms used in computing the shared weights.

# Quantities related to how we split up the `data'
rp_max 	=	100.0 # The maximum projected radius (Mpc/h)
rp_min	=	0.01 # The minimum projected radius (Mpc/h)
N_bins	=	20.  # The number of bins of projected radius 
zmin 	=	0.0 # Minimum spec-z
zmax 	= 	3.0 # Maximum spec-z
zmin_ph	=	0.0 # Minimu photo-z
zmax_ph	=	5.0 # Maximum photo-z
close_cut = 100 # The separation in Mpc/h within which we consider source galaxies could be subject to intrinsic alingments

# The parameter labels to be constrainted in the Fisher forecast (here for a power-law model for IA)
A       		=       0
beta    		=       1
par     		=       [A, beta]
A_fid   		=       0.059
beta_fid        =       -0.73

# Other
cov_perc = 0.6 #percentage covariance between methods
sig_sys_dNdz = np.asarray([ 0.0014777 ,  0.0037014 ,  0.00917435,  0.02051145,  0.03701631, 0.05260886,  0.06284505]) # systematic error from dNdz in Ncorr. Estimated from varying alpha and zs, with sigz=0.08 and Boost(1Mpc/h) = 1.2
sig_sys_dp = np.asarray( [ 0.00654311,  0.01624936,  0.0396385 ,  0.08656573,  0.15238458, 0.21256654,  0.2512192 ] ) # systematic error from p(z) in Ncorr
sigz=0.09 # Uncertainty on photo z (Gaussian model, sigma = sigz*(1+z))
plotfile =	'./test_shapes_sys10%_sigz0.08_log.pdf'  # Location of file for plot showing 1 sigma error bars on gamma_IA
boost_samp = 1.2 # Boost at 1 Mpc/h for our associated sample
boost_tot = 1.04 # Boost at 1 Mpc/h for all the source-lens pairs in the survey (all z)
sigB	= 0.0 # The statistical error from the boost in Ncorr.
ProjMax = 100.

# The edges of the source redshift bin to be considered:
#zS_min 	= 	0.32
#zS_max 	= 	0.49

# LA model parameters
bs = 1.77
bd = 1.77
Ai = 5.0
C1rho = 0.0134

# 1 halo IA term parameters
q11 = 0.005 #0.02056    
q12 = 5.909
q13 = 0.3798
q21 = 0.6 #1.978     
q22 = 1.087
q23 = 0.6655
q31 = 3.1 #4.154     
q32 = 0.1912
q33 = 0.4368
ah =  1. # 0.08


