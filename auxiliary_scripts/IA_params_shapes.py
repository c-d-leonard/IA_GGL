# This is an input parameter file for use in calculating forecast constraints on intrinsic alignments using two shape-measurement methods.

import numpy as np

# Survey properties: 
e_rms_a 		= 	0.29 # The rms ellipticity under measurement method a.
e_rms_b			=	0.31 # The rms ellipticity under measurement method b.
n_l     		=   8.7 # The number of lenses in the lens sample per square DEGREE
Area    		=   7131 # Area associated with the lens sample in square DEGREES
n_s     		=   1.2 # The number density of sources in the sample per square ARCMINUTE
a_con			=	1./1.8	# Fiducial constant offset, approximated from Singh 2016 assuming unprimed method isophotal and primed ReGaussianisation
cov_perc 		= 	0.7 #percentage covariance between methods
S_to_N 			= 	15. # The signal to noise - necessary for estimating sigma_e
e_rms_mean 		=	np.abs((e_rms_b+e_rms_a)/2.) # This is the e_rms used in computing the shared weight

# The factor by which the boost is proportional to the projected correlation function, for different source samples.
boost_assoc 	= 	0.2 # Boost at 1 Mpc/h for our associated sample
boost_tot 		= 	0.06 # Boost at 1 Mpc/h for all the source-lens pairs in the survey (all z)

# Files to import error on the boost
sigBF_a = './txtfiles/boost_error_from_rachel_assoc.txt' # File containing two columns: rp (kpc/h), sigma(Boost-1) for sample a

# Quantities related to how we split up the `data'
rp_max 	=	50. # The maximum projected radius (Mpc/h)
rp_min	=	0.05 # The minimum projected radius (Mpc/h)
N_bins	=	15  # The number of bins of projected radius 

#Parameters of the dNdz of sources, if using an analytic distribution. 'fid' = the fiducial value, 'sys' = 1sigma off from fid value to evaluate systematic error.
alpha_fid 	= 	2.338
zs_fid 		=	0.303
alpha_sys 	=	2.738 # Derived from Nakajima et al 2011 on April 4 2017
zs_sys		= 	0.323 # Derived from Nakajima et al 2011 on April 4 2017
zpts		=	1000  # Number of points in the z vector at which we are evaluating dNdz
sigz_fid	=	0.11  # The photometric redshift error given a Gaussian photo_z model
sigz_sys 	= 	0.13  #Derived from Nakajima et al 2011 on April 4 2017

# Parameters related to the spec and photo z's of the source sample and other redshift cuts.
zpts	=	10000  # Number of points in the z vector at which we are evaluating dNdz
zeff 	= 	0.28   # The effective redshift of the lens sample
zmin 	=	0.0 # Minimum spec-z
zmax 	= 	3.0 # Maximum spec-z
zmin_ph	=	0.0 # Minimu photo-z
zmax_ph	=	5.0 # Maximum photo-z
close_cut = 100 # The separation in Mpc/h within which we consider source galaxies could be subject to intrinsic alingments

# Constants / conversions
mperMpc = 3.0856776*10**22
Msun = 1.989*10**30 # in kg
Gnewt = 6.67408*10**(-11)

# Cosmological parameters and constants
c		=		2.99792458*10**(8) # Speed of light in units of m/s
Nnu		=		3.046    # Massless neutrinos
HH0 	= 		70. #67.26 #72.
OmR		=		2.47*10**(-5)/(HH0/100.)**2
OmN		=		Nnu*(7./8.)*(4./11.)**(4./3.)*OmR
OmB 	= 		0.046 #0.02222/(HH0/100.)**2
OmC 	= 		0.236 #0.1199/(HH0/100.)**2
OmM 	= 		OmB + OmC
H0		=		10**(5)/c

# Parameters for getting the fiducial gamma_IA
# 2 halo term parmaeters 
bs = 1.77
bd = 1.77
Ai = 5.0
C1rho = 0.0134
sigz_gwin = 0.001
kpts_wgg = 10000
kpts_wgp = 2000
# 1 halo gal-gal term parameters
Mvir = 10**(13.18) 
ng = 3. * 10**(-4) # volume density of galaxies for BOSS, in h^3 / Mpc^3
fsat = 0.14 #0.0636 # Satelite fraction from Reid & Spergel 2008.
# 1 halo IA term parameters, from Singh et al. 2014 Table 1
q11 = 0.005   #0.02056  
q12 = 5.909
q13 = 0.3798
q21 = 0.6    # 1.978  
q22 = 1.087
q23 = 0.6655
q31 = 3.1    #4.154  
q32 = 0.1912
q33 = 0.4368
ah =  1.

# Other
#sig_sys_dNdz = np.asarray([ 0.0014777 ,  0.0037014 ,  0.00917435,  0.02051145,  0.03701631, 0.05260886,  0.06284505]) # systematic error from dNdz in Ncorr. Estimated from varying alpha and zs, with sigz=0.08 and Boost(1Mpc/h) = 1.2
#sig_sys_dp = np.asarray( [ 0.00654311,  0.01624936,  0.0396385 ,  0.08656573,  0.15238458, 0.21256654,  0.2512192 ] ) # systematic error from p(z) in Ncorr
# The parameter labels to be constrainted in the Fisher forecast (here for a power-law model for IA)
#A       		=       0
#beta    		=       1
#par     		=       [A, beta]
#A_fid   		=       0.059
#beta_fid        =       -0.73
 
