# This is an input parameter file for use in calculating forecast constraints on intrinsic alignments using the Blazek et al 2012 formalism.

import numpy as np

# Parameters associated with the sample / shape noise calcuation 
e_rms_Bl_a = 	0.3 # rms ellipticity of sample a
e_rms_Bl_b	=	0.3 # rms ellipticity of sample b
n_l 	= 	8.7 # The number of lenses in the lens sample per square DEGREE
Area_l 	=	7131 # Area associated with the lens sample in square DEGREES
n_s 	=	1.2 # The number density of sources in the sample per square ARCMINUTE
S_to_N = 15. # The signal to noise of the lensing measurement (?) - necessary for estimating sigma_e
 
# The factor by which the boost is proportional to the projected correlation function, for different source samples.
boost_assoc = 0.2 # Boost at 1 Mpc/h for our associated sample
boost_tot = 0.06 # Boost at 1 Mpc/h for all the source-lens pairs in the survey (all z)
boost_far = 0.03
boost_close = 0.1

# Files to import error on the boost
sigBF_a = './txtfiles/boost_error_from_rachel_assoc.txt' # File containing two columns: rp (kpc/h), sigma(Boost-1) for sample a
sigBF_b ='./txtfiles/boost_error_from_rachel_background.txt' # Same for b

# Parameters associated with the projected radial bins
rp_max 	=	50.0 # The maximum projected radius (Mpc/h)
rp_min	=	0.05 # The minimum projected radius (Mpc/h)
N_bins	=	15 # The number of bins of projected radius 

#Parameters of the dNdz of sources, if using an analytic distribution. 'fid' = the fiducial value, 'sys' = 1sigma off from fid value to evaluate systematic error.
alpha_fid 	= 	2.338
zs_fid 		=	0.303
alpha_sys 	=	2.738 # Derived from Nakajima et al 2011 on April 4 2017
zs_sys		= 	0.323 # Derived from Nakajima et al 2011 on April 4 2017
zpts		=	1000  # Number of points in the z vector at which we are evaluating dNdz
sigz_fid	=	0.11  # The photometric redshift error given a Gaussian photo_z model
sigz_sys 	= 	0.13  #Derived from Nakajima et al 2011 on April 4 2017

# Parameters related to the spec and photo z's of the source sample and other redshift cuts.
zeff 	= 	0.28  # The effective redshift of the lens sample
zsmin 	=	0.0
zsmax 	= 	3.0
zphmin	=	zeff
zphmax	=	5.0
delta_z	=	0.17  # The width of the redshift slice which begins at the lens and ends at the top of sample a
close_cut = 100 # Mpc/h # The maximum separation from a lens to consider part of `rand-close', in Mpc/h

# Constants / conversions
mperMpc = 3.0856776*10**22
Msun = 1.989*10**30 # in kg
Gnewt = 6.67408*10**(-11)
c=2.99792458*10**(8)

# Cosmological parameters:
Nnu	=	3.046    # Massless neutrinos
HH0 = 70. #67.26 #72
OmR	=	2.47*10**(-5)/(HH0/100.)**2
OmN	=	Nnu*(7./8.)*(4./11.)**(4./3.)*OmR
OmB	=	0.05 #0.02222/(HH0/100.)**2 #0.046
OmC	=	0.2 #0.1199/(HH0/100.)**2 #0.236
OmM=  OmB+OmC
H0	=	10**(5)/c

# Parameters for getting the fiducial gamma_IA
# 2 halo term parmaeters
sigz_gwin = 0.0001 
kpts_wgg = 5000 #10000
kpts_wgp = 2000
bs = 2.07 #1.77
bd = 2.07 #1.77
Ai = 5.0  # CURRENTLY USING BOSS LOWZ VALUE.
C1rho = 0.0134
# 1 halo gal-gal term parameters
Mvir = 4.5 * 10**13 #heavily estimated SDSS LRG value from Reid & Spergel 2009. #10**(13.18) BOSS LOWZ value
ng =  10**(-4) # SDSS LRG value #3. * 10**(-4) # volume density of galaxies for BOSS, in h^3 / Mpc^3
#nh = 2*10**(-4) # volume density of halos. Must be chosen appropriately for Mvir above. This is a super rough approximation.
fsat = 0.0636 # Satelite fraction from Reid & Spergel 2008 # 0.14 approximate boss lowz val .
# 1 halo IA term parameters, from Singh et al. 2014 Table 1
q11 = 0.02056  #0.005 LOWZ   
q12 = 5.909
q13 = 0.3798
q21 = 1.978  #0.6    
q22 = 1.087
q23 = 0.6655
q31 = 4.154  #3.1    
q32 = 0.1912
q33 = 0.4368
ah =  1.






