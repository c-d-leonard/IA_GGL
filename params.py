# Parameters for both shape-measurement and Blazek et al. method of estimating IA. 

import numpy as np

# Parameters associated with the sample / shape noise calcuation 
e_rms_Bl_a 		= 	0.3 # rms ellipticity of sample a, Blazek method
e_rms_Bl_b		=	0.3 # rms ellipticity of sample b, Blazek method
e_rms_a 		= 	0.29 # rms ellipticity of sample measured with method a, shapes method
e_rms_b 		= 	0.31 # rms ellipticity of sample measured with method b, shapes method
n_l 			= 	8.7 # The number of lenses in the lens sample per square DEGREE
Area_l 			=	7131 # Area associated with the lens sample in square DEGREES
fsky			=   Area_l / 41253. # Assumes the lens area is the limiting factor
n_s 			=	1.2 # The number density of sources in the sample per square ARCMINUTE
S_to_N 			= 	15. # The signal to noise of the lensing measurement (?) - necessary for estimating sigma_e
a_con			=	[1./1.25, 1./1.5, 1./1.75]	# Fiducial constant offset, approximated from Singh 2016 assuming unprimed method isophotal and primed ReGaussianisation
cov_perc 		= 	[0.4, 0.6, 0.8] #percentage covariance between methods, shape measurement method
e_rms_mean 		=	np.abs((e_rms_b+e_rms_a)/2.) # This is the e_rms used in computing the shared weight for shapes methods
N_shapes		= 	3.0*10**7 # Number of galaxies in the shape sample.
N_LRG			=	62081 # Numbr of galaxies in the LRG sample.
 
# The factor by which the boost is proportional to the projected correlation function, for different source samples.
boost_assoc = 0.2 # Boost at 1 Mpc/h for our associated sample
boost_tot = 0.06 # Boost at 1 Mpc/h for all the source-lens pairs in the survey (all z)
boost_far = 0.03
boost_close = 0.1

# Files to import error on the boost
sigBF_a = './txtfiles/boost_error_from_rachel_assoc.txt' # File containing two columns: rp (kpc/h), sigma(Boost-1) for sample a
sigBF_b ='./txtfiles/boost_error_from_rachel_background.txt' # Same for b

# Parameters associated with the projected radial bins
rp_max 	=	20.0 # The maximum projected radius (Mpc/h)
rp_min	=	0.05 # The minimum projected radius (Mpc/h)
N_bins	=	7 # The number of bins of projected radius 

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
close_cut = 100 # Mpc/h # The maximum separation from a lens to consider part of `rand-close', in Mpc/h
#Blazek et al. case
zsmin 	=	0.0
zsmax 	= 	3.0
zphmin	=	zeff
zphmax	=	5.0
delta_z	=	0.17  # The width of the redshift slice which begins at the lens and ends at the top of sample a
zmin_dndz = zsmin
zmax_dndz = 1.25 # The effective point at which the spectroscopic dNdz goes to 0 (input by looking at dNdz for now).
# Shape measurment case
zpts	=	10000  # Number of points in the z vector at which we are evaluating dNdz
zeff 	= 	0.28   # The effective redshift of the lens sample
zmin 	=	0.0 # Minimum spec-z
zmax 	= 	3.0 # Maximum spec-z
zmin_ph	=	0.0 # Minimu photo-z
zmax_ph	=	5.0 # Maximum photo-z

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
bs_Bl = 1. # SDSS shape sample - "garden variety" galaxies.
bs_shapes = 1. # SDSS shape sample - "garden variety" galaxies.
bd_Bl = 2.07 #1.77
bd_shapes = 2.07 #1.77
Ai_Bl = 3.0 # CURRENTLY USING BOSS LOWZ VALUE scaled ad hoc for luminosity.
Ai_shapes = 3.0  # CURRENTLY USING BOSS LOWZ VALUE scaled ad hoc for luminosity.
C1rho = 0.0134

# 1 halo gal-gal term parameters
Mvir = 4.5 * 10**13 #heavily estimated SDSS LRG value from Reid & Spergel 2009. #10**(13.18) BOSS LOWZ value
ng_Bl =  10**(-4) # SDSS LRG value #3. * 10**(-4) # volume density of galaxies for BOSS, in h^3 / Mpc^3.
Mstar_src_low = 8.*10**9 * (HH0/100.) # Lower edge of stellar mass range in units of Msol / h^2
Mstar_src_high = 1.2*10**10 * (HH0/100.) # Upper edge of stellar mass range in units of Msol / h^2
fsat_LRG = 0.0636 # Satelite fraction from Reid & Spergel 2008 # 0.14 approximate boss lowz val .
# From Zu & Mandelbaum 2015, 1505.02781:
delta = 0.42
gamma= 1.21
Mso = 10**(10.31)
M1 = 10**(12.10)
beta = 0.33
Bsat = 8.98
beta_sat =0.90
alpha_sat = 1.00
Bcut = 0.86
beta_cut = 0.41
eta = -0.04
sigMs = 0.50



# 1 halo IA term parameters, from Singh et al. 2014 Table 1
q11_Bl = 0.02056  #0.005 LOWZ   
q12_Bl  = 5.909
q13_Bl  = 0.3798
q21_Bl  = 1.978  #0.6    
q22_Bl  = 1.087
q23_Bl  = 0.6655
q31_Bl  = 4.154  #3.1    
q32_Bl  = 0.1912
q33_Bl  = 0.4368
ah_Bl  =  0.05 # BOSS LOWZ value scaled in an adhoc was for luminosity (April 11 2017)

q11_shapes = 0.02056  #0.005 LOWZ   
q12_shapes = 5.909
q13_shapes = 0.3798
q21_shapes = 1.978  #0.6    
q22_shapes = 1.087
q23_shapes = 0.6655
q31_shapes = 4.154  #3.1    
q32_shapes = 0.1912
q33_shapes = 0.4368
ah_shapes =  0.05


