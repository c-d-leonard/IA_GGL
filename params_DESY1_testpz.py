# Parameters for testing effect of getting dNdz wrong

# Things I need:
# Bias of lens sample
# Weights / parameters of weights as a function of source redshift 
# dNdz of sources or parameters 
# photo-z parameters (sigz) 
# effective redshift of lens sample
# Pimax
# deltaz
# z extent
# DES cosmological parameters
# HOD for lenses

import numpy as np

survey			=	'DESY1'

N_bins	=	10 # The number of bins of projected radius 

bd = 1.60 # Our lenses are bin 2 of the Y1 galaxy clustering sample. This value taken from Table IV of 1708.01536.
bs = 1.0 # I'm assuming that the source galaxy sample bias is 1.0.

# Parameters associated with the sample / shape noise calcuation 
e_rms 			= 	0.21
e_rms_Bl_a 		= 	e_rms # rms ellipticity of sample a, Blazek method   # from Reyes et al. 2012 (reports in terms of distortion = 0.36)
e_rms_Bl_b		=	e_rms # rms ellipticity of sample b, Blazek method
e_rms_Bl_full	=	e_rms # rms ellipticity of the full sample, Blazek method
e_rms_a 		= 	e_rms # rms ellipticity of sample measured with method a, shapes method  
e_rms_b 		= 	e_rms # rms ellipticity of sample measured with method b, shapes method
n_l 			= 	8.7 # The number of lenses in the lens sample per square DEGREE
Area_l 			=	7131 # Area associated with the lens sample in square DEGREES
fsky			=   Area_l / 41253. # Assumes the lens area is the limiting factor
#n_s 			=	1. # The EFFECTIVE number density of sources in the sample per square ARCMINUTE - 1.2 is unweighted. This number is from Rachel in an email June 7.
S_to_N 			= 	15. # The per-galaxy signal to noise- necessary for estimating sigma_e
a_con			=	[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # Fiducial constant offset, approximated from Singh 2016 assuming unprimed method isophotal and primed ReGaussianisation
cov_perc 		= 	[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #percentage covariance between methods, shape measurement method
e_rms_mean 		=	np.abs((e_rms_b+e_rms_a)/2.) # This is the e_rms used in computing the shared weight for shapes methods
#N_shapes		= 	Area_l * n_s * 3600. # Number of galaxies in the shape sample.

#Parameters of the dNdz of sources, if using an analytic distribution. 'fid' = the fiducial value, 'sys' = 1sigma off from fid value to evaluate systematic error.
# NEED DNDZ OR FUNCTIONAL FORM. Have now the dNdz for each bin wrt zmc and zmean. zmc corresponds roughly to 'true' redshift and zmean to 'photoz'.
# Need to change the code so I'm importing these and using them directly.
dNdztype	=	'Nakajima'
alpha_fid 	= 	2.338
zs_fid 		=	0.303
dNdzpar_fid	=	[alpha_fid, zs_fid] #Put these in a list to facilitate passing around 

percent_change = 5
alpha_true	=	alpha_fid*(1.+percent_change/100.)
zs_true 	=	zs_fid*(1.+percent_change/100.)
dNdzpar_true	=	[alpha_true, zs_true]

# NEED APPROPRIATE PARAMETERS FOR DES Y1
pztype		=	'Gaussian'
sigz_fid	=	0.11 # The photometric redshift error given a Gaussian photo_z model
pzpar_fid 	=	[sigz_fid] # Make this a list to make it more generic to pass around


# Parameters related to the spec and photo z's of the source sample and other redshift cuts.

# Lenses: Bin 2 of galaxy clustering sample from DES Y1
zeff 	= 	0.37  # Need to get this manually
zLmin	= 	0.301  
zLmax	=	0.448  
dNdzL_file	=  '/DESY1_quantities_fromSara/z_dNdz_lenses_subbin.dat'

# MAKE SURE I'M USING THE RIGHT VALUES HERE
close_cut = 100.# Mpc/h # The maximum separation from a lens to consider part of `rand-close', in Mpc/h. Waiting on Jonathan for this value.
#Blazek et al. case
zsmin 	=	0.02
zsmax 	= 	3.0
zphmin	=	0.
zphmax	=	5.0
delta_z	=	0.17 # The width of the redshift slice which begins at the lens and ends at the top of sample a
zmin_dndz = zsmin
zmax_dndz = 1.25 # The effective point at which the spectroscopic dNdz goes to 0 (input by looking at dNdz for now).
# Shape measurment case
zmin 	=	0.0 # Minimum spec-z
zmax 	= 	3.0 # Maximum spec-z
zmin_ph	=	0.0 # Minimu photo-z
zmax_ph	=	5.0 # Maximum photo-zboost
delta_z = 0.17 

# Constants / conversions
mperMpc = 3.0856776*10**22
Msun = 1.989*10**30 # in kg
Gnewt = 6.67408*10**(-11)
c=2.99792458*10**(8)

# Cosmological parameters. Using the same ones that Sara used as fiducial values in her analysis (as a default).
# Need to work out how to vary these for the shear-ratio test.
Nnu	=	3.046    # Massless neutrinos, standard value
HH0 	=   	72.0     # Default value Sara used
OmR	=	2.47*10**(-5)/(HH0/100.)**2
OmN	=	Nnu*(7./8.)*(4./11.)**(4./3.)*OmR
OmM	= 	0.3	 # Default value Sara used
OmB	=	0.05 
OmC	=	OmM-OmB 
H0	=	10**(5)/c
sigma8  =      0.817 	# From DES Y1
n_s	=	0.9652


# We also have another set of cosmological parameters:
# to match the parameters simultaneously fit or fixed in sims when fitting HOD parameters:
#HH0_l = 72.0
#OmC_l = 0.216
#OmB_l = 0.044
#sigma8_l = 0.77
#n_s_l = 0.95

# (From Reid & Spergel paper)
#HH0_l = 70.1
#OmB_l = 0.0462
#OmC_l = 0.2792-OmB_l
#sigma8_l = 0.817
#n_s_l = 0.960

# Here are the parameters from the HOD from Reid and Spergel 2008 (0811.1025). For SDSS LRG sample 0.16<z<0.36.
#Mcut_reid = 5.0 * 10**13 # Msol
#M1_reid = 4.95 * 10**14 # Msol
#alpha_reid = 1.035
#Mmin_reid = 8.05*10**13 # Msol
#sigLogM_reid = 0.7


