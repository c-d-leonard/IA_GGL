# Parameters for both shape-measurement and Blazek et al. method of estimating IA. 

import numpy as np

run_quants 		=	False # For the Blazek case, whether you want to recompute F, cz, and SigIA
survey			=	'SDSS'

# Parameters associated with the sample / shape noise calcuation 
e_rms_Bl_a 		= 	0.3 # rms ellipticity of sample a, Blazek method  # I JUST MADE THIS UP AT SOME POINT, NEED TO RE-RUN WITH THE PROPER VALUE! 
e_rms_Bl_b		=	0.3 # rms ellipticity of sample b, Blazek method
e_rms_a 		= 	0.29 # rms ellipticity of sample measured with method a, shapes method  # I THINK I SHOULD ACTUALLY BE RUNNING WITH THESE THE SAME FOR BOTH METHODS?
e_rms_b 		= 	0.31 # rms ellipticity of sample measured with method b, shapes method
n_l 			= 	8.7 # The number of lenses in the lens sample per square DEGREE
Area_l 			=	7131 # Area associated with the lens sample in square DEGREES
fsky			=   Area_l / 41253. # Assumes the lens area is the limiting factor
n_s 			=	1. # The EFFECTIVE number density of sources in the sample per square ARCMINUTE - 1.2 is unweighted. This number is from Rachel in an email June 7.
S_to_N 			= 	15. # The signal to noise of the lensing measurement (?) - necessary for estimating sigma_e
a_con			=	[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #[1./1.25, 1./1.5, 1./1.75]	# Fiducial constant offset, approximated from Singh 2016 assuming unprimed method isophotal and primed ReGaussianisation
cov_perc 		= 	[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #percentage covariance between methods, shape measurement method
e_rms_mean 		=	np.abs((e_rms_b+e_rms_a)/2.) # This is the e_rms used in computing the shared weight for shapes methods
N_shapes		= 	Area_l * n_s * 3600 # Number of galaxies in the shape sample.
#N_LRG			=	62081 # Numbr of galaxies in the LRG sample.
 
# The factor by which the (boost-1) is proportional to the projected correlation function at 1 Mpc/h, for different source samples.
# THESE DEPEND ON THE SAMPLE VIA THE P(Z_S, Z_P) PROPERTIES.
# Currently set for SDSS LRGS / shapes.
boost_assoc = 0.2 
boost_far = 0.03
boost_close = 0.1

# Parameters associated with the projected radial bins
rp_max 	=	20.0 # The maximum projected radius (Mpc/h)
rp_min	=	0.05 # The minimum projected radius (Mpc/h)
N_bins	=	7 # The number of bins of projected radius 

#Parameters of the dNdz of sources, if using an analytic distribution. 'fid' = the fiducial value, 'sys' = 1sigma off from fid value to evaluate systematic error.
dNdztype	=	'Nakajima'
alpha_fid 	= 	2.338
zs_fid 		=	0.303
dNdzpar_fid	=	[alpha_fid, zs_fid] #Put these in a list to facilitate passing around
zpts		=	1000  # Number of points in the z vector at which we are evaluating dNdz

pztype		=	'Gaussian'
sigz_fid	=	0.11  # The photometric redshift error given a Gaussian photo_z model
pzpar_fid 	=	[sigz_fid] # Make this a list to make it more generic to pass around
boost_sys	=	1.03 # Multiplier for the boost systematic error. This value is given in Blazek et al. 2012.

# Parameters related to the spec and photo z's of the source sample and other redshift cuts.
zeff 	= 	0.28  # The effective redshift of the lens sample
close_cut = 100 # Mpc/h # The maximum separation from a lens to consider part of `rand-close', in Mpc/h
#Blazek et al. case
zsmin 	=	0.0
zsmax 	= 	3.0
zphmin	=	0.
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

# Fractional errors on 'fudge factors' we are using to get a handle on the relative importance of different systematic errors.
fudge_frac_level = np.logspace(-2, 0, 15)
#fudge_frac_level = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8]
fudge_Ncorr = 	0. #[0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8]
fudge_czA	= 	0.
fudge_czB	=	0.
fudge_sigA	=	0.
fudge_sigB	=	0.
fudge_FA	=	0.
fudge_FB	=	0.

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
kpts_wgg = 10000
kpts_wgp = 2000
bs_Bl = 1. # SDSS shape sample - "garden variety" galaxies.
bs_shapes = 1. # SDSS shape sample - "garden variety" galaxies.
bd_Bl = 2.07 #1.77 # 
bd_shapes = 2.07 #1.77
Ai_Bl = 0.47 # assumes SDSS shapes have mean abs mag -19.5
Ai_shapes = 0.47
#Ai_Bl = 0.25 # SDSS shape sample has luminosity ~ SDSS L4 aka M_r ~ [-20 -> -19], aka much fainter than LOWZ. This Ai computed by using this M_r in eqn 34 of Singh 2014 (pwr law A_i(L_r)).
#Ai_shapes = 0.25  # Is this power law still valid for such dimmer galaxies? Not entirely sure.
C1rho = 0.0134

# 1 halo gal-gal term parameters
ng_Bl =  10**(-4) # SDSS LRG value #3. * 10**(-4) # volume density of galaxies for BOSS, in h^3 / Mpc^3.
fsat_LRG = 0.0636 # Satelite fraction from Reid & Spergel 2008 # 0.14 approximate boss lowz val . # WE DON'T USE THIS ANYMORE SHOULD REMOVE PROBABLY. 

##### Parameters of the HOD model, taken from Zu & Mandelbaum 2015, 1505.02781.  #####
# Ncen params: these refer to the central galaxies for SDSS MGS, which I think is close enough to SDSS LRGs to be okay to map to our case:
sigMs = 0.50
eta = -0.04
M1 = 10**(12.10)

# Nsat params: these refer at the moment to satellite occupation for SDSS MGS galaxies, which are much brighter than the SDSS shapes sample we care about...
Bsat = 8.98
beta_sat =0.90
Bcut = 0.86
beta_cut = 0.41
alpha_sat = 1.00

# f_SHMR parameters: these are used in getting Ncen and Nsat. They are probably okay for Ncen but perhaps not for Nsat. We may need to have two sets of these parameters.
delta = 0.42
gamma= 1.21
Mso = 10**(10.31)
beta = 0.33

# Here are the parameters from the HOD from Reid and Spergel 2008. It's less complex but it's specifically for the SDSS LRGs
Mcut_reid = 5.0 * 10**13 # Msol
M1_reid = 4.95 * 10**14 # Msol
alpha_reid = 1.035
Mmin_reid = 8.05*10**13 # Msol
sigLogM_reid = 0.7 

# 1-halo IA term parameters.
# These are for the model given by Schneider & Bridle 2010, 0903.3870, but for the parameters given in Table 1 of Singh et al. 2014.
# The q_ij parameters are taken directly from this table. a_h is computed by taking the power law of a_h as a function of luminosity a_h(L) from Singh et al. 2014 and integrating it over the Schechter luminosity function from Krause et al. 2015 for the limiting magnitude of SDSS shapes, r<21.8, and the Nakajima redshift distribution. (see ./secondary_scripts/ah_calculation.ipynb)
#ah_Bl  =  0.01 
ah_Bl = 6.4*10**(-4) # Stopgap, assumes M abs avg = -19.5
q11_Bl = 0.005    
q12_Bl  = 5.909
q13_Bl  = 0.3798
q21_Bl  = 0.6    
q22_Bl  = 1.087
q23_Bl  = 0.6655
q31_Bl  = 3.1    
q32_Bl  = 0.1912
q33_Bl  = 0.4368
ah_shapes =  0.01
q11_shapes = 0.005  
q12_shapes = 5.909
q13_shapes = 0.3798
q21_shapes = 0.6    
q22_shapes = 1.087
q23_shapes = 0.6655
q31_shapes = 3.1    
q32_shapes = 0.1912
q33_shapes = 0.4368



