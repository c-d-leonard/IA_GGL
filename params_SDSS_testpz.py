# Parameters for both shape-measurement and Blazek et al. method of estimating IA. 

import numpy as np

survey			=	'SDSS'

N_bins	=	7 # The number of bins of projected radius 

bd = 2.2 # Calculated using Reid & Spergel 2008 HOD, see HOD bias_check.ipynb

e_rms 			= 	0.21
e_rms_Bl_a 		= 	e_rms # rms ellipticity of sample a, Blazek method   # from Reyes et al. 2012 (reports in terms of distortion = 0.36)
e_rms_Bl_b		=	e_rms # rms ellipticity of sample b, Blazek method
e_rms_Bl_full	=	e_rms # rms ellipticity of the full sample, Blazek method
e_rms_a 		= 	e_rms # rms ellipticity of sample measured with method a, shapes method  
e_rms_b 		= 	e_rms # rms ellipticity of sample measured with method b, shapes method
S_to_N 			= 	15. # The per-galaxy signal to noise- necessary for estimating sigma_e

#Parameters of the dNdz of sources, if using an analytic distribution. 'fid' = the fiducial value, 'sys' = 1sigma off from fid value to evaluate systematic error.
dNdztype	=	'Nakajima'
alpha_fid 	= 	2.338
zs_fid 		=	0.303
dNdzpar_fid	=	[alpha_fid, zs_fid] #Put these in a list to facilitate passing around 

percent_change = -10
alpha_true	=	alpha_fid*(1.+percent_change/100.)
zs_true 	=	zs_fid*(1.+percent_change/100.)
dNdzpar_true	=	[alpha_true, zs_true]


pztype		=	'Gaussian'
sigz_fid	=	0.11 # The photometric redshift error given a Gaussian photo_z model
pzpar_fid 	=	[sigz_fid] # Make this a list to make it more generic to pass around

# Parameters related to the spec and photo z's of the source sample and other redshift cuts.
zeff 	= 	0.28  # See table 1, Kazin 2010, 0908.2598, DR7-Dim sample. 
zLmin	= 	0.16  # See Kazin 2010
zLmax	=	0.36  # See Kazin 2010. 
dNdzL_file	=  'SDSS_LRG_DR7dim_nofz.txt'

close_cut = 100.# Mpc/h # The maximum separation from a lens to consider part of `rand-close', in Mpc/h
#Blazek et al. case
zsmin 	=	0.02
zsmax 	= 	3.0
zphmin	=	0.
zphmax	=	5.0
delta_z	=	0.17 # The width of the redshift slice which begins at the lens and ends at the top of sample a
#zmin_dndz = zsmin
#zmax_dndz = 1.25 # The effective point at which the spectroscopic dNdz goes to 0 (input by looking at dNdz for now).
# Shape measurment case
zeff 	= 	0.28   # The effective redshift of the lens sample
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

# Cosmological parameters. Planck 2015 results XIII: cosmological parameters. Table 1, column 6.
Nnu	=	3.046    # Massless neutrinos
HH0 =   67.26  
OmR	=	2.47*10**(-5)/(HH0/100.)**2
OmN	=	Nnu*(7./8.)*(4./11.)**(4./3.)*OmR
OmB	=	0.02222/(HH0/100.)**2 
OmC	=	0.1199/(HH0/100.)**2 
H0	=	10**(5)/c
A_s	=	2.2 * 10**(-9)
sigma8 = 0.84
n_s	=	0.9652

cos_par_std = [HH0, OmC, OmB, sigma8]

# We also have two other sets of cosmological parameters:
# one each for the lens and source sample to match the parameters
# simultaneously fit or fixed in sims when fitting HOD parameters:
# (From HOD papers)
HH0_l = 72.0
OmC_l = 0.216
OmB_l = 0.044
sigma8_l = 0.77
n_s_l = 0.95

cos_par_l = [HH0_l, OmC_l, OmB_l, sigma8_l]

HH0_s = 72.0
OmC_s = 0.216
OmB_s = 0.044
sigma8_s = 0.77
n_s_s = 0.95

cos_par_s = [HH0_s, OmC_s, OmB_s, sigma8_s]

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
# The q_ij parameters are taken directly from this table. a_h is computed by taking the power law of a_h as a function of luminosity a_h(L) from Singh et al. 2014 and integrating it over the Schechter luminosity function from Krause et al. 2015 for the limiting magnitude of SDSS shapes, r<22. (see ./ah_Ai_calculation_zLext.ipynb)
q11 = 0.005    
q12 = 5.909
q13 = 0.3798
q21 = 0.6    
q22 = 1.087
q23 = 0.6655
q31 = 3.1    
q32 = 0.1912
q33 = 0.4368

#Parameters required for computing the luminosity function (from Loveday 2012 / Krause et al. 2015 unless otherwise noted)
mlim = 22.0 # See Figure 3 of Reyes et al. 2012.
Mp = -22. # From Singh et al. 2014 (but this is kind of an arbitrary choice)

Mr_s_red = -20.34
#Q_red = 1.8 # GAMA
Q_red = 1.20 # Deep 2 (Krause et al. 2015, scaled from Faber et al. 2007)
alpha_lum_red = -0.57
phi_0_red = 0.011
#P_red = -1.2
P_red = -1.15 # Deep 2

Mr_s_all = -20.70
#Q_all 	= 0.7 # GAMA
Q_all = 1.23 # Deep 2
alpha_lum_all = -1.23
phi_0_all = 0.0094
#P_all = 1.8
P_all = -0.3 # Deep 2
lumparams_red = [Mr_s_red, Q_red, alpha_lum_red, phi_0_red, P_red]
lumparams_all = [Mr_s_all, Q_all, alpha_lum_all, phi_0_all, P_all]
