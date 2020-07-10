# Parameters for both shape-measurement and Blazek et al. method of estimating IA. 

import numpy as np

run_quants		=	True
survey 			=	'LSST_DESI'

delta_z 		= 	0.1

# Parameters associated with the sample / shape noise calcuation 
e_rms_Bl_a 		= 	0.26 # rms ellipticity of sample a, Blazek method.  percomponent value (=0.26) given on confluence page for LSST forecasts, see Chang et al. 2013
e_rms_Bl_b		=	0.26 # rms ellipticity of sample b, Blazek method. Source, same as previous line. 
e_rms_Bl_full	= 	0.26
e_rms_a 		= 	0.26 # rms ellipticity of sample measured with method a, shapes method  
e_rms_b 		= 	0.26 # rms ellipticity of sample measured with method b, shapes method
n_l 			= 	300. # The number of lenses in the lens sample per square DEGREE. DESI Final Design Report, top of page 52. 
Area_l 			=	3000. # Area associated with the lens sample in square DEGREES. Overlap with LSST given in "Spectroscopic Needs for Calibration of LSST Photometric Redshifts" whitepaper.
fsky			=   Area_l / 41253. # Assumes the lens area is the limiting factor
n_s 			=	26. # The effective number density of the full distribution of sources in the sample per square ARCMINUTE. In the abstract of Chang et al. 2013.
S_to_N 			= 	15.6 # Per-galaxy signal-to-noise. Derived from Chang et al. 2013, eqn 9 + values in table 1
a_con			=	[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]	# Fiducial constant offset, approximated from Singh 2016 assuming unprimed method isophotal and primed ReGaussianisation
cov_perc 		= 	[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #percentage covariance between methods, shape measurement method
e_rms_mean 		=	np.abs((e_rms_b+e_rms_a)/2.) # This is the e_rms used in computing the shared weight for shapes methods
N_shapes		= 	Area_l * 3600. * n_s # Number of galaxies in the shape sample.


# Fractional errors on 'fudge factors' we are using to get a handle on the relative importance of different systematic errors.
fudge_frac_level = np.logspace(-4, 0, 500)

# Parameters associated with the projected radial bins
rp_max 	=	20.0 # The maximum projected radius (Mpc/h)
rp_min	=	0.05 # The minimum projected radius (Mpc/h)
N_bins	=	7 # The number of bins of projected radius 

#Parameters of the dNdz of sources, if using an analytic distribution. 'fid' = the fiducial value, 'sys' = 1sigma off from fid value to evaluate systematic error.
dNdztype	=	'Smail'  # Fiducial sample from Chang 2013
alpha_fid 	= 	1.24	 # Fiducial sample from Chang 2013
z0_fid 		=	0.51	 # Fiducial sample from Chang 2013
beta_fid	= 	1.01	 # Fiducial sample from Chang 2013
dNdzpar_fid	=	[alpha_fid, z0_fid, beta_fid] #Put these in a list to facilitate passing around

pztype		=	'Gaussian'
sigz_fid	=	0.05  # The photometric redshift error given a Gaussian photo_z model. From LSST specifications confluence page.
pzpar_fid 	=	[sigz_fid] # Make this a list to make it more generic to pass around

boost_sys	=	0.03 # Multiplier for the boost systematic error. This is the value given in Blazek et al. 2012. It is probably quite a conservative estimate for LSST + DESI.

# Parameters related to the spec and photo z's of the source sample and other redshift cuts.
zeff 	= 	0.77  # The effective redshift of the lens sample. Estimated from Figure 3.8 (COSMOS photo-z estimate) of the DESI final design report, see ./plotting_scripts/DESI_zeff.ipynb
zLmin	=	0.025   # From same figure as above
zLmax	=	1.175   # From same figure as above. 
dNdzL_file	=	'DESI_redshifts_2col.txt'
# Also set the effective z region where we have lenses
zLmin_eff = 0.4
zLmax_eff = 1.175

close_cut = 100 # Mpc/h # The maximum separation from a lens to consider part of `rand-close', in Mpc/h.
#Blazek et al. case
zsmin 	=	0.02
zsmax 	= 	6.0
zphmin	=	0.
zphmax	=	7.0
#delta_z 	=	0.64
zmin_dndz = zsmin
zmax_dndz = 5.0 # The effective point at which the spectroscopic dNdz goes to 0 (input by looking at dNdz for now). For computing volume occupied by source sample.
# Shape measurment case
zmin 	=	zsmin # Minimum spec-z
zmax 	= 	zsmax # Maximum spec-z
zmin_ph	=	0.0 # Minimum photo-z
zmax_ph	=	zphmax # Maximum photo-z

# Constants / conversions
mperMpc = 3.0856776*10**22
Msun = 1.989*10**30 # in kg
Gnewt = 6.67408*10**(-11)
c=2.99792458*10**(8)

# Cosmological parameters:
Nnu	=	3.046    # Massless neutrinos
HH0 =  67.26 
OmR	=	2.47*10**(-5)/(HH0/100.)**2
OmN	=	Nnu*(7./8.)*(4./11.)**(4./3.)*OmR
OmB	=	0.02222/(HH0/100.)**2 
OmC	=	0.1199/(HH0/100.)**2 
OmM=  OmB+OmC
H0	=	10**(5)/c
sigma8	=	0.84
A_s	=	2.2 * 10**(-9)
n_s_cosmo	=	0.9652

cos_par_std = [HH0, OmC, OmB, sigma8]

# We also have two other sets of cosmological parameters:
# one each for the lens and source sample to match the parameters
# simultaneously fit or fixed in sims when fitting HOD parameters:
# (From HOD papers)
HH0_l = 70.3
OmB_l = 0.045
OmC_l = 0.265
sigma8_l = 0.785
n_s_l = 0.95

cos_par_l = [HH0_l, OmC_l, OmB_l, sigma8_l]

HH0_s = 72.0
OmB_s = 0.044
OmC_s = 0.216
sigma8_s = 0.77
n_s_s = 0.95

cos_par_s = [HH0_s, OmC_s, OmB_s, sigma8_s]

# Parameters for getting the fiducial gamma_IA
# 2 halo term parmaeters
kpts_wgg = 10000
kpts_wgp = 2000
bs = 1.3 # Calculated using Zu & Mandelbaum 2015 HOD, see HOD_bias_check.ipynb
bd = 3.9 # Calculated using CMASS More 2014 HOD, see HOD bias_check.ipynb.
C1rho = 0.0134

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

# Here are the parameters for the CMASS HOD model of More et al. 2014 1407.1856. This is higher redshift so it's a better approximation for DESI
kappa_CMASS = 1.25
Mmin_CMASS = 10**(13.13)
M1_CMASS = 10**(14.21)
alpha_CMASS = 1.13
alphainc_CMASS = 0.44
Minc_CMASS = 10**(13.57)
siglogM_CMASS = np.sqrt(0.22)

# 1-halo IA term parameters.
# These are for the model given by Schneider & Bridle 2010, 0903.3870, but for the parameters given in Table 1 of Singh et al. 2014.
# The q_ij parameters are taken directly from this table. a_h is computed by taking the power law of a_h as a function of luminosity a_h(L) from Singh et al. 2014 and integrating it over the Schechter luminosity function from Krause et al. 2015 for the limiting magnitude of LSST shape sample (r<27.5) (see ./ah_Ai_calculation_zLext.ipynb)
q11 = 0.005    
q12  = 5.909
q13 = 0.3798
q21 = 0.6    
q22 = 1.087
q23 = 0.6655
q31 = 3.1    
q32 = 0.1912
q33 = 0.4368

#Parameters required for computing the luminosity function (from Loveday 2012 / Krause et al. 2015 unless otherwise noted)
mlim = 25.3 # See Chang et al. 2013; Science book Chapter 3 Gold sample (i band)
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
