# Parameters for both shape-measurement and Blazek et al. method of estimating IA. 

import numpy as np

run_quants		=	False
survey 			=	'LSST_DESI'

# Parameters associated with the sample / shape noise calcuation 
e_rms_Bl_a 		= 	0.18 # rms ellipticity of sample a, Blazek method. 1/sqrt(2) * percomponent value (=0.26) given on confluence page for LSST forecasts.
e_rms_Bl_b		=	0.18 # rms ellipticity of sample b, Blazek method. Source, same as previous line.
e_rms_a 		= 	0.175 # rms ellipticity of sample measured with method a, shapes method
e_rms_b 		= 	0.185 # rms ellipticity of sample measured with method b, shapes method
n_l 			= 	300. # The number of lenses in the lens sample per square DEGREE. DESI Final Design Report, top of page 52. 
Area_l 			=	3000. # Area associated with the lens sample in square DEGREES. Overlap with LSST given in "Spectroscopic Needs for Calibration of LSST Photometric Redshifts" whitepaper.
fsky			=   Area_l / 41253. # Assumes the lens area is the limiting factor
n_s 			=	26. # The effective number density of sources in the sample per square ARCMINUTE. In the abstract of Chang et al. 2013.
a_con			=	[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]	# Fiducial constant offset, approximated from Singh 2016 assuming unprimed method isophotal and primed ReGaussianisation
cov_perc 		= 	[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #percentage covariance between methods, shape measurement method
e_rms_mean 		=	np.abs((e_rms_b+e_rms_a)/2.) # This is the e_rms used in computing the shared weight for shapes methods
N_shapes		= 	Area_l * 3600. * n_s# Number of galaxies in the shape sample.
N_LRG			=	Area_l * n_l # Number of galaxies in the LRG sample.

# Parameters for getting sigma_e, measurements error. From Chang et al. 2013
a_sm 			=	1.58
b_sm			= 	5.03
c_sm			=	0.39
SN_med			=	11.8
R_med			=	1.63

# Fractional errors on 'fudge factors' we are using to get a handle on the relative importance of different systematic errors.
fudge_frac_level = np.logspace(-4, 0, 15)
fudge_Ncorr = 	0.
fudge_czA	= 	0.
fudge_czB	=	0.
fudge_sigA	=	0.
fudge_sigB	=	0.
fudge_FA	=	0.
fudge_FB	=	0.
 
# The factor by which (Boost - 1) is proportional to the projected correlation function at r_p = 1 Mpc/h
# Computed for this sample using equation 4.10 of Blazek et al. 2012 and the parameters in this file; see ../secondary_scripts/get_boost_amplitudes.ipynb
boost_assoc = 0.76
boost_far = 10**(-7)
boost_close = 0.03

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
#zpts		=	1000  # Number of points in the z vector at which we are evaluating dNdz
sigz_fid	=	0.05  # The photometric redshift error given a Gaussian photo_z model. From LSST specifications confluence page.
pzpar_fid 	=	[sigz_fid] # Make this a list to make it more generic to pass around

#boost_sys	=	1.00 # Setting this to 1 for now because I want to ignore boost sys error at the moment.
boost_sys	=	1.03 # Multiplier for the boost systematic error. This is the value given in Blazek et al. 2012. It is probably quite a conservative estimate for LSST + DESI.

# Parameters related to the spec and photo z's of the source sample and other redshift cuts.
zeff 	= 	0.77  # The effective redshift of the lens sample. Estimated from Figure 3.8 (COSMOS photo-z estimate) of the DESI final design report, see ./plotting_scripts/DESI_zeff.ipynb
close_cut = 100 # Mpc/h # The maximum separation from a lens to consider part of `rand-close', in Mpc/h.
#Blazek et al. case
zsmin 	=	0.0
zsmax 	= 	6.0
zphmin	=	0.
zphmax	=	7.0
delta_z	=	0.57 # The width of the redshift slice which begins at the lens and ends at the top of sample a. Chosen to have roughly same number of l-s pairs in each bin for zeff = 0.77 and LSST dNdzph.
zmin_dndz = zsmin
zmax_dndz = 5.0 # The effective point at which the spectroscopic dNdz goes to 0 (input by looking at dNdz for now). For computing volume occupied by source sample.
# Shape measurment case
#zpts	=	10000  # Number of points in the z vector at which we are evaluating dNdz
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
HH0 = 	70. #67.26 #72
OmR	=	2.47*10**(-5)/(HH0/100.)**2
OmN	=	Nnu*(7./8.)*(4./11.)**(4./3.)*OmR
OmB	=	0.05 #0.02222/(HH0/100.)**2 #0.046
OmC	=	0.2 #0.1199/(HH0/100.)**2 #0.236
OmM=  	OmB+OmC
H0	=	10**(5)/c

# Parameters for getting the fiducial gamma_IA
# 2 halo term parmaeters
sigz_gwin = 0.0001 # This is the variance of the gaussian we are using for the lens distribution, chosen to approximate a delta function.
kpts_wgg = 5000
kpts_wgp = 2000
bs_Bl = 2.05 # This is found by converting the median M_B of the four bins in table 2 of 0708.0004 (DEEP2) to Luminosities, then fitting a line to b(L). ----> (next line)
bs_shapes = 2.05 # Then, integrate this b(L) over the luminosity function from Krause et al 2015 (1506.08730) and over dNdz. see ../secondar_scripts/get_source_bias.ipynb
bd_Bl = 3.57 # Lens sample, assuming zeff = 0.77 and b = 1.7 / D(z)
bd_shapes = 3.57 # Lens sample, assuming zeff = 0.77 and b = 1.7 / D(z)
Ai_Bl = 4.28 # Computed assuming r_lim = 27 and lum func / A(L) relationship from Krause et al 2015 (1506.08730). See ../secondary_scripts/Ai_calculation.ipynb
Ai_shapes = 4.28 # Computed assuming r_lim = 27 and lum func / A(L) relationship from Krause et al 2015 (1506.08730). See ../secondary_scripts/Ai_calculation.ipynb
C1rho = 0.0134

# 1 halo gal-gal term parameters
Mvir = 1.6 * 10**(14) # I have estimated this from extending the Mavg line in Figure 3.4 of the DESI design book to z=0.8. This is for the BGS sample in the first place so I'm not sure it makes sense. 
ng_Bl = 3. * 10**(-4)
fsat_LRG = 0.14 # The DESI volume density is roughly the same as that of BOSS LOWZ, so I am using the value for BOSS LOWZ that Sukhdeep told me. But I'm not sure sure about this, because it assumes the DESI LRG sample is volume-limited and there is no effect from going to higher redshift.

##### Parameters of the HOD model, taken from Zu & Mandelbaum 2015, 1505.02781.  #####
#######  THESE ARE PARAMETERS DESCRIBING THE HALO OCCUPATION FOR SDSS MGS SPEC-Z GALAXIES IN HALOS ASSOCIATED TO THE SAME SAMPLE. HAVE NOT YET CHANGED TO DESI + LSST #######
# Ncen params: these refer to the central galaxies for SDSS MGS:
sigMs = 0.50
eta = -0.04
M1 = 10**(12.10)

# Nsat params: these refer at the moment to satellite occupation for SDSS MGS galaxies
Bsat = 8.98
beta_sat =0.90
Bcut = 0.86
beta_cut = 0.41
alpha_sat = 1.00

# f_SHMR parameters: these are used in getting Ncen and Nsat. We may need to have two sets of these parameters.
delta = 0.42
gamma= 1.21
Mso = 10**(10.31)
beta = 0.33

# 1-halo IA term parameters.
# These are for the model given by Schneider & Bridle 2010, 0903.3870, but for the parameters given in Table 1 of Singh et al. 2014.
# The q_ij parameters are taken directly from this table. a_h is computed by taking the power law of a_h as a function of luminosity a_h(L) from Singh et al. 2014 and integrating it over the Schechter luminosity function from Krause et al. 2015 for the limiting magnitude of LSST shape sample (r<27), and the LSST smail-form redshift distribution. (see ./secondary_scripts/ah_calculation.ipynb)
ah_Bl  =  0.46
q11_Bl = 0.005    
q12_Bl  = 5.909
q13_Bl  = 0.3798
q21_Bl  = 0.6    
q22_Bl  = 1.087
q23_Bl  = 0.6655
q31_Bl  = 3.1    
q32_Bl  = 0.1912
q33_Bl  = 0.4368
ah_shapes =  0.46
q11_shapes = 0.005  
q12_shapes = 5.909
q13_shapes = 0.3798
q21_shapes = 0.6    
q22_shapes = 1.087
q23_shapes = 0.6655
q31_shapes = 3.1    
q32_shapes = 0.1912
q33_shapes = 0.4368


