# This is an input parameter file for use in calculating forecast constraints on intrinsic alignments using the Blazek et al 2012 formalism.

import numpy as np

# The rms ellipticity of sample a and b.
e_rms_a = 	0.3
e_rms_b	=	0.3

# The signal to noise - necessary for estimating sigma_e
S_to_N = 15.

# The photometric redshift error given a Gaussian photo_z model
sigz=0.08
 
# The factor by which the boost is proportional to the projected correlation function
boost_assoc = 0.2 # Boost at 1 Mpc/h for our associated sample
boost_tot = 0.06 # Boost at 1 Mpc/h for all the source-lens pairs in the survey (all z)
boost_far = 0.03
boost_close = 0.1

# The number of lenses in the lens sample per square DEGREE
n_l 	= 	8.7

# Area associated with the lens sample in square DEGREES
Area 	=	7131

# The systematic error to be assumed on the observed Sigma_c excess (from insufficient spec-z calibration sample)
sys_sigc = 1.0*10**(-5) # This is a dummy number for now.

# The number density of sources in the sample per square ARCMINUTE
n_s 	=	1.2

# The maximum projected radius (Mpc/h)
rp_max 	=	50.0

# The minimum projected radius (Mpc/h)
rp_min	=	0.05

# The number of bins of projected radius 
N_bins	=	15

#Parameters of the dNdz of sources, if using an analytic distribution.
alpha 	= 	2.338
zs	=	0.303
zpts	=	1000  # Number of points in the z vector at which we are evaluating dNdz

# The width of the redshift slice which begins at the lens and ends at the top of sample a
delta_z	=	0.17

# The effective redshift of the lens sample
zeff 	= 	0.28

# The minimum and maximum redshift to consider in the source sample
zsmin 	=	0.0
zsmax 	= 	3.0
zphmin	=	zeff
zphmax	=	5.0

# The maximum separation from a lens to consider part of `rand-close', in Mpc/h
close_cut = 100 # Mpc/h


# Speed of light in units of m/s
c=2.99792458*10**(8)
#G = 

# Cosmological parameters:
Nnu	=	3.046    # Massless neutrinos
HH0 = 70.
#HH0	=	67.26
#HH0 = 72.
OmR	=	2.47*10**(-5)/(HH0/100.)**2
OmN	=	Nnu*(7./8.)*(4./11.)**(4./3.)*OmR
#OmB	=	0.02222/(HH0/100.)**2
#OmC	=	0.1199/(HH0/100.)**2
#OmM=  0.25
OmB = 0.046
OmC = 0.236
OmM = OmB + OmC
H0	=	10**(5)/c

# Constants / conversions
mperMpc = 3.0856776*10**22
Msun = 1.989*10**30 # in kg
Gnewt = 6.67408*10**(-11)


#c14 = 10**(0.7) # Parameter in the concentration / mass relationship (Neto 2007)
Mvir = 10**(13.08) #/ (HH0 / 100.)
#Mvir = 3.4 * 10**12 # units Msol/h
#Mvir = 6. * 10**13 #/ (HH0 / 100.)
#Mvir = 4.7 * 10**12 #in units of M sol
#Mvir = 9.3 * 10**13
kpts_wgg = 10000
kpts_wgp = 2000
sigz_gwin = 0.001
ProjMax = 100.
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

# Files to import error on the boost
sigBF_a = './txtfiles/boost_error_from_rachel_assoc.txt' # File containing two columns: rp (kpc/h), sigma(Boost-1) for sample a
sigBF_b ='./txtfiles/boost_error_from_rachel_background.txt' # Same for b

# 1 halo gal-gal term parameters
ng = 3. * 10**(-4) # volume density of galaxies for BOSS, in h^3 / Mpc^3
nh = 2*10**(-4) # volume density of halos. Must be chosen appropriately for Mvir above. This is a super rough approximation.
Mmin = 1.4 * 10**12 # Taken from Chen 2009, "bright", table 2, in Msol / h
M1 = 2.0*10**13 # Taken from Chen 2009, "bright", table 2, in Msol / h
sat_exp = 1.0 # Taken from Chen 2009, "bright", table 2
Mcut = 2.0*10**12 # Taken from chen 2009, "bright", table 2, in Msol/ h
logMmin = 11.60 # Taken from Zheng 2007, table 1, M_r < 19 (arbitrarily), Mass in Msol/h
siglogM = 0.26 # Taken from Zheng 2007, table 1, M_r< 19






