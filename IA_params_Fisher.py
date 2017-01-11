# This is an input parameter file for use in calculating forecast constraints on intrinsic alignments using the Blazek et al 2012 formalism.

import numpy as np

# The parameter labels to be constrainted in the Fisher forecast
A	=	0
beta	=	1
par	=	[A, beta]
A_fid	=	0.059
beta_fid	=	-0.73

# The rms ellipticity of sample a and b.
e_rms_a = 	0.3
e_rms_b	=	0.3

# The signal to noise - necessary for estimating sigma_e
S_to_N = 15.

# The photometric redshift error given a Gaussian photo_z model
sigz=0.08
 
# The factor by which the boost is proportional to the projected correlation function
Boost_prop = 0.05

# The number of lenses in the lens sample per square DEGREE
n_l 	= 	8.7

# Area associated with the lens sample in square DEGREES
Area 	=	7131

# The systematic error to be assumed on the observed Sigma_c excess (from insufficient spec-z calibration sample)
sys_sigc = 1.0*10**(-5) # This is a dummy number for now.

# The number density of sources in the sample per square ARCMINUTE
n_s 	=	1.2

# The maximum projected radius (Mpc/h)
rp_max 	=	200.0

# The minimum projected radius (Mpc/h)
rp_min	=	0.05

# The number of bins of projected radius 
N_bins	=	7

#Parameters of the dNdz of sources, if using an analytic distribution.
alpha 	= 	2.338
#beta 	= 	1.5
#zm 	= 	0.5  #0.4
zs	=	0.303
zpts	=	1000  # Number of points in the z vector at which we are evaluating dNdz


# The width of the redshift slice which begins at the lens and ends at the top of sample a
delta_z	=	0.17

# The effective redshift of the lens sample
zeff 	= 	0.32

# The minimum and maximum redshift to consider in the source sample
zmin 	=	0.0
zmax 	= 	3.0
zphmin	=	0.0	
zphmax	=	5.0

# The maximum separation from a lens to consider part of `rand-close', in Mpc/h
close_cut = 100 # Mpc/h

# Location of file for plot showing 1 sigma error bars on gamma_IA
plotfile =	'./Nakajima_nofz.pdf'

# Speed of light in units of m/s
c=2.99792458*10**(8)
#G = 

# Cosmological parameters:
Nnu	=	3.046    # Massless neutrinos
HH0	=	67.26
OmR	=	2.47*10**(-5)/(HH0/100.)**2
OmN	=	Nnu*(7./8.)*(4./11.)**(4./3.)*OmR
OmB	=	0.02222/(HH0/100.)**2
OmC	=	0.1199/(HH0/100.)**2
H0	=	10**(5)/c

#
c14 = 10**(0.7) # Parameter in the concentration / mass relationship (Neto 2007)
Mvir = 10.**(13.18) / (HH0/100.)
#Mvir = 10.**11
