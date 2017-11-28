""" This script computes the covariance matrix of Upsilon_{gm} in bins in R.
This version assumes an effective redshift for the lenses, parameterized by comoving distance chiLmean."""

import numpy as np
import scipy
import scipy.interpolate
import scipy.integrate
#import pylab
import time
#import matplotlib.pyplot as plt
import shared_functions_setup as setup
#import shared_functions_wlp_wls as ws
#import pyccl as ccl

##########################################################################################
######################################## FUNCTIONS #######################################
##########################################################################################

def setup_vectors():
	""" This function sets up all the vectors of points we will need """
	
	lvec			=		scipy.logspace(np.log10(lmin), np.log10(lmax), lpts)
	lvec_less_1			= 		scipy.linspace(lmin, lpts_less, lpts_less-lmin+1)
	lvec_less_2			= 		scipy.logspace(np.log10(lpts_less), np.log10(lmax), lpts_less)
	lvec_less = np.append(lvec_less_1, lvec_less_2)
	Rvec			=		scipy.logspace(np.log10(Rmin), np.log10(Rmax), Rpts)
	Redges			=		scipy.logspace(np.log10(Rmin), np.log10(Rmax), numRbins+1)
	
	# Want to get the centres of the bins as well so we know where to plot. But, we need to get the centres in log space.
	logRedges=np.log10(Redges)
	Rcentres=np.zeros(numRbins)
	for ri in range(0,numRbins):
		Rcentres[ri]	=	10**((logRedges[ri+1] - logRedges[ri])/2. +logRedges[ri])
		
	return (lvec, lvec_less, Rvec, Redges, Rcentres)

	
def get_lint():
	""" Gets the integral over ell at each R and R' """

	Pgkterm		=	np.loadtxt('./txtfiles/Pgkterm_DeltaSig_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt')
	PggPkkterm	=	np.loadtxt('./txtfiles/PggPkkterm_DeltaSig_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt')
	Pkkterm		= 	np.loadtxt('./txtfiles/Pkkterm_DeltaSig_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt')
	Pggterm		=	np.loadtxt('./txtfiles/Pggterm_DeltaSig_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt')
	
	# Interpolate these things to get the result in terms of the more highly sampled lvec
	Pgg_interp = scipy.interpolate.interp1d(lvec_less, Pggterm)
	Pgg_higher_res = Pgg_interp(lvec)
	
	Pgk_interp = scipy.interpolate.interp1d(lvec_less, Pgkterm)
	Pgk_higher_res = Pgk_interp(lvec)
	
	PggPkk_interp = scipy.interpolate.interp1d(lvec_less, PggPkkterm)
	PggPkk_higher_res = PggPkk_interp(lvec)
	
	Pkk_interp = scipy.interpolate.interp1d(lvec_less, Pkkterm)
	Pkk_higher_res = Pkk_interp(lvec)
	

	
	# It might be faster to contruct the bessel function as an array in l and r just once, since it's hte same for both factors just with different r values, then you don't have to call it so often:
	Bessel_two = np.zeros((len(lvec), len(Rvec)))
	for ri in range(0,len(Rvec)):
		print "ri=", ri
		for li in range(0,len(lvec)):
			Bessel_two[li, ri] = scipy.special.jv(2, Rvec[ri] * lvec[li] / chiLmean)
	lint_ans=np.zeros((len(Rvec), len(Rvec)))
	for ri in range(0,len(Rvec)):
		print "ri, outside ints R=", ri
		for rip in range(ri,len(Rvec)):	
			lint_ans[ri, rip] = scipy.integrate.trapz(( Pgk_higher_res + PggPkk_higher_res + Pkk_higher_res/nl + Pgg_higher_res*gam**2)  * Bessel_two[:, ri] * Bessel_two[:, rip] * lvec, lvec)

			
	for ri in range(0, len(Rvec)):
		for rip in range(0, ri):
			lint_ans[ri,rip] = lint_ans[rip,ri]
	
	return lint_ans
	
def do_outsideints_SigR(i_Rbin, j_Rbin, lint_ans):
	""" This function does the integrals in l, R, and R' for the Delta Sigma(R) term """

	wbar = np.loadtxt('./txtfiles/wbar_extl'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt')
		
	# Now do the Rprime integral.
	Rlowind_bini    =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[i_Rbin])
	Rhighind_bini   =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[i_Rbin+1])
		
	Rprime_intans=np.zeros(len(Rvec))	
	for ri in range(len(Rvec)):
		Rprime_intans[ri] = scipy.integrate.simps(lint_ans[ri,:][Rlowind_bini:Rhighind_bini]*Rvec[Rlowind_bini:Rhighind_bini]  , Rvec[Rlowind_bini:Rhighind_bini])*2. / (Rvec[Rhighind_bini]**2 - Rvec[Rlowind_bini]**2)

	# Now the r integral:
	Rlowind_binj    =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[j_Rbin])
	Rhighind_binj   =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[j_Rbin+1])
	Rintans = scipy.integrate.simps(Rprime_intans[Rlowind_binj:Rhighind_binj]*Rvec[Rlowind_binj:Rhighind_binj]  , Rvec[Rlowind_binj:Rhighind_binj]) * 2. / (Rvec[Rhighind_binj]**2 - Rvec[Rlowind_binj]**2)	

	# Add factors:
	ans_thisRbin	= Rintans  / (8. * np.pi**2) / fsky /wbar**2
	print "ans_thisRbin=", ans_thisRbin
	
	return ans_thisRbin	
	
def add_shape_noise(i_Rbin, j_Rbin, ravg):
	""" Adds the shape noise term to the diagonal elements """

	wbar = np.loadtxt('./txtfiles/wbar_extl'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt')

	if (i_Rbin != j_Rbin):
		with_shape_noise = ravg
		shapenoise_alone = 0.
	else:
		constterm      =       np.loadtxt('./txtfiles/const_DeltaSig_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt')
		Rlowind_bini    =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[i_Rbin])
	        Rhighind_bini   =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[i_Rbin+1])
		Rlowind_binj    =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[j_Rbin])
        	Rhighind_binj   =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[j_Rbin+1])

		with_shape_noise = ravg + constterm*chiLmean**2 / (4.*np.pi**2*fsky*wbar**2 * (Rvec[Rhighind_bini]**2 - Rvec[Rlowind_bini]**2))

		shapenoise_alone = constterm*chiLmean**2 / (4.*np.pi**2*fsky*wbar**2 * (Rvec[Rhighind_bini]**2 - Rvec[Rlowind_bini]**2))

	return (with_shape_noise, shapenoise_alone)


##########################################################################################
####################################### SET UP ###########################################
##########################################################################################

SURVEY = 'LSST_DESI'
SAMPLE = 'A'
# Import the parameter file:
if (SURVEY=='SDSS'):
	import params as pa
elif (SURVEY=='LSST_DESI'):
	import params_LSST_dz0pt1 as pa
else:
	print "We don't have support for that survey yet; exiting."
	exit()

# Cosmological parameters from parameters file
Nnu	= pa.Nnu; HH0 =	pa.HH0; OmegaR = pa.OmR; OmegaN	= pa.OmN; OmegaB =	pa.OmB; OmegaM = pa.OmC; OmegaK	= 0.0; h =	HH0/100.; 
OmegaL		=	1.-OmegaM-OmegaB-OmegaR-OmegaN

# Constants from parameter file 
c			=	pa.c; MpCm	=	pa.mperMpc; G =	pa.Gnewt; H0 =	10**(5)/c; 

#Directory set up
endfilename		=	SURVEY

# Lenses:
zval 		= 	pa.zeff
chiLmean 	=	setup.com(zval, SURVEY)
nl			=	pa.n_l * 3282.8 # n_l is in # / square degree, numerical factor converts to # / steradian

# Sources:
gam				=	pa.e_rms_mean
fsky			=	pa.fsky

	
#Vector set up
Rpts			=	10
Rmin			=	pa.rp_min
Rmax			=	pa.rp_max
lpts			=	10**5
lpts_less		=	500
lmin			=	3
lmax			=	10**6
numRbins		=	pa.N_bins


##########################################################################################
################################  MAIN FUNCTION CALLS ####################################
##########################################################################################

a = time.time()

# Set up
(lvec, lvec_less, Rvec, Redges, Rcentres)					= 		setup_vectors()
z_ofchi, com_of_z								=		setup.z_interpof_com(SURVEY)


# First, get the l integral in terms of R and R'. This is the long part, and needs only to be done once instead of over and over for each bin.
lint = get_lint()

#This must be done for each set of bins
R_averaged		=	np.zeros((numRbins, numRbins))
covariance 		=	np.zeros((numRbins, numRbins))
cov_shapenoise_alone	=	np.zeros((numRbins, numRbins))
for i_R in range(0, numRbins):
	for j_R in range(0, numRbins):
		print "i bin=", i_R, "j bin=", j_R
		R_averaged[i_R, j_R]	=	do_outsideints_SigR(i_R, j_R, lint)
		covariance[i_R, j_R], cov_shapenoise_alone[i_R, j_R]	=	add_shape_noise(i_R, j_R, R_averaged[i_R, j_R])
		print "Covariance=", covariance[i_R, j_R]
		
np.savetxt('./txtfiles/cov_DelSig_zLext_'+endfilename+'_sample='+SAMPLE+'_rpts'+str(Rpts)+'_lpts'+str(lpts)+'_deltaz='+str(pa.delta_z)+'.txt', covariance)
np.savetxt('./txtfiles/shapenoiseonly_DelSig_zLext_'+endfilename+'_sample='+SAMPLE+'_rpts'+str(Rpts)+'_lpts'+str(lpts)+'_deltaz='+str(pa.delta_z)+'.txt', cov_shapenoise_alone)

print '\nTime for completion:', '%.1f' % (time.time() - a), 'seconds'
