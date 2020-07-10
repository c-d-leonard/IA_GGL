""" This script computes the covariance matrix of Upsilon_{gm} in bins in R.
This version assumes an effective redshift for the lenses, parameterized by comoving distance chiLmean."""

import numpy as np
import scipy
import scipy.interpolate
import scipy.integrate
import pylab
import time
import matplotlib.pyplot as plt
import shared_functions_setup as setup
import shared_functions_wlp_wls as ws
import pyccl as ccl

endfile = 'HOD_fully_updated'

##########################################################################################
######################################## FUNCTIONS #######################################
##########################################################################################

def setup_vectors():
	""" This function sets up all the vectors of points we will need """
	
	lvec				=		scipy.logspace(np.log10(lmin), np.log10(lmax), lpts)
	lvec_less_1			= 		scipy.linspace(lmin, lpts_less, lpts_less-lmin+1)
	lvec_less_2			= 		scipy.logspace(np.log10(lpts_less), np.log10(lmax), lpts_less)
	lvec_less 			= 		np.append(lvec_less_1, lvec_less_2)
	Rvec				=		scipy.logspace(np.log10(Rmin), np.log10(Rmax), Rpts)
	Redges				=		scipy.logspace(np.log10(Rmin), np.log10(Rmax), numRbins+1)
	
	# Want to get the centres of the bins as well so we know where to plot. But, we need to get the centres in log space.
	logRedges=np.log10(Redges)
	Rcentres=np.zeros(numRbins)
	for ri in range(0,numRbins):
		Rcentres[ri]	=	10**((logRedges[ri+1] - logRedges[ri])/2. +logRedges[ri])
		
	return (lvec, lvec_less, Rvec, Redges, Rcentres)
		
def N_of_zph(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, dNdzpar, pzpar, dNdztype, pztype):
	""" Returns dNdz_ph, the number density in terms of photometric redshift, defined over photo-z range (z_a_def_ph, z_b_def_ph), normalized over the photo-z range (z_a_norm_ph, z_b_norm_ph), normalized over the spec-z range (z_a_norm, z_b_norm), and defined on the spec-z range (z_a_def, z_b_def)"""
	
	(z, dNdZ) = setup.get_NofZ_unnormed(dNdzpar, dNdztype, z_a_def_s, z_b_def_s, src_spec_pts, SURVEY)
	(z_norm, dNdZ_norm) = setup.get_NofZ_unnormed(dNdzpar, dNdztype, z_a_norm_s, z_b_norm_s, src_spec_pts, SURVEY)
	
	z_ph_vec = scipy.linspace(z_a_def_ph, z_b_def_ph, src_ph_pts)
	z_ph_vec_norm = scipy.linspace(z_a_norm_ph, z_b_norm_ph, src_ph_pts)
	
	int_dzs = np.zeros(len(z_ph_vec))
	for i in range(0,len(z_ph_vec)):
		int_dzs[i] = scipy.integrate.simps(dNdZ*setup.p_z(z_ph_vec[i], z, pzpar, pztype), z)
	
	int_dzs_norm = np.zeros(len(z_ph_vec_norm))
	for i in range(0,len(z_ph_vec_norm)):
		int_dzs_norm[i] = scipy.integrate.simps(dNdZ_norm*setup.p_z(z_ph_vec_norm[i], z_norm, pzpar, pztype), z_norm)
		
	norm = scipy.integrate.simps(int_dzs_norm, z_ph_vec_norm)
	
	return (z_ph_vec, int_dzs / norm)
	
def sum_weights(photoz_sample, specz_cut, dNdz_par, pz_par):
	""" Returns the sum over lens-source pairs of the estimated weights."""
	
	# Get the distribution of lenses
	zL = scipy.linspace(pa.zLmin, pa.zLmax, 100)
	dndzl = setup.get_dNdzL(zL, SURVEY)
	chiL = com_of_z(zL)
	if (min(chiL)>pa.close_cut):
		zminclose = z_of_com(chiL - pa.close_cut)
	else:
		zminclose = np.zeros(len(chiL))
		for cli in range(0,len(chiL)):
			if (chiL[cli]>pa.close_cut):
				zminclose[cli] = z_of_com(chiL[cli] - pa.close_cut)
			else:
				zminclose[cli] = 0.
	zmaxclose = z_of_com(chiL + pa.close_cut)
	
	# Sum in zphoto at each lens redshift value
	sum_in_zph = np.zeros(len(zL))
	# Loop over lens redshift values
	for zi in range(0,len(zL)):
		
		if (photoz_sample=='close'):
			
			if (specz_cut=='close'):
				(z_ph, dNdz_ph) = N_of_zph(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, zminclose[zi], zmaxclose[zi], pa.zphmin, pa.zphmax, dNdz_par, pz_par, pa.dNdztype, pa.pztype)
			elif(specz_cut=='nocut'):
				(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zminclose[zi], zmaxclose[zi], pa.zphmin, pa.zphmax, dNdz_par, pz_par, pa.dNdztype, pa.pztype)
			else:
				print "We do not have support for that spec-z cut. Exiting."
				exit()
		elif (photoz_sample=='full'):
			if (specz_cut=='close'):
				(z_ph, dNdz_ph) = N_of_zph(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, pa.dNdztype, pa.pztype)
			elif(specz_cut=='nocut'):
				(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, pa.dNdztype, pa.pztype)
			else:
				print "We do not have support for that spec-z cut. Exiting."
				exit()
		else:
			print "We do not have support for that photo-z sample. Exiting."
			exit()
		weight = weights(pa.e_rms_mean, z_ph)
		sum_in_zph[zi] = scipy.integrate.simps(weight * dNdz_ph, z_ph)
	
	# Now sum over all the lenses
	sum_ans = scipy.integrate.simps(sum_in_zph * dndzl, zL)
	
	return sum_ans
	
def sigma_e(z_s_):
	""" Returns a value for the model for the per-galaxy noise as a function of source redshift"""

	if hasattr(z_s_, "__len__"):
		sig_e = 2. / pa.S_to_N * np.ones(len(z_s_))
	else:
		sig_e = 2. / pa.S_to_N

	return sig_e

def weights(e_rms, z_):
	""" Returns the inverse variance weights as a function of redshift. """
	
	weights = (1./(sigma_e(z_)**2 + e_rms**2)) * np.ones(len(z_))
	
	return weights
		
####################### FUNCTIONS FOR GETTING POWER SPECTRA ##############################

def getHconf(xivec):
    """Returns the conformal Hubble constant as a function of zLvec."""
    
    #Get zLvec to correspond with chiLvec
    zLvec	=	z_of_com(xivec)
    Hconf	=	H0 * ( (OmegaM+OmegaB)*(1+zLvec) + OmegaL / (1+zLvec)**2 + (OmegaR+OmegaN) * (1+zLvec)**2 )**(0.5)
    
    return Hconf
    
def getOmMx(xivec):
	"""Returns OmM(x) where OmM(x)=OmB(x)+OmC(x)"""
	#Get zLvec to correspond with chiLvec
	zLvec	=	z_of_com(xivec)

	OmMx= ( OmegaM + OmegaB ) * (1+zLvec)**3 / ((OmegaM+OmegaB)*(1+zLvec)**3 + OmegaL + (OmegaN+OmegaR)*(1+zLvec)**4)
    
	return OmMx

def Pgg_1h2h(xivec):
	""" Returns 1h+2h galaxy x matter power spectrum as a 2 parameter function of l and chi (xivec)."""
	
	# First do 2-halo
	zivec = z_of_com(xivec)
	aivec = 1./ (1. + zivec)
	
	# Compute the power spectrum at a bunch of z's and k's from CCL

	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), sigma8 = pa.sigma8, n_s=pa.n_s_cosmo)
	
	cosmo = ccl.Cosmology(p)
  
	k = scipy.logspace(-4, 4, 1000)
	h = (pa.HH0 / 100.)
	P_2h=np.zeros((len(k), len(aivec)))
	for ai in range(0, len(aivec)):
		P_2h[:, ai] = h**3 * ccl.nonlin_matter_power(cosmo, k * h , aivec[ai])
		
	# Now do 1-halo (this is done in a separate function
	P_1h = ws.get_Pkgg_ll_1halo_kz(k, zivec, y_dm, Mh, kv, SURVEY)
	
	# Add 
	Pofkz = P_1h + bias**2 * P_2h
	
	# Interpolate in k
	Pofkint=[0]*len(zivec)	
	for zi in range(0,len(zivec)):
		Pofkint[zi]=scipy.interpolate.interp1d(k, Pofkz[:,zi])

	# evaluate at k = l / chi
	Poflandx=np.zeros((len(lvec_less),len(xivec)))
	for li in range(0,len(lvec_less)):
		for xi in range(0,len(xivec)):
			if (lvec_less[li]/xivec[xi]<k[-1] and lvec_less[li]/xivec[xi]>k[0]):
				Poflandx[li,xi]=Pofkint[xi](lvec_less[li]/xivec[xi])
			else:
				Poflandx[li,xi]=0.0

	return Poflandx


def get_ns_partial():
	""" Gets the fractional value of ns appropriate for this subsample."""
	
	# To do this, just get the fraction of dNdzph within the sample:
	frac = sum_weights('close', 'nocut', pa.dNdzpar_fid, pa.pzpar_fid) / sum_weights('full', 'nocut', pa.dNdzpar_fid, pa.pzpar_fid)
	
	return frac * ns_tot
############################# FUNCTIONS FOR DOING THE INTEGRALS #######################

def doints_Pgg():
	""" This function does the integrals on the <gg> term"""
	
	# Define a vector of lens redshifts.
	zL = np.linspace(pa.zLmin, pa.zLmax, 100)
	
	# Get the quantities we will need: comoving distance and galaxy power spectrum.
	chi = com_of_z(zL)
	Pdelta = Pgg_1h2h(chi)
	H = getHconf(chi) * (1. + zL)
	
	# Get the lens redshift distribution.	
	dndzl = setup.get_dNdzL(zL, SURVEY)
	
	# Do the integral over dndzl. This includes the 1/ ns term that goes into this term, because it needs integrating over the lens distribution.
	ns = get_ns_partial()
	int_gg = np.zeros(len(lvec_less))
	#clgg= np.zeros(len(lvec_less))
	for li in range(0,len(lvec_less)):
		int_gg[li] = scipy.integrate.simps( dndzl**2 * H * Pdelta[li, :] / (chi**2 * ns), zL)
		#clgg[li] = scipy.integrate.simps( dndzl**2 * H * Pdelta[li, :] / (chi**2), zL)
	
	"""# Compare to CCL	
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = pa.A_s, n_s=pa.n_s_cosmo)
	cosmo = ccl.Cosmology(p)
	b_of_z = bias * np.ones(len(zL))
	gtracer = ccl.cls.ClTracerNumberCounts(cosmo = cosmo, has_rsd = False, has_magnification = False, n = dndzl, bias = b_of_z, z = zL)
	Clgg_ccl = ccl.cls.angular_cl(cosmo, gtracer, gtracer, lvec_less)  
	
	plt.figure()
	plt.loglog(lvec_less, clgg, 'm+')
	plt.hold(True)
	plt.loglog(lvec_less, Clgg_ccl, 'g+')
	#plt.ylim(10**(-11), 10**(-3))
	plt.savefig('./plots/clgg_compare_CCL.pdf')
	plt.close()"""
			
	np.savetxt('./txtfiles/Pgg/Pggterm_gammat_extl_survey='+SURVEY+'_method='+METHOD+'_'+endfile+'.txt', int_gg)
	
	return int_gg

def doconstint():
	""" This function does the integrals for the constant term """
	
	# Integrate answer over dndzl, including over ns for each zl. This squared will be the value we need (because the integrals in primed and unprimed quantities are exactly symmetric).
	ns = get_ns_partial()
	
	save=[0]
	save[0]= gam ** 2 / nl / ns
	
	np.savetxt('./txtfiles/const/const_gammat_extl_survey='+SURVEY+'_method='+METHOD+'_'+endfile+'.txt', save)
	
	return gam ** 2 / nl  / ns

def get_lint():	
#def get_lint(Pgkterm, PggPkkterm, Pkkterm, Pggterm, constterm):
	""" Gets the integral over ell at each R and R' """
	Pggterm		=	np.loadtxt('./txtfiles/Pgg/Pggterm_gammat_extl_survey='+SURVEY+'_method='+METHOD+'_'+endfile+'.txt')
	constterm	=	np.loadtxt('./txtfiles/const/const_gammat_extl_survey='+SURVEY+'_method='+METHOD+'_'+endfile+'.txt')
	
	# plot each thing to see what is dominating:
	plt.figure()
	plt.loglog(lvec_less, Pggterm*gam**2 , 'g+', label='$\propto C_{gg} \gamma^2 / n_s$')
	plt.hold(True)
	plt.loglog(lvec_less, constterm * np.ones(len(lvec_less)), 'k+', label='$\gamma^2 / (n_l n_s)$')
	plt.hold(True)
	plt.loglog(lvec_less, (Pggterm*gam**2 + constterm), 'y+', label='tot')
	plt.ylim(10**(-16), 10**(-10))
	plt.ylabel('Contributions to covariance')
	plt.xlabel('$l$')
	plt.title('Survey='+SURVEY)
	plt.legend()
	plt.savefig('./plots/compareterms_gammat_extl_survey='+SURVEY+'_method='+METHOD+'_'+endfile+'.pdf')
	plt.close()

	
	return 
	

##########################################################################################
####################################### SET UP ###########################################
##########################################################################################

SURVEY = 'SDSS'
METHOD = '1'

print "Survey=", SURVEY

# Import the parameter file:
if (SURVEY=='SDSS'):
	import params as pa
elif (SURVEY=='LSST_DESI'):
	import params_LSST_DESI as pa
else:
	print "We don't have support for that survey yet; exiting."
	exit()
	
# Avoid needing to run twice if we use the same e_rms for both methods:
if ((pa.e_rms_a == pa.e_rms_b)):
	METHOD = 'same_rms'

# Cosmological parameters from parameters file
Nnu	= pa.Nnu; HH0 =	pa.HH0; OmegaR = pa.OmR; OmegaN	= pa.OmN; OmegaB =	pa.OmB; OmegaM = pa.OmC; OmegaK	= 0.0; h =	HH0/100.; 
OmegaL	=	1.-OmegaM-OmegaB-OmegaR-OmegaN

# Constants from parameter file 
c			=	pa.c; MpCm	=	pa.mperMpc; G =	pa.Gnewt; H0 =	10**(5)/c; 

#Directory set up
endfilename		=	SURVEY

# Lenses:
nl			=	pa.n_l * 3282.8 # n_l is in # / square degree, numerical factor converts to # / steradian
bias		=	pa.bd

# Sources:
ns_tot			=	pa.n_s * 3600.*3282.8 # n_s is in # / sqamin, numerical factor converts to / steraidan
fsky			=	pa.fsky
if (METHOD=='1'):
	gam = pa.e_rms_a
elif(METHOD=='2'):
	gam = pa.e_rms_b
elif (METHOD == 'same_rms'):
	gam = pa.e_rms_a
else:
	print "We don't have support for that shape measurement method."
	exit()
	
#Vector set up
src_spec_pts	=	100
src_ph_pts		=	100
Rpts			=	1500
Rmin			=	pa.rp_min
Rmax			=	pa.rp_max
lpts			=	100000
lpts_less		=	500
lmin			=	3
lmax			=	10**6
numRbins		=	pa.N_bins
chiLext_min		=	0.001
chiLext_max		=	setup.com(pa.zphmax, SURVEY, pa.cos_par_std)
chiLextpts		=	250

##########################################################################################
################################  MAIN FUNCTION CALLS ####################################
##########################################################################################

a = time.time()

# Set up
(lvec, lvec_less, Rvec, Redges, Rcentres)					= 		setup_vectors()
z_of_com, com_of_z								=		setup.z_interpof_com(SURVEY)
(z_spec, dNdz_spec)								= 		setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, src_spec_pts, SURVEY)

print "getting y"
# For getting the Fourier transformed density profile
Mh = np.logspace(7., 16., 30)
kv = np.logspace(-4, 4, 40)
y_dm = ws.gety_ldm(Mh, kv, SURVEY)
print "got y"

ns = get_ns_partial()

# Do the integrals on each term up to the l integral (so chiS, bchiS, chiL, bchiL)
Pggints = doints_Pgg() 
print "Done with Pgg integrals. Now do constant:"
constterm = doconstint()
print "Done with constant integrals."
# First, get the l integral in terms of R and R'. This is the long part, and needs only to be done once instead of over and over for each bin.
lint = get_lint()
