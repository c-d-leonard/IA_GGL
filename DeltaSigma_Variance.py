""" This script computes the covariance matrix of Upsilon_{gm} in bins in R.
This version assumes an effective redshift for the lenses, parameterized by comoving distance chiLmean."""

import numpy as np
import scipy
import scipy.interpolate
import scipy.integrate
import scipy.misc
import pylab
import time
import matplotlib.pyplot as plt
import shared_functions_setup as setup
import shared_functions_wlp_wls as ws
import pyccl as ccl

##########################################################################################
######################################## FUNCTIONS #######################################
##########################################################################################

def setup_vectors():
	""" This function sets up all the vectors of points we will need """
	
	lvec			=		scipy.logspace(np.log10(lmin), np.log10(lmax), lpts)
	lvec_less_1			= 		scipy.linspace(lmin, lpts_less, lpts_less-lmin+1)
	lvec_less_2			= 		scipy.logspace(np.log10(lpts_less), np.log10(lmax), lpts_less)
	lvec_less = np.append(lvec_less_1, lvec_less_2)
	#lvec_less		=		scipy.linspace(lmin, lmax, lmax-lmin+1)
	print "lvec_less, len=", len(lvec_less)
	#lvec_less		=		scipy.logspace(np.log10(lmin), np.log10(lmax), lpts_less)
	Rvec			=		scipy.logspace(np.log10(Rmin), np.log10(Rmax), Rpts)
	Redges			=		scipy.logspace(np.log10(Rmin), np.log10(Rmax), numRbins+1)
	
	# Want to get the centres of the bins as well so we know where to plot. But, we need to get the centres in log space.
	logRedges=np.log10(Redges)
	Rcentres=np.zeros(numRbins)
	for ri in range(0,numRbins):
		Rcentres[ri]	=	10**((logRedges[ri+1] - logRedges[ri])/2. +logRedges[ri])
		
	return (lvec, lvec_less, Rvec, Redges, Rcentres)

def get_SigmaC_inv(z_s_, z_l_):
	""" Returns the theoretical value of 1/Sigma_c, (Sigma_c = the critcial surface mass density) """

	com_s = com_of_z(z_s_) 
	com_l = com_of_z(z_l_) 

	# Get scale factors for converting between angular-diameter and comoving distances.
	a_l = 1. / (z_l_ + 1.)
	a_s = 1. / (z_s_ + 1.)
	
	D_s = a_s * com_s # Angular diameter source distance.
	D_l = a_l * com_l # Angular diameter lens distance
	D_ls = (D_s - D_l) 
	
	# Units are pc^2 / (h Msun), comoving
	Sigma_c_inv = 4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc *   D_l * D_ls * (1 + z_l_)**2 / D_s
	
	if hasattr(z_s_, "__len__"):
		for i in range(0,len(z_s_)):
			if(z_s_[i]<=z_l_):
				Sigma_c_inv[i] = 0.
	else:
		if (z_s_<=z_l_):
			Sigam_c_inv = 0.

	return Sigma_c_inv
		
def N_of_zph(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, dNdzpar, pzpar, dNdztype, pztype):
	""" Returns dNdz_ph, the number density in terms of photometric redshift, defined over photo-z range (z_a_def_ph, z_b_def_ph), normalized over the photo-z range (z_a_norm_ph, z_b_norm_ph), normalized over the spec-z range (z_a_norm, z_b_norm), and defined on the spec-z range (z_a_def, z_b_def)"""
	
	(z, dNdZ) = setup.get_NofZ_unnormed(dNdzpar, dNdztype, z_a_def_s, z_b_def_s, src_spec_pts)
	(z_norm, dNdZ_norm) = setup.get_NofZ_unnormed(dNdzpar, dNdztype, z_a_norm_s, z_b_norm_s, src_spec_pts)
	
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
	
def sum_weights(z_l, z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, erms, rp_bins_, rp_bin_c_, dNdz_par, pz_par, dNdztype, pztype):
	""" Returns the sum over rand-source pairs of the estimated weights, in each projected radial bin. Pass different z_min_s and z_max_s to get rand-close, rand-far, and all-rand cases."""
	
	(z_ph, dNdz_ph) = N_of_zph(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, dNdz_par, pz_par, dNdztype, pztype)
	
	norm = scipy.integrate.simps(dNdz_ph, z_ph)
	
	sum_ans = [0]*(len(rp_bins_)-1)
	for i in range(0,len(rp_bins_)-1):
		Integrand = dNdz_ph* weights(erms, z_ph, z_l)
		sum_ans[i] = scipy.integrate.simps(Integrand, z_ph)

	return sum_ans
	
def sum_weights_SigC(z_l, z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, erms, rp_bins_, rp_bin_c_, dNdz_par, pz_par, dNdztype, pztype):
	""" Returns the sum over rand-source pairs of the estimated weights multiplied by estimated SigC, in each projected radial bin. Pass different z_min_s and z_max_s to get rand-close, rand-far, and all-rand cases."""
	
	(z_ph, dNdz_ph) = N_of_zph(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, dNdz_par, pz_par, dNdztype, pztype)

	sum_ans = [0]*(len(rp_bins_)-1)
	for i in range(0,len(rp_bins_)-1):
		Integrand = dNdz_ph* weights_times_SigC(erms, z_ph, z_l) 
		sum_ans[i] = scipy.integrate.simps(Integrand, z_ph)

	return sum_ans

def weights(e_rms, z_, z_l_):
	
	""" Returns the inverse variance weights as a function of redshift. """
	
	SigC_t_inv = get_SigmaC_inv(z_, z_l_)
	
	weights = SigC_t_inv**2/(sigma_e(z_)**2 + e_rms**2 * np.ones(len(z_)))
	
	return weights	
	
def sigma_e(z_s_):
	""" Returns a value for the model for the per-galaxy noise as a function of source redshift"""
	
	if (pa.survey=='SDSS'):
		
		if hasattr(z_s_, "__len__"):
			sig_e = 2. / pa.S_to_N * np.ones(len(z_s_))
		else:
			sig_e = 2. / pa.S_to_N
			
	elif(pa.survey=='LSST_DESI'):
		if hasattr(z_s_, "__len__"):
			sig_e = pa.a_sm / pa.SN_med * ( 1. + (pa.b_sm / pa.R_med)**pa.c_sm) * np.ones(len(z_s_))
		else:
			sig_e = pa.a_sm / pa.SN_med * ( 1. + (pa.b_sm / pa.R_med)**pa.c_sm) 

	return sig_e
####################### FUNCTIONS FOR GETTING POWER SPECTRA ##############################

def getHconf(xivec):
    """Returns the conformal Hubble constant as a function of zLvec."""
    
    #Get zLvec to correspond with chiLvec
    zLvec	=	z_ofchi(xivec)
    Hconf	=	H0 * ( (OmegaM+OmegaB)*(1+zLvec) + OmegaL / (1+zLvec)**2 + (OmegaR+OmegaN) * (1+zLvec)**2 )**(0.5)
    
    return Hconf
    
def getOmMx(xivec):
	"""Returns OmM(x) where OmM(x)=OmB(x)+OmC(x)"""
	#Get zLvec to correspond with chiLvec
	zLvec	=	z_ofchi(xivec)

	OmMx= ( OmegaM + OmegaB ) * (1+zLvec)**3 / ((OmegaM+OmegaB)*(1+zLvec)**3 + OmegaL + (OmegaN+OmegaR)*(1+zLvec)**4)
    
	return OmMx

def PofkGR(xivec):
	""" Returns the nonlinear (halofit) 2-halo matter power spectrum today as a 2 parameter function of l and chi (xivec)."""
	
	zivec = z_ofchi(xivec)
	aivec = 1./ (1. + zivec)
	
	# Compute the power spectrum at a bunch of z's and k's from CCL
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = 2.1*10**(-9), n_s=0.96)
	cosmo = ccl.Cosmology(p)
  
	k = scipy.logspace(-4, 4, 1000)
	Pofkz=np.zeros((len(k), len(aivec)))
	for ai in range(0, len(aivec)):
		#for ai in range(0, len(aivec)):
		Pofkz[:, ai] = ccl.nonlin_matter_power(cosmo, k, aivec[ai])
	
	Pofkint=[0]*len(zivec)	
	for zi in range(0,len(zivec)):
		Pofkint[zi]=scipy.interpolate.interp1d(k, Pofkz[:,zi])

	# evaluate at k = l / chi
	Poflandx=np.zeros((len(lvec_less),len(xivec)))
	for li in range(0,len(lvec_less)):
		for xi in range(0,len(xivec)):
			if (lvec_less[li]/xivec[xi]<k[-1] and lvec_less[li]/xivec[xi]>k[0]):
				Poflandx[li,xi]=Pofkint[xi](lvec_less[li]/xivec[xi])
				#if (np.abs(Poflandx[li,xi])<10**(-15)): 
				#	Poflandx[li,xi]=0.0
			else:
				Poflandx[li,xi]=0.0

	return Poflandx
	
def PofkGR_chimean(xi):
	""" Returns the nonlinear (halofit) 2-halo matter power spectrum today as a function of l at the comoving distance OF THE LENSES."""
	
	# Compute the power spectrum at a bunch of k's and at z of the lenses from CCL
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = 2.1*10**(-9), n_s=0.96)
	cosmo = ccl.Cosmology(p)
    
	k = scipy.logspace(-4, 2, 1000)
	Pofk=np.zeros(len(k))
	for ki in range(0, len(k)):
		Pofk[ki] = ccl.nonlin_matter_power(cosmo, k[ki], 1. / (1. + zval))
	
	Pofkint=scipy.interpolate.interp1d(k, Pofk)

	#Interpolate such that k = l / chi
	Poflandx=np.zeros(len(lvec_less))
	for li in range(0,len(lvec_less)):
		if (lvec_less[li]/xi<k[-1] and lvec_less[li]/xi>k[0]):
			Poflandx[li]=Pofkint(lvec_less[li]/xi)
			if (np.abs(Poflandx[li]))<10**(-15): 
				Poflandx[li]=0.0
		else:
			Poflandx[li]=0.0

	return Poflandx
	
def get_Pgg():
	""" This function computes P_{gg}(l, chiLmean) """
	
	# Get the nonlinear matter power spectrum:
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = 2.1*10**(-9), n_s=0.96)
	cosmo = ccl.Cosmology(p)
	
	# We are going to get Cl_{gg} using CCL. For this, we need to define a N(z) for the lenses, even though we are using an effective redshifts. We're going to use a narrow Gaussian.
	
	sig_fudge = 0.05
	z = np.linspace(zval - 5. * sig_fudge, zval + 5. * sig_fudge, 1000)
	N_of_z = 1. / np.sqrt(2. * np.pi) / sig_fudge * np.exp( - (z-zval)**2 / (2. *sig_fudge**2))
	b_of_z = bias * np.ones(len(z))
	
	gtracer = ccl.cls.ClTracerNumberCounts(cosmo = cosmo, has_rsd = False, has_magnification = False, n = N_of_z, bias = b_of_z, z = z)
	
	ltracer = ccl.cls.ClTracerLensing(cosmo=cosmo,has_intrinsic_alignment=False, n= dNdz_spec, z= z_spec)
	
	Clgg = ccl.cls.angular_cl(cosmo, gtracer, gtracer, lvec_less)
	
	Clgk = ccl.cls.angular_cl(cosmo, gtracer, ltracer, lvec_less)
	
	Clkk = ccl.cls.angular_cl(cosmo, ltracer, ltracer, lvec_less)
	
	plt.figure()
	plt.loglog(lvec_less, Clgg, 'g+')
	plt.hold(True)
	plt.loglog(lvec_less, Clgk, 'm+')
	plt.hold(True)
	plt.loglog(lvec_less, Clkk, 'b+')
	plt.savefig('./plots/three_spectra_CCL.pdf')
	plt.close()
	
	

	return  Clgg
	
def get_Pgk():
	""" This function computes P_{gk}(l, chi_L, chi_S) """
	
	# We're going to try doing this in terms of a slightly less thing lens distribution to see if a failing Limber approximation is causing a problem:
	sig_fudge = 0.05
	z = z_spec
	#z = np.linspace(zval - 5. * sig_fudge, zval + 5. * sig_fudge, 200)
	N_of_z = 1. / np.sqrt(2. * np.pi) / sig_fudge * np.exp( - (z-zval)**2 / (2. *sig_fudge**2))
	
	#N_of_z_src = 1. / np.sqrt(2. * np.pi) / sig_fudge * np.exp( - (z-0.8)**2 / (2. *sig_fudge**2))
	b_of_z = bias * np.ones(len(z))
	chiLext			=		com_of_z(z)
	
	norm_dNdz_spec = scipy.integrate.simps(dNdz_spec, z_spec)
	
	chiSvec = setup.com(z_spec, SURVEY)
	
	H=getHconf(chiLext)
	Omz=getOmMx(chiLext)
	Pof_lx=PofkGR(chiLext)
	
	H_interp = scipy.interpolate.interp1d(chiLext, H)
	Omz_interp = scipy.interpolate.interp1d(chiLext, Omz)
	P_interp = [0] * len(lvec_less)
	for li in range(0,len(lvec_less)):
		P_interp[li] = scipy.interpolate.interp1d(chiLext, Pof_lx[li, :])
	
	#plt.figure()
	#plt.plot(z_spec, N_of_z_src)
	#plt.show()
	
	# Get the max index in chiLext up to which we want to integrate:
	index_chiL = [0]*len(chiSvec)
	for ci in range(0, len(chiSvec)):
		if (chiSvec[ci]<max(chiLext)):
			index_chiL[ci] = next(j[0] for j in enumerate(chiLext) if j[1]>=chiSvec[ci])
			#print "ci=", ci, "index=", index_chiL[ci]
	
	Clgk=np.zeros((len(lvec_less), len(chiSvec)))
	for li in range(0, len(lvec_less)):
		#print "li=", li
		for xiS in range(0, len(chiSvec)):
			if (index_chiL[xiS]==0):
				Clgk[li, xiS] = 0.
			else:
				chiext = scipy.linspace(chiLext[0], chiLext[index_chiL[xiS]], 1000)
				N_of_z_inst =1. / np.sqrt(2. * np.pi) / sig_fudge * np.exp( - (z_ofchi(chiext) - zval)**2 / (2. *sig_fudge**2))
				#Clgk[li, xiS] = 1.5 * bias * H0**3 * scipy.integrate.simps((chiSvec[xiS] - chiLext[0:index_chiL[xiS]]) / chiSvec[xiS] * N_of_z[0:index_chiL[xiS]] * H[0:index_chiL[xiS]] / H0 / chiLext[0:index_chiL[xiS]]* (H[0:index_chiL[xiS]] / H0)**2 * Omz[0:index_chiL[xiS]] * Pof_lx[li, :][0:index_chiL[xiS]], chiLext[0:index_chiL[xiS]])
				Clgk[li, xiS] = 1.5 * bias * H0**3 * scipy.integrate.simps((chiSvec[xiS] - chiext) / chiSvec[xiS] * N_of_z_inst * H_interp(chiext) / H0 / chiext* (H_interp(chiext) / H0)**2 * Omz_interp(chiext) * P_interp[li](chiext), chiext)
			#if (chiSvec[xiS]<min(chiLext)):
			#	Clgk[li, xiS] = 0.
			#elif (chiSvec[xiS]<max(chiLext)):
			#	Clgk[li, xiS] = 1.5 * bias * scipy.integrate.simps((chiSvec[xiS] - chiLext[0:index_chiL[xiS]]) / chiSvec[xiS] * N_of_z[0:index_chiL[xiS]] * H[0:index_chiL[xiS]] / chiLext[0:index_chiL[xiS]]* H[0:index_chiL[xiS]]**2 * Omz[0:index_chiL[xiS]] * Pof_lx[li, :][0:index_chiL[xiS]], chiLext[0:index_chiL[xiS]]) 
			#else: 
			#	Clgk[li, xiS] = 0.
				
	# Try seeing what this looks like integrated over dNdzspec- it should in principle match ccl pretty well:
	Clgk_intS = np.zeros(len(lvec_less))
	for li in range(0,len(lvec_less)):
		Clgk_intS[li] = scipy.integrate.simps(Clgk[li, :] * dNdz_spec / norm_dNdz_spec, z_spec)
		
	
		
	#plt.figure()
	#plt.plot(z_spec, dNdz_spec)
	#plt.show()
	
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = 2.1*10**(-9), n_s=0.96)
	cosmo = ccl.Cosmology(p)	
	gtracer = ccl.cls.ClTracerNumberCounts(cosmo = cosmo, has_rsd = False, has_magnification = False, n = N_of_z, bias = b_of_z, z = z)
	ltracer = ccl.cls.ClTracerLensing(cosmo=cosmo,has_intrinsic_alignment=False, n= dNdz_spec, z= z_spec)
	Clgg_CCL = ccl.cls.angular_cl(cosmo, gtracer, gtracer, lvec_less)
	Clgk_CCL = ccl.cls.angular_cl(cosmo, gtracer, ltracer, lvec_less)
	Clkk_CCL = ccl.cls.angular_cl(cosmo, ltracer, ltracer, lvec_less)
	
	#plt.figure()
	#plt.loglog(lvec_less, Clgg, 'g+')
	#plt.hold(True)
	#plt.loglog(lvec_less, Clgk, 'm+')
	#plt.hold(True)
	#plt.loglog(lvec_less, Clkk, 'b+')
	#plt.savefig('./plots/three_spectra_CCL.pdf')
	#plt.close()

		
	plt.figure()
	plt.loglog(lvec_less, Clgg_CCL, 'k+')
	plt.hold(True)
	plt.loglog(lvec_less, Clgk_intS, 'm+')
	plt.hold(True)
	plt.loglog(lvec_less, Clgk_CCL, 'g+')
	plt.hold(True)
	plt.loglog(lvec_less, Clkk_CCL, 'k+')
	plt.ylim(10**(-13), 10**(-3))
	plt.savefig('./plots/spectra_compare.pdf')
	plt.close()
	
	plt.figure()
	plt.plot(lvec_less, np.abs((Clgk_intS - Clgk_CCL) / Clgk_CCL), 'm+')
	#plt.ylim(10**(-13), 10**(-3))
	plt.savefig('./plots/Clgk_fracdiff_300zpts.pdf')
	plt.close()
	
	return  Clgk

def getwbar():
	""" This function computes wbar, the integral over the SigmaC factors associated with the weights."""
	# Get dNdzph:
	(z_ph, Nofzph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, minz_src_p, maxz_src_p, minz_src_p, maxz_src_p, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
	
	# Do the integral in photo-z
	Sigma_inv = get_SigmaC_inv(z_ph, zval)
	wbar = scipy.integrate.simps(Sigma_inv**2 * Nofzph, z_ph)
	
	return wbar

def get_ns_partial():
	""" Gets the fractional value of ns appropriate for this subsample."""
	
	# To do this, just get the weighted fraction of dNdzph within the sample:
	
	frac = np.asarray(sum_weights(pa.zeff, pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, minz_src_p, maxz_src_p, pa.zphmin, pa.zphmax, gam, Redges, Rcentres, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype))[0] / np.asarray(sum_weights(pa.zeff, pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax,gam, Redges, Rcentres, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype))[0]
	
	print "frac_weighted=", frac

	return frac * ns_tot
	
############################# FUNCTIONS FOR DOING THE INTEGRALS #######################
	
def doints_Pgg(Clgg):
	""" This function does the integrals in dchiL, dbarchiL, dchiS, dbarchiS on the P_{gg} * gamma^2 / ns term"""
	
	# Get dNdzph:
	(z_ph, Nofzph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, minz_src_p, maxz_src_p, minz_src_p, maxz_src_p, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
	
	# Do the integral in photo-z
	Sigma_inv = get_SigmaC_inv(z_ph, zval)
	barchiS_int = scipy.integrate.simps(Sigma_inv * Nofzph, z_ph)

	# Now load Clgg
	#Clgg=np.loadtxt(folderpath+outputfolder+'/Clgg_'+endfilename+'.txt')	
	np.savetxt('./txtfiles/Pggterm_DeltaSig_'+endfilename+'_sample='+SAMPLE+'.txt', barchiS_int**2 * Clgg )
	
	return barchiS_int**2 * Clgg
	
def doints_Pgk(Clgk):
	""" This function does the integrals in dchiL, dbarchiL, dchiS, dbarchiS on the P_{gk} term"""

	# Define the zph vector
	z_ph_vec = scipy.linspace(minz_src_p, maxz_src_p, src_ph_pts)
	
	# Do the integral over spectroscopic redshift: dNdz_s * p(zs, zp) * Clgk
	int_dzs = np.zeros((len(lvec_less),len(z_ph_vec)))
	for li in range(0,len(lvec_less)):
		for zi in range(0,len(z_ph_vec)):
			int_dzs[li, zi] = scipy.integrate.simps(dNdz_spec * setup.p_z(z_ph_vec[zi], z_spec, pa.pzpar_fid, pa.pztype) * Clgk[li,:], z_spec)
			
	# Now do the integral in zph
	Sigma_inv = get_SigmaC_inv(z_ph_vec, zval)
	int_dzph = np.zeros(len(lvec_less))
	for li in range(0,len(lvec_less)):
		int_dzph[li] = scipy.integrate.simps(Sigma_inv * int_dzs[li,:], z_ph_vec)
	
	# Get the factor to normalize this stuff
	int_dzs_norm = np.zeros(len(z_ph_vec))
	for i in range(0,len(z_ph_vec)):
		int_dzs_norm[i] = scipy.integrate.simps(dNdz_spec*setup.p_z(z_ph_vec[i], z_spec, pa.pzpar_fid, pa.pztype), z_spec)
		
	norm = scipy.integrate.simps(int_dzs_norm, z_ph_vec)
	
	chiL_intans = int_dzph / norm
	
	np.savetxt('./txtfiles/Pgkterm_DeltaSig_'+endfilename+'_sample='+SAMPLE+'.txt', chiL_intans**2 )
		
	return chiL_intans**2
	
def doints_Pkk():
	""" This function does the integrals in dchiL, dbarchiL, dchiS, dbarchiS on the P_{kk} / nl term. For this one, we haven't precomputed Cl_kk because it's more efficient to do the integrals in a less pedogogical order. We do them all in this function."""
	 
	# Get the chi over which we integrate. It's called chiLext because this is a cosmic shear power spectrum so the lenses are extended in space.
	chiLext			=		scipy.linspace(chiLext_min, chiLext_max, chiLextpts)	
	
	H=getHconf(chiLext)
	Omz=getOmMx(chiLext)
	Pof_lx=PofkGR(chiLext)
	zLext = z_ofchi(chiLext)
	
	# Define the zph vector
	z_ph_vec = scipy.linspace(minz_src_p, maxz_src_p, src_ph_pts)
	chiS = com_of_z(z_spec)
	
	# For each value in the vector of chiLext, get the first index of the chiSvector for which chiS is greater than chiLext:
	index_chiS = [0]*len(chiLext)
	for ci in range(0, len(chiLext)):
		if (chiLext[ci]<max(chiS)):
			index_chiS[ci] = next(j[0] for j in enumerate(chiS) if j[1]>=chiLext[ci])
	
	# Integral over z_s
	int_z_s = np.zeros((len(chiLext), len(z_ph_vec)))
	for xiL in range(0, len(chiLext)):
		for zpi in range(0, len(z_ph_vec)):
			if (chiLext[xiL]>max(chiS)):
				int_z_s[xiL,zpi] =0.
			else:
				int_z_s[xiL, zpi] = scipy.integrate.simps( dNdz_spec[index_chiS[xiL]:] * setup.p_z(z_ph_vec[zpi], z_spec[index_chiS[xiL]:], pa.pzpar_fid, pa.pztype) * ( chiS[index_chiS[xiL]:] - chiLext[xiL]) / chiS[index_chiS[xiL]:] , z_spec[index_chiS[xiL]:] )
			
	# Integral over corresponding z_ph
	Sigma_inv = get_SigmaC_inv(z_ph_vec, zval)
	int_z_p = np.zeros(len(chiLext))
	for xi in range(0,len(chiLext)):
		int_z_p[xi] = scipy.integrate.simps( Sigma_inv * int_z_s[xi, :] , z_ph_vec ) 
		
	# Integral over z_s'
	int_z_s_prime = np.zeros((len(chiLext), len(z_ph_vec)))
	for xiL in range(0, len(chiLext)):
		for zpi in range(0, len(z_ph_vec)):
			if (chiLext[xiL]>max(chiS)):
				int_z_s_prime[xiL, zpi] = 0.
			else:
				int_z_s_prime[xiL, zpi] = scipy.integrate.simps( dNdz_spec[index_chiS[xiL]:]  * setup.p_z(z_ph_vec[zpi], z_spec[index_chiS[xiL]:] , pa.pzpar_fid, pa.pztype) * ( chiS[index_chiS[xiL]:]  - chiLext[xiL]) / chiS[index_chiS[xiL]:]  * int_z_p[xiL] , z_spec[index_chiS[xiL]:]  )
			
	# And the corresponding z_ph'
	int_z_p_prime = np.zeros(len(chiLext))
	for xi in range(0,len(chiLext)):
		int_z_p_prime[xi] = scipy.integrate.simps( Sigma_inv * int_z_s_prime[xi, :], z_ph_vec)
		
	# Now do the integral over chi, the "extended lens" comoving distance for cosmic shear
	int_chiLext = np.zeros(len(lvec_less))
	for li in range(0,len(lvec_less)):
		int_chiLext[li] = 9./4.*H0**4 * scipy.integrate.simps((H/H0)**4 * Omz**2 * Pof_lx[li,:] * int_z_p_prime , chiLext)
		
	# Get the factor to normalize this:
	int_dzs_norm = np.zeros(len(z_ph_vec))
	for i in range(0,len(z_ph_vec)):
		int_dzs_norm[i] = scipy.integrate.simps(dNdz_spec*setup.p_z(z_ph_vec[i], z_spec, pa.pzpar_fid, pa.pztype), z_spec)
		
	norm = scipy.integrate.simps(int_dzs_norm, z_ph_vec)

	# Norm is squared because there are two integrals over dNdz_s here.
	bchiS_intans = int_chiLext / norm**2
	
	np.savetxt('./txtfiles/Pkkterm_DeltaSig_'+endfilename+'_sample='+SAMPLE+'.txt', bchiS_intans )
	
	return bchiS_intans
	
def doints_PggPkk(Pkkterm, Clgg):
	""" This function constructs the Pgg*Pkk term from Pkk and Clgg"""
	
	#Load integrals over Pkk
	#Pkkterm = np.loadtxt(folderpath+outputfolder+'/Pkkterm_'+endfilename+'.txt')
	
	#And load Clgg
	#Clgg=np.loadtxt(folderpath+outputfolder+'/Clgg_'+endfilename+'.txt')
	
	#The PggPkk term is simply these two things multiplied in the effective lens redshift case:
	PggPkk = Clgg * Pkkterm
	
	np.savetxt(folderpath+outputfolder+'/PggPkkterm_DeltaSig_'+endfilename+'_sample='+SAMPLE+'.txt', PggPkk)
	
	return PggPkk
	
def doconstint():
	""" This function does the integrals in chiS, bchiS, chiL and bchiL for the constant term """
	
	# Get dNdzph:
	(z_ph, Nofzph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, minz_src_p, maxz_src_p, minz_src_p, maxz_src_p, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
	
	# Do the integral in photo-z
	Sigma_inv = get_SigmaC_inv(z_ph, zval)
	chiSans = scipy.integrate.simps(Sigma_inv * Nofzph, z_ph)
	
	# The bchiL / bchiS integrals are the exact same so we don't need to do them again 
	ns = get_ns_partial()
	save=[0]
	save[0]=chiSans ** 2 * gam ** 2 / ns / nl
	
	np.savetxt('./txtfiles/const_DeltaSig_'+endfilename+'_sample='+SAMPLE+'.txt', save)
	ns = get_ns_partial()
	
	return chiSans ** 2 * gam ** 2 / ns / nl
	
def get_lint():
	""" Gets the integral over ell at each R and R' """
	Pgkterm		=	np.loadtxt('./txtfiles/Pgkterm_DeltaSig_'+endfilename+'_sample='+SAMPLE+'.txt')
	PggPkkterm	=	np.loadtxt('./txtfiles/PggPkkterm_DeltaSig_'+endfilename+'_sample='+SAMPLE+'.txt')
	Pkkterm		= 	np.loadtxt('./txtfiles/Pkkterm_DeltaSig_'+endfilename+'_sample='+SAMPLE+'.txt')
	Pggterm		=	np.loadtxt('./txtfiles/Pggterm_DeltaSig_'+endfilename+'_sample='+SAMPLE+'.txt')
	constterm	=	np.loadtxt('./txtfiles/const_DeltaSig_'+endfilename+'_sample='+SAMPLE+'.txt')
	
	# plot each thing to see what is dominating:
	plt.figure()
	plt.loglog(lvec_less, Pgkterm, 'b+', label='$\propto (C_{g \kappa})^2$')
	plt.hold(True)
	plt.loglog(lvec_less, PggPkkterm, 'r+', label='$\propto (C_{gg} C_{\kappa \kappa})$')
	plt.hold(True)
	plt.loglog(lvec_less, Pkkterm , 'm+', label='$\propto C_{\kappa \kappa}$')
	plt.hold(True)
	plt.loglog(lvec_less, Pggterm, 'g+', label='$\propto C_{gg}$')
	#plt.ylim(10**(-22), 10**(-14))
	plt.ylabel('Contributions to covariance, power spectra only')
	plt.xlabel('$l$')
	plt.title('Survey='+SURVEY+', sample='+SAMPLE)
	plt.legend()
	plt.savefig('./plots/compareterms_powerspectra_alone_survey='+SURVEY+'_sample='+SAMPLE+'.pdf')
	plt.close()
	
	print "nl=", nl
	print "ns=", ns
	print "gam**2=", gam**2
	
	# plot each thing to see what is dominating:
	plt.figure()
	plt.loglog(lvec_less, Pgkterm, 'b+', label='$\propto (C_{g \kappa})^2$')
	plt.hold(True)
	plt.loglog(lvec_less, PggPkkterm, 'r+', label='$\propto (C_{gg} C_{\kappa \kappa})$')
	plt.hold(True)
	plt.loglog(lvec_less, Pkkterm / nl, 'm+', label='$\propto C_{\kappa \kappa} / n_l$')
	plt.hold(True)
	plt.loglog(lvec_less, Pggterm*gam**2 / ns, 'g+', label='$\propto C_{gg} \gamma^2 / n_s$')
	plt.hold(True)
	plt.loglog(lvec_less, constterm * np.ones(len(lvec_less)), 'k+', label='$\gamma^2 / (n_l n_s)$')
	plt.hold(True)
	plt.loglog(lvec_less, ( Pgkterm + PggPkkterm + Pkkterm/nl + Pggterm*gam**2 / ns + constterm), 'y+', label='tot')
	plt.ylim(10**(-25), 10**(-16))
	plt.ylabel('Contributions to covariance')
	plt.xlabel('$l$')
	plt.title('Survey='+SURVEY+', sample='+SAMPLE)
	plt.legend()
	plt.savefig('./plots/compareterms_DeltaSigcov_survey='+SURVEY+'_sample='+SAMPLE+'.pdf')
	plt.close()
	
	exit()
	
	# Interpolate these things to get the result in terms of the more highly sampled lvec
	Pgg_interp = scipy.interpolate.interp1d(lvec_less, Pggterm)
	Pgg_higher_res = Pgg_interp(lvec)
	
	Pgk_interp = scipy.interpolate.interp1d(lvec_less, Pgkterm)
	Pgk_higher_res = Pgk_interp(lvec)
	
	PggPkk_interp = scipy.interpolate.interp1d(lvec_less, PggPkkterm)
	PggPkk_higher_res = PggPkk_interp(lvec)
	
	Pkk_interp = scipy.interpolate.interp1d(lvec_less, Pkkterm)
	Pkk_higher_res = Pkk_interp(lvec)
	
	exit()

	
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
			lint_ans[ri, rip] = scipy.integrate.trapz((constterm) * Bessel_two[:, ri] * Bessel_two[:, rip] * lvec, lvec)
			#lint_ans[ri, rip] = scipy.integrate.simps(( Pgkterm + PggPkkterm + Pkkterm/nl + Pggterm*gam**2 / ns + constterm) * scipy.special.jv(2, Rvec[ri] * lvec) * scipy.special.jv(2, Rvec[rip] * lvec) * lvec, lvec)
			#lint_ans[ri, rip] = scipy.integrate.simps(( Pgkterm + PggPkkterm + Pkkterm/nl + Pggterm*gam**2 / ns + constterm) * scipy.special.jv(2, Rvec[ri] * lvec / chiLmean) * scipy.special.jv(2, Rvec[rip] * lvec / chiLmean) * lvec, lvec)
			#lint_ans[ri, rip] = scipy.integrate.trapz((constterm) * scipy.special.jv(2, Rvec[ri] * lvec / chiLmean) * scipy.special.jv(2, Rvec[rip] * lvec / chiLmean) * lvec, lvec)
			#print "rip=",rip, "lint=", lint_ans[ri, rip]
		#exit()
			#print "lint_ans[ri, rip]=", lint_ans[ri, rip]
			
	for ri in range(0, len(Rvec)):
		for rip in range(0, ri):
			lint_ans[ri,rip] = lint_ans[rip,ri]
	
	return lint_ans
	
def do_outsideints_SigR(i_Rbin, j_Rbin, lint_ans):
	""" This function does the integrals in l, R, and R' for the Delta Sigma(R) term """
		
	# Now do the Rprime integral.
	Rlowind_bini    =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[i_Rbin])
	Rhighind_bini   =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[i_Rbin+1])
		
	Rprime_intans=np.zeros(len(Rvec))	
	for ri in range(len(Rvec)):
		Rprime_intans[ri] = scipy.integrate.simps(lint_ans[ri,:][Rlowind_bini:Rhighind_bini], Rvec[Rlowind_bini:Rhighind_bini]) / (Rvec[Rhighind_bini] - Rvec[Rlowind_bini])

	# Now the r integral:
	Rlowind_binj    =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[j_Rbin])
	Rhighind_binj   =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[j_Rbin+1])
	Rintans = scipy.integrate.simps(Rprime_intans[Rlowind_binj:Rhighind_binj], Rvec[Rlowind_binj:Rhighind_binj]) / (Rvec[Rhighind_binj] - Rvec[Rlowind_binj])	

	# Add factors:
	ans_thisRbin	= Rintans  / (8. * np.pi**2) / fsky /wbar**2
	print "ans_thisRbin=", ans_thisRbin
	
	return ans_thisRbin	
	

##########################################################################################
####################################### SET UP ###########################################
##########################################################################################

SURVEY = 'LSST_DESI'
SAMPLE = 'B'
# Import the parameter file:
if (SURVEY=='SDSS'):
	import params as pa
elif (SURVEY=='LSST_DESI'):
	import params_LSST_DESI as pa
else:
	print "We don't have support for that survey yet; exiting."
	exit()

# Cosmological parameters from parameters file
Nnu	= pa.Nnu; HH0 =	pa.HH0; OmegaR = pa.OmR; OmegaN	= pa.OmN; OmegaB =	pa.OmB; OmegaM = pa.OmC; OmegaK	= 0.0; h =	HH0/100.; 
OmegaL		=	1.-OmegaM-OmegaB-OmegaR-OmegaN

# Constants from parameter file 
c			=	pa.c; MpCm	=	pa.mperMpc; G =	pa.Gnewt; H0 =	10**(5)/c; 

#Directory set up
folderpath 		= 	'/home/danielle/Dropbox/CMU/Research/Intrinsic_Alignments/'
inputfolder		=	'/txtfiles/'
outputfolder		=	'/txtfiles/'
endfilename		=	SURVEY

# Lenses:
zval 		= 	pa.zeff
chiLmean 	=	setup.com(zval, SURVEY)
nl			=	pa.n_l * 3282.8 # n_l is in # / square degree, numerical factor converts to # / steradian
bias		=	pa.bd_Bl

# Sources:
ns_tot			=	pa.n_s * 3600.*3282.8 # n_s is in # / sqamin, numerical factor converts to / steraidan
gam				=	pa.e_rms_mean
fsky			=	pa.fsky

#maxchival_s 	=	com(pa.zsmax, SURVEY)
#minchival_s		=	com(pa.zsmin, SURVEY)
if (SAMPLE == 'A'):
	maxz_src_p	=	pa.zeff+pa.delta_z
	minz_src_p	=	pa.zeff
if (SAMPLE == 'B'):
	maxz_src_p	=	pa.zphmax
	minz_src_p	=	pa.zeff+pa.delta_z
	
#Vector set up
src_spec_pts	=	200
src_ph_pts		=	200
Rpts			=	1500
Rmin			=	pa.rp_min
Rmax			=	pa.rp_max
lpts			=	10**5
lpts_less		=	500
lmin			=	3
lmax			=	10**6
numRbins		=	pa.N_bins
chiLext_min		=	0.1
chiLext_max		=	setup.com(pa.zsmax, SURVEY)
chiLextpts		=	100

##########################################################################################
################################  MAIN FUNCTION CALLS ####################################
##########################################################################################

a = time.time()

# Set up
(lvec, lvec_less, Rvec, Redges, Rcentres)					= 		setup_vectors()
print "len(lvec_less)=", len(lvec_less)
z_ofchi, com_of_z								=		setup.z_interpof_com(SURVEY)
(z_spec, dNdz_spec)								= 		setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, src_spec_pts)

ns = get_ns_partial()

# Get power spectra
Clgg 	= 	get_Pgg()
print "get Pgg done"
Clgk	=	get_Pgk()
print "get Pgk done"

# Do the integrals on each term up to the l integral (so chiS, bchiS, chiL, bchiL)
Pggints = doints_Pgg(Clgg)
print "Done with Pgg integrals. Now do Pgk:"
Pgkints = doints_Pgk(Clgk)
print "Done with Pgk integrals. Now do Pkk:"
Pkkints = doints_Pkk()
print "Done with Pkk integrals. Now do constant:"
constterm = doconstint()
print "Done with constant integrals. Now do PggPkk:"
PggPkkints = doints_PggPkk(Pkkints, Clgg)
print "Done with PggPkk integrals. Now getting wbar:"
wbar = getwbar()
print "wbar=", wbar
print "Done with getting wbar. Now doing integrals over R:"


# First, get the l integral in terms of R and R'. This is the long part, and needs only to be done once instead of over and over for each bin.
lint = get_lint()

#This must be done for each set of bins
covariance		=	np.zeros((numRbins, numRbins))

for i_R in range(0, numRbins):
	for j_R in range(0, numRbins):
		print "i bin=", i_R, "j bin=", j_R
		covariance[i_R, j_R]	=	do_outsideints_SigR(i_R, j_R, lint)
		print "Covariance=", covariance[i_R, j_R]
		
np.savetxt(folderpath+outputfolder+'/cov_Upgm_'+endfilename+'_sample='+SAMPLE+'_rpts'+str(Rpts)+'_lpts'+str(lpts)+'_TEST_ELL_DOWNSAMPLE.txt', covariance)

print '\nTime for completion:', '%.1f' % (time.time() - a), 'seconds'
