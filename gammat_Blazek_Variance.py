""" This script computes the covariance matrix of DeltaSigma_{gm} in bins in R.
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
	
def sum_weights(photoz_sample, specz_cut, color_cut, rp_bins, dNdz_par, pz_par, dNdztype, pztype):
	""" Returns the sum over weights for each projected radial bin. 
	photoz_sample = 'A', 'B', or 'full'
	specz_cut = 'close', or 'nocut'
	"""
	
	# Get lens redshift distribution
	zL = np.linspace(pa.zLmin, pa.zLmax, 100)
	dNdzL = setup.get_dNdzL(zL, SURVEY)
	chiL = com_of_z(zL)
	if (min(chiL)> (pa.close_cut + com_of_z(pa.zsmin))):
		zminclose = z_of_com(chiL - pa.close_cut)
	else:
		zminclose = np.zeros(len(chiL))
		for cli in range(0,len(chiL)):
			if (chiL[cli]>pa.close_cut + com_of_z(pa.zsmin)):
				zminclose[cli] = z_of_com(chiL[cli] - pa.close_cut)
			else:
				zminclose[cli] = pa.zsmin
	zmaxclose = z_of_com(chiL + pa.close_cut)
	
	# Get norm, required for the color cut case:
	zph_norm = np.linspace(pa.zphmin, pa.zphmax, 1000)
	(zs_norm, dNdzs_norm) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 1000)
	zs_integral_norm = np.zeros(len(zph_norm))
	for zpi in range(0,len(zph_norm)):
		pz = setup.p_z(zph_norm[zpi], zs_norm, pa.pzpar_fid, pa.pztype)
		zs_integral_norm[zpi] = scipy.integrate.simps(pz * dNdzs_norm, zs_norm)
	norm = scipy.integrate.simps(zs_integral_norm, zph_norm)
	
	# Loop over lens redshift values
	sum_ans_zph = np.zeros(len(zL))
	for zi in range(0,len(zL)):
		
		if (color_cut=='all'):
			if(photoz_sample =='assocBl'):
				if (pa.delta_z<zL[zi]):
					zminph = zL[zi] - pa.delta_z
				else:
					zminph = 0.
						
				if (specz_cut == 'nocut'):
					(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zminph, zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
					weight = weights(pa.e_rms_Bl_a, z_ph)
					sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
				elif (specz_cut == 'close'):
					(z_ph, dNdz_ph) = N_of_zph(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, zminph, zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
					weight = weights(pa.e_rms_Bl_a, z_ph)
					sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
				else:
					print "We do not have support for that spec-z cut. Exiting."
					exit()
			
			elif(photoz_sample == 'B'):
			
				if (specz_cut == 'nocut'):
					(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi] + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
					weight = weights(pa.e_rms_Bl_b, z_ph)
					sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
				elif (specz_cut == 'close'):
					(z_ph, dNdz_ph) = N_of_zph(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, zL[zi] + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
					weight = weights(pa.e_rms_Bl_b, z_ph)
				
					sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
				else:
					print "We do not have support for that spec-z cut. Exiting."
					exit()
		
			elif(photoz_sample == 'full'):
			
				if (specz_cut == 'nocut'):
					(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
					weight = weights(pa.e_rms_Bl_full, z_ph)
				
					sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
				elif (specz_cut == 'close'):
					(z_ph, dNdz_ph) = N_of_zph(zminclose[zi], zmaxclose[zi], pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
					weight = weights(pa.e_rms_Bl_full, z_ph)
				
					sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
				else:
					print "We do not have support for that spec-z cut. Exiting."
					exit()
			else:
				print "We do not have support for that photo-z sample cut. Exiting."
				print photoz_sample
				exit()
	
		else:
			print "We do not have support for that color cut, exiting."
			exit()
			
	# Now sum over lens redshift:
	sum_ans = scipy.integrate.simps(sum_ans_zph * dNdzL, zL)
	
	return sum_ans
	
def weights(e_rms, z_):
	""" Returns the inverse variance weights as a function of redshift. """
	
	weights = (1./(sigma_e(z_)**2 + e_rms**2)) * np.ones(len(z_))
	
	return weights	
	
def sigma_e(z_s_):
	""" Returns a value for the model for the per-galaxy noise as a function of source redshift"""
	
	if hasattr(z_s_, "__len__"):
		sig_e = 2. / pa.S_to_N * np.ones(len(z_s_))
	else:
		sig_e = 2. / pa.S_to_N 

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

def Pgm_1h2h(xivec):
	""" Returns 1h+2h galaxy x matter power spectrum as a 2 parameter function of l and chi (xivec)."""
	
	# First do 2-halo
	zivec = z_ofchi(xivec)
	aivec = 1./ (1. + zivec)
	# Compute the power spectrum at a bunch of z's and k's from CCL
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = pa.A_s, n_s=pa.n_s_cosmo)
	cosmo = ccl.Cosmology(p)
  
	k = scipy.logspace(-4, 4, 1000)
	P_2h=np.zeros((len(k), len(aivec)))
	h = (pa.HH0/100.)
	for ai in range(0, len(aivec)):
		# Need to be careful about h factors here - ccl calls class which uses Mpc units but we use Mpc/h.
		P_2h[:, ai] = h**3 * ccl.nonlin_matter_power(cosmo, k * h, aivec[ai])
		
	# Now do 1-halo (this is done in a separate function)
	P_1h = ws.get_Pkgm_1halo_kz(k, zivec, SURVEY)
	
	# Add 
	Pofkz = P_1h + bias * P_2h 
	
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
				#if (np.abs(Poflandx[li,xi])<10**(-15)): 
				#	Poflandx[li,xi]=0.0
			else:
				Poflandx[li,xi]=0.0

	return Poflandx
	
def Pgg_1h2h(xivec):
	""" Returns 1h+2h galaxy x matter power spectrum as a 2 parameter function of l and chi (xivec)."""
	
	# First do 2-halo
	zivec = z_ofchi(xivec)
	aivec = 1./ (1. + zivec)
	# Compute the power spectrum at a bunch of z's and k's from CCL
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = pa.A_s, n_s=pa.n_s_cosmo)
	cosmo = ccl.Cosmology(p)
  
	k = scipy.logspace(-4, 4, 1000)
	P_2h=np.zeros((len(k), len(aivec)))
	h = (pa.HH0/100.)
	for ai in range(0, len(aivec)):
		# Need to be careful about h factors here - ccl calls class which uses Mpc units but we use Mpc/h.
		P_2h[:, ai] = h**3 * ccl.nonlin_matter_power(cosmo, k * h , aivec[ai])
		
	# Now do 1-halo (this is done in a separate function
	P_1h = ws.get_Pkgg_ll_1halo_kz(k, zivec, SURVEY)
	
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
	
def Pmm_1h2h(xivec):
	""" Returns 1h+2h galaxy x matter power spectrum as a 2 parameter function of l and chi (xivec)."""
	
	# First do 2-halo
	zivec = z_ofchi(xivec)
	aivec = 1./ (1. + zivec)
	# Compute the power spectrum at a bunch of z's and k's from CCL
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = pa.A_s, n_s=pa.n_s_cosmo)
	cosmo = ccl.Cosmology(p)
  
	k = scipy.logspace(-4, 4, 1000)
	P_2h=np.zeros((len(k), len(aivec)))
	h = (pa.HH0/100.)
	for ai in range(0, len(aivec)):
		# Need to be careful about h factors here - ccl calls class which uses Mpc units but we use Mpc/h.
		P_2h[:, ai] = h**3 * ccl.nonlin_matter_power(cosmo, k * h , aivec[ai])
		
	# Now do 1-halo (this is done in a separate function
	P_1h = ws.get_Pkmm_1halo_kz(k, zivec, SURVEY)
	
	# Add 
	Pofkz = P_1h + P_2h
	
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
				#if (np.abs(Poflandx[li,xi])<10**(-15)): 
				#	Poflandx[li,xi]=0.0
			else:
				Poflandx[li,xi]=0.0

	return Poflandx

##############################################

def get_ns_partial(zL):
	""" Gets the fractional value of ns appropriate for this subsample."""
	
	# We have the effective surface density of sources for the full sample, ns_tot. but, for every lens redshift we have a different section of the source parameter space that is included. We get the effective surface density for the sample at each value of the given vector zL:
	
	dNdzph_int = np.zeros(len(zL))
	ns_samp = np.zeros(len(zL))
	for zi in range(0,len(zL)):
		if (SAMPLE=='assocBl'):
			if (pa.delta_z<zL[zi]):
				zminph = zL[zi] - pa.delta_z
			else:
				zminph = 0.
			(z_ph, Nofzph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zminph, zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)

		elif (SAMPLE=='B'):
			(z_ph, Nofzph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi] + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
		else:
			print "We do not have support for that sample. Exiting."
			exit()
		ns_samp[zi] = ns_tot * scipy.integrate.simps(Nofzph, z_ph)
		dNdzph_int[zi] = scipy.integrate.simps(Nofzph, z_ph)
		
	#print "dNdzph=", dNdzph_int
	#exit()
		
	# Just for testing, get this integrated over the lens redshift. Really, we pass this out as a function of zl
	#dndzl = setup.get_dNdzL(zL, SURVEY)
	
	#averaged_ns_samp = scipy.integrate.simps(ns_samp * dndzl, zL)

	return ns_samp
	
############################# FUNCTIONS FOR DOING THE INTEGRALS #######################

def doints_Pgg():
	""" This function does the integrals on the <gg> term"""
	
	# Define a vector of lens redshifts.
	zL = np.linspace(pa.zLmin, pa.zLmax, 100)
	
	# Get the quantities we will need: comoving distance and limber-approximated galaxy power spectrum.
	chi = com_of_z(zL)
	Pdelta = Pgg_1h2h(chi)
	H = getHconf(chi) * (1. + zL)
	
	# Get the lens redshift distribution.	
	dndzl = setup.get_dNdzL(zL, SURVEY)
	
	# Do the final integral over dndzl. This includes the 1/ ns term that goes into this term, because it needs integrating over the lens distribution.
	ns = get_ns_partial(zL)
	int_gg = np.zeros(len(lvec_less))
	#clgg= np.zeros(len(lvec_less))
	for li in range(0,len(lvec_less)):
		int_gg[li] = scipy.integrate.simps( dndzl**2 * H * Pdelta[li, :] / (chi**2 * ns), zL)
		#clgg[li] = scipy.integrate.simps( dndzl**2 * H * Pdelta[li, :] / (chi**2), zL)
	
	# Compare to CCL	
	"""p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = pa.A_s, n_s=pa.n_s_cosmo)
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
			
	np.savetxt('./txtfiles/Pggterm_gamtBl_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt', int_gg)
	
	return int_gg

def doints_Pgk():
	""" This function does the integrals on the <gk> term""" 
	
	# Define a vector of lens redshifts.
	zL = np.linspace(pa.zLmin, pa.zLmax, 100)
	
	# Get the quantities we will need
	chi = com_of_z(zL)
	H=getHconf(chi)
	Omz=getOmMx(chi)
	Pdelta = Pgm_1h2h(chi)
	
	# Get the norm of the spectroscopic redshift dNdzs - Use this only for computing Clgk to compare with CCL
	(zsnorm, dNdzsnorm) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 500)
	#normzs = scipy.integrate.simps(dNdzsnorm, zsnorm)
	
	int_in_zl = np.zeros(len(zL))
	#int_in_zs_clgk = np.zeros(len(zL))
	z_ph = [0] * len(zL)
	for zi in range(0, len(zL)):
		
		# Define the photo-z vector:
		if (SAMPLE=='assocBl'):
			if (pa.delta_z<zL[zi]):
				zminph = zL[zi] - pa.delta_z
			else:
				zminph = 0.
			z_ph[zi] = np.linspace(zminph, zL[zi]+pa.delta_z, 500)
		elif (SAMPLE=='B'):
			z_ph[zi] = np.linspace(zL[zi] + pa.delta_z, pa.zphmax, 500)
		
		# Get the integral over spec z
		(zs, dNdzs) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zL[zi], pa.zsmax, 500)
		int_in_zs = np.zeros(len(z_ph[zi]))
		for zpi in range(0,len(z_ph[zi])):
			pz = setup.p_z(z_ph[zi][zpi], zs, pa.pzpar_fid, pa.pztype)
			int_in_zs[zpi] = scipy.integrate.simps(pz*dNdzs*(com_of_z(zs) - chi[zi])/(com_of_z(zs)), zs)
			
		# Get the integral over photo z
		int_in_zl[zi] = scipy.integrate.simps(int_in_zs, z_ph[zi])
		
		#int_in_zs_clgk[zi] = scipy.integrate.simps(dNdzs / normzs *(com_of_z(zs) - chi[zi])/(com_of_z(zs)), zs)
		
	# Get the normalization we want for dNdzph
	zphnorm = np.zeros(len(zL))
	for zi in range(0, len(zL)):
		int_in_zs = np.zeros(len(z_ph[zi]))
		for zpi in range(0, len(z_ph[zi])):
			pz = setup.p_z(z_ph[zi][zpi], zsnorm, pa.pzpar_fid, pa.pztype)
			int_in_zs[zpi] = scipy.integrate.simps(pz * dNdzsnorm, zsnorm)
		zphnorm[zi] = scipy.integrate.simps(int_in_zs, z_ph[zi])
				
	# Get the lens redshift distribution
	dndzl = setup.get_dNdzL(zL, SURVEY)
	
	# Do the integral over the lens redshift distribution:
	int_gk = np.zeros(len(lvec_less)) 
	#clgk = np.zeros(len(lvec_less))
	for li in range(0,len(lvec_less)):
		int_gk[li] = H0**2 * scipy.integrate.simps(1.5 * dndzl * Omz * (H / H0)**2 * Pdelta[li, :] * int_in_zl / (chi * zphnorm), zL)
		#clgk[li] = H0**2 * scipy.integrate.simps(1.5 * dndzl * Omz * (H / H0)**2 * Pdelta[li, :] * int_in_zs_clgk / (chi ), zL)
		
	# Compare with CCL
	"""p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = pa.A_s, n_s=pa.n_s_cosmo)
	cosmo = ccl.Cosmology(p)
	gtracer = ccl.cls.ClTracerNumberCounts(cosmo = cosmo, has_rsd = False, has_magnification = False, n = dndzl, bias = pa.bd*np.ones(len(zL)), z = zL)
	(zs, dNdzs_unnormed) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 500)
	dNdzs = dNdzs_unnormed / normzs
	ltracer = ccl.cls.ClTracerLensing(cosmo=cosmo,has_intrinsic_alignment=False, n=dNdzs, z= zs)
	Clgk_ccl = ccl.cls.angular_cl(cosmo, gtracer, ltracer, lvec_less) 
	
	plt.figure()
	plt.loglog(lvec_less, clgk, 'm+')
	plt.hold(True)
	plt.loglog(lvec_less, Clgk_ccl, 'g+')
	#plt.ylim(10**(-11), 10**(-3))
	plt.savefig('./plots/clgk_compare_CCL.pdf')
	plt.close()"""
	
	np.savetxt('./txtfiles/Pgkterm_gamtBl_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt', int_gk**2 )
		
	return int_gk**2
	
def doints_Pkk():
	""" This function does the integrals in dchiL, dbarchiL, dchiS, dbarchiS on the P_{kk} / nl term. """
	 
	# Get the chi over which we integrate. It's called chiLext because this term deals with a cosmic shear power spectrum so the 'lenses' for this are extended in space.
	chiLext			=		scipy.linspace(chiLext_min, chiLext_max, chiLextpts)	
	H=getHconf(chiLext)
	Omz=getOmMx(chiLext)
	Pof_lx = Pmm_1h2h(chiLext)
	zLext = z_ofchi(chiLext)
	
	# We also need the z vector for the actual lens galaxy distribution
	zlens = np.linspace(pa.zLmin, pa.zLmax, 500)
	dndzl = setup.get_dNdzL(zlens, SURVEY)
	
	# Get the norm of the spectroscopic redshift source distribution over the whole range - use this only for getting Clkk
	(zs_norm, dNdzs_norm) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 1000)
	#norm_zs = scipy.integrate.simps(dNdzs_norm, zs_norm)
	
	# First, do the first integral in spectroscopic redshift over chiext->inf. Use a zph vector that is the longest possible necessary one for this to avoid having to keep track of three things.
	zph_long = np.linspace(0., pa.zphmax, 2000)
	zs_int = np.zeros((len(zph_long), len(chiLext)))
	#zs_int_clkk = np.zeros(len(chiLext))
	for ci in range(0,len(chiLext)):
		(zs, dNdzs) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zLext[ci], pa.zsmax, 1000)
		chis = com_of_z(zs)
		#zs_int_clkk[ci] = scipy.integrate.simps(dNdzs / norm_zs * (chis - chiLext[ci]) / chis, zs)
		for zpi in range(0,len(zph_long)):
			pz = setup.p_z(zph_long[zpi], zs, pa.pzpar_fid, pa.pztype)
			zs_int[zpi, ci] = scipy.integrate.simps(pz * dNdzs * (chis - chiLext[ci]) / chis, zs)
	
	# Now do the first integral in zph. Here we will integrate over a smaller zph vector than the one we just used
	zph_int = np.zeros((len(zlens), len(chiLext)))
	zph_shorter = [0]* len(zlens)
	for zi in range(0,len(zlens)):
		# Get zph over a shorter region and the integral in zs over that region.
		
		if (SAMPLE=='assocBl'):
			if (pa.delta_z<zlens[zi]):
				zphmin = zlens[zi] - pa.delta_z
			else:
				zphmin = 0.
			zphmax = zlens[zi] + pa.delta_z
		elif (SAMPLE=='B'):
			zphmin = zlens[zi] + pa.delta_z; zphmax = pa.zphmax
		else:
			print "We don't have support for that sample; exiting."
			exit()
			
		zph_shorter[zi] = np.linspace(zphmin, zphmax, 1000)
		for ci in range(0,len(chiLext)):
			interp_zph = scipy.interpolate.interp1d(zph_long, zs_int[:, ci])
			zs_int_zphvec = interp_zph(zph_shorter[zi])
			
			zph_int[zi, ci] = scipy.integrate.simps( zs_int_zphvec, zph_shorter[zi])
			
	# Get the normalization of dNdzph
	zph_norm = np.zeros(len(zlens))
	for zi in range(0, len(zlens)):
		zs_int_for_norm = np.zeros(len(zph_shorter[zi]))
		for zpi in range(0,len(zph_shorter[zi])):
			pz = setup.p_z(zph_shorter[zi][zpi], zs_norm, pa.pzpar_fid, pa.pztype)
			zs_int_for_norm[zpi] = scipy.integrate.simps(pz * dNdzs_norm , zs_norm)
		zph_norm[zi] = scipy.integrate.simps(zs_int_for_norm, zph_shorter[zi])
		
	# Now do the first integral in zl
	zl_int = np.zeros(len(chiLext))
	for ci in range(0,len(chiLext)):
		zl_int[ci] = scipy.integrate.simps(dndzl * zph_int[:, ci] / zph_norm, zlens)
		
	# This terms gets squared because the exact same thing shows up twice here, then we integrate over chiLext.
	int_kk = np.zeros(len(lvec_less))
	#clkk = np.zeros(len(lvec_less))
	for li in range(0,len(lvec_less)):
		int_kk[li] = (9. / 4.) * H0**4 * scipy.integrate.simps( (H/H0)**4 * Omz**2 * Pof_lx[li, :] * (zl_int)**2, chiLext)
		#clkk[li] = (9. / 4.) * H0**4 * scipy.integrate.simps( (H/H0)**4 * Omz**2 * Pof_lx[li, :] * (zs_int_clkk)**2, chiLext)
		
	"""# Compare with CCL
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = pa.A_s, n_s=pa.n_s_cosmo)
	cosmo = ccl.Cosmology(p)
	(zs, dNdzs_unnormed) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 500)
	dNdzs = dNdzs_unnormed / norm_zs
	ltracer = ccl.cls.ClTracerLensing(cosmo=cosmo,has_intrinsic_alignment=False, n=dNdzs, z= zs)
	Clkk_ccl = ccl.cls.angular_cl(cosmo, ltracer, ltracer, lvec_less) 
		
	plt.figure()
	plt.loglog(lvec_less, clkk, 'm+')
	plt.hold(True)
	plt.loglog(lvec_less, Clkk_ccl, 'g+')
	plt.ylim(10**(-20), 10**(-7))
	plt.savefig('./plots/clkk_compare_CCL.pdf')
	plt.close()"""
	
	np.savetxt('./txtfiles/Pkkterm_gamtBl_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt', int_kk)
	
	return int_kk
	
def doints_PggPkk():
	""" This function does the integrals for the <gg><kk> term in the covariance matrix"""
	
	# Get the chi over which we integrate. It's called chiLext because this term deal with a cosmic shear power spectrum so the 'lenses' for this are extended in space.
	chiLext			=		scipy.linspace(chiLext_min, chiLext_max, chiLextpts)	
	H=getHconf(chiLext)
	Omz=getOmMx(chiLext)
	Pof_lx_ext = Pmm_1h2h(chiLext)
	zLext = z_ofchi(chiLext)
	
	# We also need the z vector for the actual lens galaxy distribution
	zlens = np.linspace(pa.zLmin, pa.zLmax, 200)
	dndzl = setup.get_dNdzL(zlens, SURVEY)
	chilens = com_of_z(zlens)
	Pof_lx_lens = Pgg_1h2h(chilens)
	Hlens = getHconf(chilens)
	
	# Get the norm of the spectroscopic redshift source distribution - this is used only for getting clkk and clgg to compare with CCL, for the actual answer we normalize over dNdzph.
	(zs_norm, dNdzs_norm) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 1000)
	norm_zs = scipy.integrate.simps(dNdzs_norm, zs_norm)
	
	# First, do the first integral in spectroscopic redshift over chiext->inf. Use a zph vector that is the longest possible necessary one for this to avoid having to keep track of three things.
	zph_long = np.linspace(0., pa.zphmax, 500)
	zs_int = np.zeros((len(zph_long), len(chiLext)))
	#zs_int_clkk = np.zeros(len(chiLext))
	for ci in range(0,len(chiLext)):
		(zs, dNdzs) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zLext[ci], pa.zsmax, 1000)
		chis = com_of_z(zs)
		#zs_int_clkk[ci] = scipy.integrate.simps(dNdzs/norm_zs * (chis - chiLext[ci]) / chis, zs)
		for zpi in range(0,len(zph_long)):
			pz = setup.p_z(zph_long[zpi], zs, pa.pzpar_fid, pa.pztype)
			zs_int[zpi, ci] = scipy.integrate.simps(pz * dNdzs * (chis - chiLext[ci]) / chis, zs)
	
	# Now do the first integral in zph. Here we will integrate over a smaller zph vector than the one we just used
	zph_int = np.zeros((len(zlens), len(chiLext)))
	zph_shorter = [0] * len(zlens)
	for zi in range(0,len(zlens)):
		# Get zph over a shorter region and the integral in zs over that region.
		if (SAMPLE=='assocBl'):
			if (pa.delta_z<zlens[zi]):
				zphmin = zlens[zi] - pa.delta_z
			else:
				zphmin = 0.
			zphmax = zlens[zi] + pa.delta_z
		elif (SAMPLE=='B'):
			zphmin = zlens[zi] + pa.delta_z; zphmax = pa.zphmax
		else:
			print "We don't have support for that sample; exiting."
			exit()
			
		zph_shorter[zi] = np.linspace(zphmin, zphmax, 1000)
			
		for ci in range(0,len(chiLext)):
			interp_zph = scipy.interpolate.interp1d(zph_long, zs_int[:, ci])
			zs_int_zphvec = interp_zph(zph_shorter[zi])
			
			zph_int[zi, ci] = scipy.integrate.simps( zs_int_zphvec, zph_shorter[zi])
	
	# Get the normalization of dNdzph that we need
	(zs, dNdzs) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 1000)
	zph_norm = np.zeros(len(zlens))
	for zi in range(0, len(zlens)):
		zs_int_for_norm = np.zeros(len(zph_shorter[zi]))
		for zpi in range(0,len(zph_shorter[zi])):
			pz = setup.p_z(zph_shorter[zi][zpi], zs, pa.pzpar_fid, pa.pztype)
			zs_int_for_norm[zpi] = scipy.integrate.simps(pz * dNdzs , zs)
		zph_norm[zi] = scipy.integrate.simps(zs_int_for_norm, zph_shorter[zi])
			
	# Now integrate over the entended chiLext. zph_int gets squared because the exact same thing shows up twice here.
	chiLext_int = np.zeros((len(lvec_less), len(zlens)))
	for li in range(0,len(lvec_less)):
		for zi in range(0,len(zlens)):
			chiLext_int[li, zi] = 9./4. * H0**4 * scipy.integrate.simps( (H/H0)**4 * Omz**2 * Pof_lx_ext[li, :] * zph_int[zi, :]**2 / (zph_norm[zi])**2, chiLext)
	
	# Finally, do the integral over zl		
	# Now do the first integral in zl
	int_kkgg = np.zeros(len(lvec_less))
	#clgg = np.zeros(len(lvec_less))
	#clkk = np.zeros(len(lvec_less))
	for li in range(0,len(lvec_less)):
		int_kkgg[li] = scipy.integrate.simps(dndzl**2 * Hlens * (1. + zlens) * chiLext_int[li, :] * Pof_lx_lens[li, :] / chilens**2, zlens)
		#clgg[li] = scipy.integrate.simps(dndzl**2 * Hlens * (1. + zlens) * Pof_lx_lens[li, :] / chilens**2, zlens)
		#clkk[li] = 9./4. * H0**4 * scipy.integrate.simps( (H/H0)**4 * Omz**2 * Pof_lx_ext[li, :] * zs_int_clkk**2, chiLext)
	
	np.savetxt(folderpath+outputfolder+'/PggPkkterm_gamtBl_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt', int_kkgg)
	
	"""# Compare to CCL	
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = pa.A_s, n_s=pa.n_s_cosmo)
	cosmo = ccl.Cosmology(p)
	gtracer = ccl.cls.ClTracerNumberCounts(cosmo = cosmo, has_rsd = False, has_magnification = False, n = dndzl, bias = pa.bd * np.ones(len(zlens)), z = zlens)
	Clgg_ccl = ccl.cls.angular_cl(cosmo, gtracer, gtracer, lvec_less)  
	dndz = dNdzs_norm / norm_zs
	ltracer = ccl.cls.ClTracerLensing(cosmo=cosmo,has_intrinsic_alignment=False, n=dndz, z= zs_norm)
	Clkk_ccl = ccl.cls.angular_cl(cosmo, ltracer, ltracer, lvec_less) 
	
		
	plt.figure()
	plt.loglog(lvec_less, clgg, 'm+')
	plt.hold(True)
	plt.loglog(lvec_less, Clgg_ccl, 'g+')
	#plt.ylim(10**(-11), 10**(-3))
	plt.savefig('./plots/clgg_compare_CCL_ggkk.pdf')
	plt.close()
	
	plt.figure()
	plt.loglog(lvec_less, clkk, 'm+')
	plt.hold(True)
	plt.loglog(lvec_less, Clkk_ccl, 'g+')
	#plt.ylim(10**(-11), 10**(-3))
	plt.savefig('./plots/clkk_compare_CCL_ggkk.pdf')
	plt.close()
	
	
	plt.figure()
	plt.loglog(lvec_less, int_kkgg)
	plt.savefig('./plots/int_kkgg_extl.pdf')
	plt.close()"""
	
	return int_kkgg
	
def doconstint():
	""" This function does the integrals for the constant term """
	
	# Define redshift vector for lenses
	zLvec = np.linspace(pa.zLmin, pa.zLmax, 200)
	dndzl = setup.get_dNdzL(zLvec, SURVEY)
	
	# Integrate answer over dndzl, including over ns for each zl. This squared will be the value we need (because the integrals in primed and unprimed quantities are exactly symmetric).
	ns = get_ns_partial(zLvec)
	
	ns_avg = scipy.integrate.simps(ns * dndzl, zLvec)
	
	print "ns_avg=", ns_avg / (3600.*3282.8)
	
	if (SAMPLE=='assocBl'):
		gam = pa.e_rms_Bl_a
	elif (SAMPLE=='B'):
		gam = pa.e_rms_Bl_b
	else:
		print "We do not have support for that sample. Exiting."
		exit()
	
	
	save=[0]
	save[0]= gam ** 2 / nl / ns_avg
	
	np.savetxt('./txtfiles/const_gamtBl_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt', save)
	
	return
	
def get_lint():
	""" Gets the integral over ell at each R and R' """
	Pgkterm		=	np.loadtxt('./txtfiles/Pgkterm_gamtBl_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt')
	PggPkkterm	=	np.loadtxt('./txtfiles/PggPkkterm_gamtBl_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt')
	Pkkterm		= 	np.loadtxt('./txtfiles/Pkkterm_gamtBl_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt')
	Pggterm		=	np.loadtxt('./txtfiles/Pggterm_gamtBl_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt')
	constterm	=	np.loadtxt('./txtfiles/const_gamtBl_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt')
	
	if (SAMPLE=='assocBl'):
		gam = pa.e_rms_Bl_a
	elif (SAMPLE=='B'):
		gam = pa.e_rms_Bl_b
	else:
		print "We do not have support for that sample. Exiting."
		exit()
	
	"""# plot each thing to see what is dominating:
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
	plt.savefig('./plots/compareterms_powerspectra_alone_extl_survey='+SURVEY+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.pdf')
	plt.close()"""
	
	
	# plot each thing to see what is dominating:
	plt.figure()
	plt.loglog(lvec_less, Pgkterm, 'b+', label='$\propto (C_{g \kappa})^2$')
	plt.hold(True)
	plt.loglog(lvec_less, PggPkkterm, 'r+', label='$\propto (C_{gg} C_{\kappa \kappa})$')
	plt.hold(True)
	plt.loglog(lvec_less, Pkkterm / nl, 'm+', label='$\propto C_{\kappa \kappa} / n_l$')
	plt.hold(True)
	plt.loglog(lvec_less, Pggterm*gam**2 , 'g+', label='$\propto C_{gg} \gamma^2 / n_s$')
	plt.hold(True)
	plt.loglog(lvec_less, constterm * np.ones(len(lvec_less)), 'k+', label='$\gamma^2 / (n_l n_s)$')
	plt.hold(True)
	plt.loglog(lvec_less, ( Pgkterm + PggPkkterm + Pkkterm/nl + Pggterm*gam**2 + constterm), 'y+', label='tot')
	plt.ylim(10**(-16), 10**(-10))
	plt.ylabel('Contributions to covariance')
	plt.xlabel('$l$')
	plt.title('Survey='+SURVEY+', sample='+SAMPLE)
	plt.legend()
	plt.savefig('./plots/compareterms_gamtBl_extl_survey='+SURVEY+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.pdf')
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
	
def add_shape_noise(i_Rbin, j_Rbin, ravg):
        """ Adds the shape noise term to the diagonal elements """

        wbar = np.loadtxt('./txtfiles/wbar_extl'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt')

        if (i_Rbin != j_Rbin):
                shapenoise_alone = 0.
        else:
                constterm      =       np.loadtxt('./txtfiles/const_DeltaSig_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'.txt')
                Rlowind_bini    =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[i_Rbin])
                Rhighind_bini   =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[i_Rbin+1])
                Rlowind_binj    =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[j_Rbin])
                Rhighind_binj   =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[j_Rbin+1])

                shapenoise_alone = constterm*chiLmean**2 / (4.*np.pi**2*fsky*wbar**2 * (Rvec[Rhighind_bini]**2 - Rvec[Rlowind_bini]**2))

        return (shapenoise_alone)


##########################################################################################
####################################### SET UP ###########################################
##########################################################################################

SURVEY = 'LSST_DESI'
SAMPLE = 'assocBl'
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
nl			=	pa.n_l * 3282.8 # n_l is in # / square degree, numerical factor converts to # / steradian
bias		=	pa.bd

# Sources:
ns_tot			=	pa.n_s * 3600.*3282.8 # n_s is in # / sqamin, numerical factor converts to / steraidan
fsky			=	pa.fsky
	
#Vector set up
src_spec_pts	=	200
src_ph_pts		=	200
Rpts			=	1500
Rmin			=	pa.rp_min
Rmax			=	pa.rp_max
lpts			=	10**6
lpts_less		=	500
lmin			=	3
lmax			=	10**6
numRbins		=	pa.N_bins
chiLext_min		=	0.001
chiLext_max		=	setup.com(pa.zphmax, SURVEY)
chiLextpts		=	250

##########################################################################################
################################  MAIN FUNCTION CALLS ####################################
##########################################################################################

a = time.time()

# Set up
(lvec, lvec_less, Rvec, Redges, Rcentres)					= 		setup_vectors()
z_ofchi, com_of_z								=		setup.z_interpof_com(SURVEY)
(z_spec, dNdz_spec)								= 		setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, src_spec_pts)

chiLmean = com_of_z(pa.zeff)

# Do the integrals on each term up to the l integral (so chiS, bchiS, chiL, bchiL)

#doints_Pgg()
#print "Done with Pgg integrals. Now do Pgk:"
#Pgkints = doints_Pgk() 
#print "Done with Pgk integrals. Now do Pkk:"
#Pkkints = doints_Pkk()
print "Done with Pkk integrals. Now do constant:"
constterm = doconstint() 

exit()
print "Done with constant integrals. Now do PggPkk:"
PggPkkints = doints_PggPkk()
print "Done with PggPkk integrals. Now doing integrals over R:"

# First, get the l integral in terms of R and R'. This is the long part, and needs only to be done once instead of over and over for each bin.
lint = get_lint()
#This must be done for each set of bins
cov_shapenoise_alone		=	np.zeros((numRbins, numRbins))

for i_R in range(0, numRbins):
	for j_R in range(0, numRbins):
		print "i bin=", i_R, "j bin=", j_R
		#covariance[i_R, j_R]	=	do_outsideints_SigR(i_R, j_R, lint)
		cov_shapenoise_alone[i_R, j_R]    =       add_shape_noise(i_R, j_R, 0.)
		
#np.savetxt(folderpath+outputfolder+'/cov_Upgm_'+endfilename+'_sample='+SAMPLE+'_rpts'+str(Rpts)+'_lpts'+str(lpts)+'_TEST_ELL_DOWNSAMPLE.txt', covariance)
np.savetxt('./txtfiles/shapenoiseonly_DelSig_zLext_'+endfilename+'_sample='+SAMPLE+'_rpts'+str(Rpts)+'_lpts'+str(lpts)+'_deltaz='+str(pa.delta_z)+'.txt', cov_shapenoise_alone)

print '\nTime for completion:', '%.1f' % (time.time() - a), 'seconds'
