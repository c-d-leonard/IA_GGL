""" This script computes the covariance matrix of DeltaSigma_{gm} in bins in R.
This version assumes an effective redshift for the lenses, parameterized by comoving distance chiLmean."""

endfile= 'fixDls'

print "Delta Sigma LSST A"

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

def get_SigmaC_inv(z_s_, z_l_):
    """ Returns the theoretical value of 1/Sigma_c, (Sigma_c = the critcial surface mass density).
    z_s_ and z_l_ can be 1d arrays, so the returned value will in general be a 2d array. """

    com_s = com_of_z(z_s_) 
    com_l = com_of_z(z_l_) 

    if ((hasattr(z_s_, "__len__")==True) and (hasattr(z_l_, "__len__")==True)):
        Sigma_c_inv = np.zeros((len(z_s_), len(z_l_)))
        for zsi in range(0,len(z_s_)):
            for zli in range(0,len(z_l_)):
                # Units are pc^2 / (h Msun), comoving
                if((com_s[zsi] - com_l[zli])<0.):
                    Sigma_c_inv[zsi, zli] = 0.
                else:
                    Sigma_c_inv[zsi, zli] = 4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc *   com_l[zli] * (com_s[zsi] - com_l[zli]) * (1 + z_l_[zli]) / com_s[zsi]
    else: 
        # Units are pc^2 / (h Msun), comoving
        if hasattr(z_s_, "__len__"):
            Sigma_c_inv = np.zeros(len(z_s_))
            for zsi in range(0, len(z_s_)):
                if(com_s[zsi]<=com_l):
                    Sigma_c_inv[zsi] = 0.
                else:
                    Sigma_c_inv[zsi] = 4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc *   com_l * (com_s[zsi] - com_l)* (1 + z_l_)/ com_s[zsi]
        elif hasattr(z_l_, "__len__"): 
            Sigma_c_inv = np.zeros(len(z_l_))
            for zli in range(0,len(z_l_)):
                if(com_s<=com_l[zli]):
                    Sigma_c_inv[zli] = 0.
                else:
                    Sigma_c_inv[zli] = 4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc *   com_l[zli] * (com_s - com_l[zli])* (1 + z_l_[zli]) / com_s
        else:
            if (com_s < com_l):
                Sigma_c_inv=0.
            else:
                 Sigma_c_inv= 4. * np.pi * (pa.Gnewt * pa.Msun) * (10**12 / pa.c**2) / pa.mperMpc *   com_l * (com_s - com_l) * (1 + z_l_) / com_s
                    
    return Sigma_c_inv
		
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
	
def sum_weights(photoz_sample, specz_cut, rp_bins, dNdz_par, pz_par, dNdztype, pztype):
	""" Returns the sum over weights for each projected radial bin. 
	photoz_sample = 'A', 'B', or 'full'
	specz_cut = 'close', or 'nocut'
	"""
	
	# Get lens redshift distribution
	zL = np.linspace(pa.zLmin, pa.zLmax, 100)
	dNdzL = setup.get_dNdzL(zL, SURVEY)
	
	# Loop over lens redshift values
	sum_ans_zph = np.zeros(len(zL))
	for zi in range(0,len(zL)):
		if (photoz_sample == 'A'):
			
			if (specz_cut == 'nocut'):
				(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi], zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
				weight = weights(pa.e_rms_Bl_a, z_ph, zL[zi])
				sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
			elif (specz_cut == 'close'):
				(z_ph, dNdz_ph) = N_of_zph(zL[zi]-z_of_com(pa.close_cut), zL[zi]+z_of_com(pa.close_cut), pa.zsmin, pa.zsmax, zL[zi], zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
				weight = weights(pa.e_rms_Bl_a, z_ph, zL[zi])
				sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
			else:
				print "We do not have support for that spec-z cut. Exiting."
				exit()
			
		elif(photoz_sample == 'B'):
			
			if (specz_cut == 'nocut'):
				(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi] + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
				weight = weights(pa.e_rms_Bl_b, z_ph, zL[zi])
				sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
			elif (specz_cut == 'close'):
				(z_ph, dNdz_ph) = N_of_zph(zL[zi]-z_of_com(pa.close_cut), zL[zi]+z_of_com(pa.close_cut), pa.zsmin, pa.zsmax, zL[zi] + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
				weight = weights(pa.e_rms_Bl_b, z_ph, zL[zi])
				
				sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
			else:
				print "We do not have support for that spec-z cut. Exiting."
				exit()
		
		elif(photoz_sample == 'full'):
			
			if (specz_cut == 'nocut'):
				(z_ph, dNdz_ph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
				weight = weights(pa.e_rms_Bl_full, z_ph, zL[zi])
				
				sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
			elif (specz_cut == 'close'):
				(z_ph, dNdz_ph) = N_of_zph(zL[zi]-z_of_com(pa.close_cut), zL[zi]+z_of_com(pa.close_cut), pa.zsmin, pa.zsmax, pa.zphmin, pa.zphmax, pa.zphmin, pa.zphmax, dNdz_par, pz_par, dNdztype, pztype)
				
				weight = weights(pa.e_rms_Bl_full, z_ph, zL[zi])
				
				sum_ans_zph[zi] = scipy.integrate.simps(dNdz_ph * weight, z_ph)
				
			else:
				print "We do not have support for that spec-z cut. Exiting."
				exit()
		
		else:
			print "We do not have support for that photo-z sample cut. Exiting."
			print photoz_sample
			exit()
	
	# Now sum over lens redshift:
	sum_ans = scipy.integrate.simps(sum_ans_zph * dNdzL, zL)
	
	return sum_ans
	
def weights(e_rms, z_, z_l_):

    """ Returns the inverse variance weights as a function of redshift. """
        
    SigC_t_inv = get_SigmaC_inv(z_, z_l_)

    if ((hasattr(z_, "__len__")==True) and (hasattr(z_l_, "__len__")==True)):
        weights = np.zeros((len(z_), len(z_l_)))
        for zsi in range(0,len(z_)):
            for zli in range(0,len(z_l_)):
                weights[zsi, zli] = SigC_t_inv[zsi, zli]**2/(sigma_e(z_)[zsi]**2 + e_rms**2)
    else:
        if (hasattr(z_, "__len__")):
            weights = SigC_t_inv**2/(sigma_e(z_)**2 + e_rms**2 * np.ones(len(z_)))
        else:
            weights = SigC_t_inv**2/(sigma_e(z_)**2 + e_rms**2 )

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
	
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), sigma8 = pa.sigma8, n_s=pa.n_s_cosmo)
	cosmo = ccl.Cosmology(p)
  
	k = scipy.logspace(-4, 4, 1000)
	P_2h=np.zeros((len(k), len(aivec)))
	h = (pa.HH0/100.)
	for ai in range(0, len(aivec)):
		# Need to be careful about h factors here - ccl calls class which uses Mpc units but we use Mpc/h.
		P_2h[:, ai] = h**3 * ccl.nonlin_matter_power(cosmo, k * h, aivec[ai])
		
	# Now do 1-halo (this is done in a separate function)
	P_1h = ws.get_Pkgm_1halo_kz(k, zivec, y_dm, Mh, kv, SURVEY)
	
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
	
	p = ccl.Parameters(Omega_c = pa.OmC_l, Omega_b = pa.OmB_l, h = (pa.HH0_l/100.), sigma8 = pa.sigma8_l, n_s=pa.n_s_cosmo)
	cosmo = ccl.Cosmology(p)
  
	k = scipy.logspace(-4, 4, 1000)
	P_2h=np.zeros((len(k), len(aivec)))
	h = (pa.HH0/100.)
	for ai in range(0, len(aivec)):
		# Need to be careful about h factors here - ccl calls class which uses Mpc units but we use Mpc/h.
		P_2h[:, ai] = h**3 * ccl.nonlin_matter_power(cosmo, k * h , aivec[ai])
		
	# Now do 1-halo (this is done in a separate function
	P_1h = ws.get_Pkgg_ll_1halo_kz(k, zivec,y_dm, Mh, kv, SURVEY)
	
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
	P_1h = ws.get_Pkmm_1halo_kz(k, zivec, y_dm, Mh, kv, SURVEY)
	
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
	
	ns_samp = np.zeros(len(zL))
	for zi in range(0,len(zL)):
		if (SAMPLE=='A'):
			(z_ph, Nofzph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi], zL[zi] + pa.delta_z, pa.zphmin, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
		elif (SAMPLE=='B'):
			(z_ph, Nofzph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zL[zi] + pa.delta_z, pa.zphmax, pa.zphmin, pa.zphmax, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
		else:
			print "We do not have support for that sample. Exiting."
			exit()
		ns_samp[zi] = ns_tot * scipy.integrate.simps(Nofzph, z_ph)
		
	# Just for testing, get this integrated over the lens redshift. Really, we pass this out as a function of zl
	dndzl = setup.get_dNdzL(zL, SURVEY)
	
	averaged_ns_samp = scipy.integrate.simps(ns_samp * dndzl, zL)

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
	
	# Do the integral over dndzl. This includes the 1/ ns term that goes into this term, because it needs integrating over the lens distribution.
	ns = get_ns_partial(zL)
	int_gg = np.zeros(len(lvec_less))
	for li in range(0,len(lvec_less)):
		int_gg[li] = scipy.integrate.simps( dndzl**2 * H * Pdelta[li, :] / (chi**2 * ns), zL)
			
	np.savetxt('./txtfiles/Pgg/Pggterm_DeltaSig_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', int_gg)
	
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
	(zsnorm, dNdzsnorm) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 500, SURVEY)
	#normzs = scipy.integrate.simps(dNdzsnorm, zsnorm)
	
	int_in_zl = np.zeros(len(zL))
	z_ph = [0] * len(zL)
	for zi in range(0, len(zL)):
		# Define the photo-z vector:
		if (SAMPLE=='A'):
			z_ph[zi] = np.linspace(zL[zi], zL[zi]+pa.delta_z, 500)
		elif (SAMPLE=='B'):
			z_ph[zi] = np.linspace(zL[zi] + pa.delta_z, pa.zphmax, 500)
		
		# Get the integral over spec z
		(zs, dNdzs) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zL[zi], pa.zsmax, 500, SURVEY)
		int_in_zs = np.zeros(len(z_ph[zi]))
		for zpi in range(0,len(z_ph[zi])):
			pz = setup.p_z(z_ph[zi][zpi], zs, pa.pzpar_fid, pa.pztype)
			int_in_zs[zpi] = scipy.integrate.simps(pz*dNdzs*(com_of_z(zs) - chi[zi])/(com_of_z(zs)), zs)
			
		# Get the integral over photo z
		int_in_zl[zi] = scipy.integrate.simps(int_in_zs, z_ph[zi])
		
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
	for li in range(0,len(lvec_less)):
		int_gk[li] = H0**2 * scipy.integrate.simps(1.5 * dndzl * Omz * (H / H0)**2 * Pdelta[li, :] * int_in_zl / (chi * zphnorm), zL)
	
	np.savetxt('./txtfiles/Pgk/Pgkterm_DeltaSig_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', int_gk**2 )
		
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
	(zs_norm, dNdzs_norm) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 1000, SURVEY)
	#norm_zs = scipy.integrate.simps(dNdzs_norm, zs_norm)
	
	# First, do the first integral in spectroscopic redshift over chiext->inf. Use a zph vector that is the longest possible necessary one for this to avoid having to keep track of three things.
	zph_long = np.linspace(pa.zLmin, pa.zphmax, 2000)
	zs_int = np.zeros((len(zph_long), len(chiLext)))
	#zs_int_clkk = np.zeros(len(chiLext))
	for ci in range(0,len(chiLext)):
		(zs, dNdzs) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zLext[ci], pa.zsmax, 1000, SURVEY)
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
		if (SAMPLE=='A'):
			zphmin = zlens[zi]; zphmax = zlens[zi] + pa.delta_z
		elif (SAMPLE=='B'):
			zphmin = zlens[zi] + pa.delta_z; zphmax = pa.zphmax
		else:
			print "We don't have support for that sample; exiting."
			exit()
			
		zph_shorter[zi] = np.linspace(zphmin, zphmax, 1000)
		for ci in range(0,len(chiLext)):
			interp_zph = scipy.interpolate.interp1d(zph_long, zs_int[:, ci])
			zs_int_zphvec = interp_zph(zph_shorter[zi])
		
			zph_int[zi, ci] = scipy.integrate.simps( zs_int_zphvec, zph_shorter[zi] )
			
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
	
	np.savetxt('./txtfiles/Pkk/Pkkterm_DeltaSig_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', int_kk)
	
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
	(zs_norm, dNdzs_norm) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 1000, SURVEY)
	#norm_zs = scipy.integrate.simps(dNdzs_norm, zs_norm)
	
	# First, do the first integral in spectroscopic redshift over chiext->inf. Use a zph vector that is the longest possible necessary one for this to avoid having to keep track of three things.
	zph_long = np.linspace(pa.zLmin, pa.zphmax, 500)
	zs_int = np.zeros((len(zph_long), len(chiLext)))
	for ci in range(0,len(chiLext)):
		(zs, dNdzs) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, zLext[ci], pa.zsmax, 1000, SURVEY)
		chis = com_of_z(zs)
		for zpi in range(0,len(zph_long)):
			pz = setup.p_z(zph_long[zpi], zs, pa.pzpar_fid, pa.pztype)
			zs_int[zpi, ci] = scipy.integrate.simps(pz * dNdzs * (chis - chiLext[ci]) / chis, zs)
	
	# Now do the first integral in zph. Here we will integrate over a smaller zph vector than the one we just used
	zph_int = np.zeros((len(zlens), len(chiLext)))
	zph_shorter = [0] * len(zlens)
	for zi in range(0,len(zlens)):
		# Get zph over a shorter region and the integral in zs over that region.
		if (SAMPLE=='A'):
			zphmin = zlens[zi]; zphmax = zlens[zi] + pa.delta_z
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
	(zs, dNdzs) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 1000, SURVEY)
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
	
	int_kkgg = np.zeros(len(lvec_less))
	for li in range(0,len(lvec_less)):
		int_kkgg[li] = scipy.integrate.simps(dndzl**2 * Hlens * (1. + zlens) * chiLext_int[li, :] * Pof_lx_lens[li, :] / chilens**2, zlens)
	
	np.savetxt(folderpath+outputfolder+'/PggPkk/PggPkkterm_DeltaSig_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', int_kkgg)
	
	return int_kkgg

def SigCsq_avg():
	""" Get Sigma C squared, averaged."""
	
	zLvec = np.linspace(pa.zLmin, pa.zLmax, 1000)
	dndzl = setup.get_dNdzL(zLvec, SURVEY)
	
	chiSans = np.zeros(len(zLvec))
	for zi in range(0,len(zLvec)):
		
		# Get zph, dNdzph: 
		if (SAMPLE=='A'):
			(z_ph, Nofzph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zLvec[zi], zLvec[zi] + pa.delta_z, zLvec[zi], zLvec[zi] + pa.delta_z, pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
			gam = pa.e_rms_Bl_a
		elif (SAMPLE =='B'):
			(z_ph, Nofzph) = N_of_zph(pa.zsmin, pa.zsmax, pa.zsmin, pa.zsmax, zLvec[zi] + pa.delta_z, pa.zphmax, zLvec[zi] + pa.delta_z, pa.zphmax,  pa.dNdzpar_fid, pa.pzpar_fid, pa.dNdztype, pa.pztype)
			gam = pa. e_rms_Bl_b
		else:
			print "We do not have support for that sample. Exiting."
			exit()
		
		Sigma_inv = get_SigmaC_inv(z_ph, zLvec[zi])
		chiSans[zi] = scipy.integrate.simps(Sigma_inv**2 * Nofzph, z_ph)
		
	chiSint_zl = scipy.integrate.simps(chiSans * dndzl, zLvec)
	
	save_SigC2 = [1./chiSint_zl]
	np.savetxt(folderpath+outputfolder+'/SigCsq/SigCsqterm_DeltaSig_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', save_SigC2)
	
	return save_SigC2
	
def doconstint():
	""" This function does the integrals for the constant term """
	
	# Define redshift vector for lenses
	zLvec = np.linspace(pa.zLmin, pa.zLmax, 1000)
	dndzl = setup.get_dNdzL(zLvec, SURVEY)
	
	# Integrate answer over dndzl, including over ns for each zl. This squared will be the value we need (because the integrals in primed and unprimed quantities are exactly symmetric).
	ns = get_ns_partial(zLvec)
	ns_avg = scipy.integrate.simps(ns * dndzl, zLvec)
	
	if (SAMPLE=='A'):
		gam = pa.e_rms_Bl_a
	elif (SAMPLE =='B'):
		gam = pa. e_rms_Bl_b
	else:
		print "We do not have support for that sample. Exiting."
		exit()
	
	save=[0]
	save[0]= gam ** 2 / nl / ns_avg 
	
	np.savetxt('./txtfiles/const/const_DeltaSig_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt', save)
	
	return
	
def get_lint():
	""" Gets the integral over ell at each R and R' """
	Pgkterm		=	np.loadtxt('./txtfiles/Pgk/Pgkterm_DeltaSig_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')
	PggPkkterm	=	np.loadtxt('./txtfiles/PggPkk/PggPkkterm_DeltaSig_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')
	Pkkterm		= 	np.loadtxt('./txtfiles/Pkk/Pkkterm_DeltaSig_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')
	Pggterm		=	np.loadtxt('./txtfiles/Pgg/Pggterm_DeltaSig_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')
	constterm	=	np.loadtxt('./txtfiles/const/const_DeltaSig_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')
	SigCterm 	=	np.loadtxt('./txtfiles/SigCsq/SigCsqterm_DeltaSig_extl_'+endfilename+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')
	
	if (SAMPLE=='A'):
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
	plt.loglog(lvec_less, SigCterm * Pgkterm, 'b+', label='$\propto (C_{g \kappa})^2$')
	plt.hold(True)
	plt.loglog(lvec_less, SigCterm * PggPkkterm, 'r+', label='$\propto (C_{gg} C_{\kappa \kappa})$')
	plt.hold(True)
	plt.loglog(lvec_less, SigCterm * Pkkterm / nl, 'm+', label='$\propto C_{\kappa \kappa} / n_l$')
	plt.hold(True)
	plt.loglog(lvec_less, SigCterm * Pggterm*gam**2 , 'g+', label='$\propto C_{gg} \gamma^2 / n_s$')
	plt.hold(True)
	plt.loglog(lvec_less, SigCterm * constterm * np.ones(len(lvec_less)), 'k+', label='$\gamma^2 / (n_l n_s)$')
	plt.hold(True)
	plt.loglog(lvec_less, SigCterm * ( Pgkterm + PggPkkterm + Pkkterm/nl + Pggterm*gam**2 + constterm), 'y+', label='tot')
	#plt.ylim(10**(-22), 10**(-14))
	plt.ylabel('Contributions to covariance')
	plt.xlabel('$l$')
	plt.title('Survey='+SURVEY+', sample='+SAMPLE)
	plt.legend()
	plt.savefig('./plots/compareterms_DeltaSigcov_extl_survey='+SURVEY+'_sample='+SAMPLE+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.pdf')
	plt.close()
	
	return 


##########################################################################################
####################################### SET UP ###########################################
##########################################################################################

SURVEY = 'LSST_DESI'
SAMPLE = 'A'
# Import the parameter file:
if (SURVEY=='SDSS'):
	import params as pa
elif (SURVEY=='LSST_DESI'):
	import params_LSST_DESI as pa
else:
	print "We don't have support for that survey yet; exiting."
	exit()

# Cosmological parameters from parameters file
Nnu	= pa.Nnu; HH0 =	pa.HH0; OmegaR = pa.OmR; OmegaN	= pa.OmN; OmegaB =	pa.OmB; OmegaM = pa.OmC; OmegaK	= 0.0; h =	HH0/100.
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
chiLext_max		=	setup.com(pa.zphmax, SURVEY, pa.cos_par_std)
chiLextpts		=	250

##########################################################################################
################################  MAIN FUNCTION CALLS ####################################
##########################################################################################


# Set up
(lvec, lvec_less, Rvec, Redges, Rcentres)					= 		setup_vectors()
z_ofchi, com_of_z								=		setup.z_interpof_com(SURVEY)
(z_spec, dNdz_spec)								= 		setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, src_spec_pts, SURVEY)

chiLmean = com_of_z(pa.zeff)

# Do the integrals on each term up to the l integral (so chiS, bchiS, chiL, bchiL)

"""print "getting y"
# For getting the Fourier transformed density profile
Mh = np.logspace(7., 16., 30)
kv = np.logspace(-4, 4, 40)
y_dm = ws.gety_ldm(Mh, kv, SURVEY)
print "got y"

doints_Pgg()
print "Done with Pgg integrals. Now do Pgk:"
Pgkints = doints_Pgk() 
print "Done with Pgk integrals. Now do Pkk:"
Pkkints = doints_Pkk() 
print "Done with Pkk integrals. Now do constant:"
constterm = doconstint() 
print "Done with constant integrals. Now do PggPkk:"
PggPkkints = doints_PggPkk()"""
print "Done with PggPkk integrals. Getting SigmaC^2 avg"
SigC2_avg = SigCsq_avg()
print "Done getting SigC term."


# First, get the l integral in terms of R and R'. This is the long part, and needs only to be done once instead of over and over for each bin.
lint = get_lint()

