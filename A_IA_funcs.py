# Functions needed to get A_IA as a function of lumiosity for a given luminosity function
__author__= "Danielle Leonard"

import numpy as np
import scipy.integrate
import scipy.interpolate
import pyccl as ccl

def get_Ai(A0, beta, cosmo, dndz, Mr_s=-20.70, Q=1.23, alpha_lum=-1.23, phi_0=0.0094, P=-0.3, mlim=25.3, Lp= 1.):
	""" Get the amplitude of the 2-halo part of w_{l+}
	A0 is the amplitude, beta is the power law exponent (see Krause et al. 2016) 
	cosmo is a CCL cosmology object 
	Lp is a pivot luminosity (default = 1)
	dndz is (z, dNdz) - vector of z and dNdz for the galaxies
	(does not need to be normalised) 
	"""
		
	z_input, dNdz = dndz
	
	# Don't evaluate at any redshifts higher than the highest value for which we have kcorr and ecorr corrections or lower than lowest
	# These high (>3) and low (<0.02) redshifts shouldn't matter anyway.
	(z_k, kcorr, x,x,x) = np.loadtxt('./txtfiles/kcorr.dat', unpack=True)
	(z_e, ecorr, x,x,x) = np.loadtxt('./txtfiles/ecorr.dat', unpack=True)
	zmaxke = min(max(z_k), max(z_e))
	zminke = max(min(z_k), min(z_e))
	if (zminke>min(z_input) and zmaxke<max(z_input)):
		z = np.linspace(zminke, zmaxke, 1000)
		interp_dNdz = scipy.interpolate.interp1d(z_input, dNdz)
		dNdz = interp_dNdz(z)    	
	elif (zmaxke<max(z_input)):
		z = np.linspace(min(z_input), zmaxke, 1000)
		interp_dNdz = scipy.interpolate.interp1d(z_input, dNdz)
		dNdz = interp_dNdz(z)
	elif (zminke>min(z_input)):
		z = np.linspace(zminke, max(z_input), 1000)
		interp_dNdz = scipy.interpolate.interp1d(z_input, dNdz)
		dNdz = interp_dNdz(z)
	else:
		z = z_input
		
	# Get the luminosity function
	(L, phi_normed) = get_phi(z, cosmo, Mr_s, Q, alpha_lum, phi_0, P, mlim)
	# Pivot luminosity:
	Lp = 1.
	
	# Get Ai as a function of lens redshift.
	Ai_ofzl = np.zeros(len(z))
	for zi in range(len(z)):
		Ai_ofzl[zi] = scipy.integrate.simps(np.asarray(phi_normed[zi]) * A0 * (np.asarray(L[zi]) / Lp)**(beta), np.asarray(L[zi]))
	
	# Integrate over dNdz	
	Ai = scipy.integrate.simps(Ai_ofzl * dNdz, z) / scipy.integrate.simps(dNdz, z)
	
	return Ai
	
def get_phi(z, cosmo, Mr_s, Q, alpha_lum, phi_0, P, mlim, Mp=-22.):
	
	""" This function outputs the Schechter luminosity function with parameters fit in Loveday 2012, following the same procedure as Krause et al. 2015, as a function of z and L 
	The output is L[z][l], list of vectors of luminosity values in z, different at each z due to the different lower luminosity limit, and phi[z][l], a list of luminosity functions at these luminosity vectors, at each z
	cosmo is a CCL cosmology object
	mlim is the magnitude limit of the survey
	Mp is the pivot absolute magnitude.
	other parameteres are the parameters of the luminosity function that are different for different samples, e.g. red vs all. lumparams = [Mr_s, Q, alpha_lum, phi_0, P]
	Note that the luminosity function is output normalized (appropriate for getting Ai)."""
	
	# Get the amplitude of the Schechter luminosity function as a function of redshift.
	phi_s = phi_0 * 10.**(0.4 * P * z)
	
	# Get M_* (magnitude), then convert to L_*
	Ms = Mr_s - Q * (z - 0.1)
	Ls = 10**(-0.4 * (Ms - Mp))
	
	# Import the kcorr and ecorr correction from Poggianti (assumes elliptical galaxies)
	# No data for sources beyon z = 3, so we keep the same value at higher z as z=3
	(z_k, kcorr, x,x,x) = np.loadtxt('./txtfiles/kcorr.dat', unpack=True)
	(z_e, ecorr, x,x,x) = np.loadtxt('./txtfiles/ecorr.dat', unpack=True)
	kcorr_interp = scipy.interpolate.interp1d(z_k, kcorr)
	ecorr_interp = scipy.interpolate.interp1d(z_e, ecorr)
	kcorr = kcorr_interp(z)
	ecorr = ecorr_interp(z)
	
	# Get the absolute magnitude and luminosity corresponding to limiting apparent magntiude (as a function of z)
	#dl = com(z, survey, pa.cos_par_std) * (1. + z)
	dl = ccl.luminosity_distance(cosmo, 1./(1.+z))
	Mlim = mlim - (5. * np.log10(dl) + 25. + kcorr + ecorr)
	Llim = 10.**(-0.4 * (Mlim-Mp))
	
	# Get the luminosity vectors - there will be a list of these, one for each redshift, because the limiting values are z-dependent
	L = [0] * len(z)
	for zi in range(0,len(z)):
		L[zi] = scipy.logspace(np.log10(Llim[zi]), 2., 1000)
		
	# Now get phi(L,z), where this exists for each z because the lenghts of the L vectors are different.
	phi_func = [0]*len(z)
	for zi in range(0,len(z)):
		phi_func[zi]= np.zeros(len(L[zi]))
		for li in range(0,len(L[zi])):
			phi_func[zi][li] = phi_s[zi] * (L[zi][li] / Ls[zi]) ** (alpha_lum) * np.exp(- L[zi][li] / Ls[zi])
			
	# Get the normalization in L as a function of z
	
	norm= np.zeros(len(z))
	phi_func_normed = [0]*len(z)
	for zi in range(len(z)):
		norm[zi] = scipy.integrate.simps(phi_func[zi], L[zi])
		phi_func_normed[zi] = phi_func[zi] / norm[zi]

	return (L, phi_func_normed)
	
	


