# This file contains setup-type functions which are shared between the method from Blazek et al 2012 and the shape measurement method.

import numpy as np
import scipy.interpolate
import scipy.integrate
import matplotlib.pyplot as plt

# Functions to set up the rp bins

def setup_rp_bins(rmin, rmax, nbins):
	""" Sets up the bins of projected radius (in units Mpc/h) """

	bins = scipy.logspace(np.log10(rmin), np.log10(rmax), nbins+1)

	return bins

def rp_bins_mid(rp_edges):
	""" Gets the middle of each projected radius bin."""

	logedges=np.log10(rp_edges)
	bin_centers=np.zeros(len(rp_edges)-1)
	for ri in range(0,len(rp_edges)-1):
		bin_centers[ri]    =       10**((logedges[ri+1] - logedges[ri])/2. +logedges[ri])

	return bin_centers
	
def average_in_bins(F_, R_, Rp_):
	""" This function takes a function F_ of projected radius evaluated at projected radial points R_ and outputs the averaged values in bins with edges Rp_""" 
	
	F_binned = np.zeros(len(Rp_)-1)
	for iR in range(0, len(Rp_)-1):
		indlow=next(j[0] for j in enumerate(R_) if j[1]>=(Rp_[iR]))
		indhigh=next(j[0] for j in enumerate(R_) if j[1]>=(Rp_[iR+1]))
		
		F_binned[iR] = scipy.integrate.simps(F_[indlow:indhigh], R_[indlow:indhigh]) / (R_[indhigh] - R_[indlow])
		
	return F_binned
	
def get_areas(bins, z_eff, survey):
	"""Gets the area of each projected radial bin, in square arcminutes. z_eff = effective lens redshift. """

	# Areas in units (Mpc/h)^2
	areas_mpch = np.zeros(len(bins)-1)
	for i in range(0, len(bins)-1):
		areas_mpch[i] = np.pi * (bins[i+1]**2 - bins[i]**2) 

	#Comoving distance out to effective lens redshift in Mpc/h
	chi_eff = com(z_eff, survey)

	# Areas in square arcminutes (466560000 / pi = sqAM in a sphere)
	areas_sqAM = areas_mpch * (466560000. / np.pi) / (4 * np.pi * chi_eff**2)

	return areas_sqAM
	
# Redshift / comoving distance-related functions 

def get_z_close(z_l, cut_MPc_h, survey):
	""" Gets the z above z_l which is the highest z at which we expect IA to be present for that lens. cut_Mpc_h is that separation in Mpc/h."""

	com_l = com(z_l, survey) # Comoving distance to z_l, in Mpc/h

	tot_com_high = com_l + cut_MPc_h
	tot_com_low = com_l - cut_MPc_h
	
	z_of_com, com_of_z = z_interpof_com(survey)

	# Convert tot_com back to a redshift.

	z_cl_high = z_of_com(tot_com_high)
	z_cl_low = z_of_com(tot_com_low)

	return (z_cl_high, z_cl_low)

def com(z_, survey):
	""" Gets the comoving distance in units of Mpc/h at a given redshift, z_ (assuming the cosmology defined in the params file). """
	
	# NO DEPENDENCE ON Z_L

	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()

	OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
	def chi_int(z):
	 	return 1. / (pa.H0 * ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )**(0.5))

	if hasattr(z_, "__len__"):
		chi=np.zeros((len(z_)))
		for zi in range(0,len(z_)):
			#print "zi in com=", zi
			chi[zi] = scipy.integrate.quad(chi_int,0,z_[zi])[0]
	else:
		chi = scipy.integrate.quad(chi_int, 0, z_)[0]

	return chi

def z_interpof_com(survey):
	""" Returns an interpolating function which can give z as a function of comoving distance. """

	# NO DEPENDENCE ON Z_L

	z_vec = scipy.linspace(0., 10., 10000) # This hardcodes that we don't care about anything over z=2100

	com_vec = com(z_vec, survey)

	z_of_com = scipy.interpolate.interp1d(com_vec, z_vec)
	com_of_z =  scipy.interpolate.interp1d(z_vec, com_vec)

	return	(z_of_com, com_of_z)
	
def p_z(z_ph, z_sp, pzpar, pztype):
	""" Returns the probability of finding a photometric redshift z_ph given that the true redshift is z_sp. """
	
	if (pztype == 'Gaussian'):
		sigz = pzpar[0]
		p_z_ = np.exp(-(z_ph - z_sp)**2 / (2.*(sigz*(1.+z_sp))**2)) / (np.sqrt(2.*np.pi)*(sigz*(1.+z_sp)))
	else:
		print "Photo-z probability distribution "+str(pztype)+" not yet supported; exiting."
		exit()
		
	return p_z_
	
def get_NofZ_unnormed(dNdzpar, dNdztype, z_min, z_max, zpts):
	""" Returns the dNdz of the sources as a function of photometric redshift, as well as the z points at which it is evaluated."""

	z = scipy.linspace(z_min, z_max, zpts)
	
	if (dNdztype == 'Nakajima'):
		# dNdz takes form like in Nakajima et al. 2011 equation 3
		a = dNdzpar[0]
		zs = dNdzpar[1]
	
		nofz_ = (z / zs)**(a-1.) * np.exp( -0.5 * (z / zs)**2)
	elif (dNdztype == 'Smail'):
		# dNdz take form like in Smail et al. 1994
		alpha = dNdzpar[0]
		z0 = dNdzpar[1]
		beta = dNdzpar[2]
		nofz_ = z**alpha * np.exp( - (z / z0)**beta)
	else:
		print "dNdz type "+str(dNdztype)+" not yet supported; exiting."
		exit()

	return (z, nofz_)
	
def get_dNdzL(zvec, survey):
	""" Imports the lens redshift distribution from file, normalizes, interpolates, and outputs at the z vector that's passed."""
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
		
	z, dNdz = np.loadtxt('./txtfiles/'+pa.dNdzL_file, unpack=True)
	
	interpolation = scipy.interpolate.interp1d(z, dNdz)
	
	# Create a well-sampled redshift vector to make sure we can get the normalization without numerical problems
	z_highres = np.linspace(z[0], z[-1], 1000)
	
	dNdz_getnorm = interpolation(z_highres)
	
	norm = scipy.integrate.simps(dNdz_getnorm, z_highres)
	
	if ((zvec[0]>=z[0]) and (zvec[-1]<=z[-1])):
		dNdz_return = interpolation(zvec)
	else:
		print "You have asked for dN/dzl at redshifts out of the known range."
		exit()
	
	return dNdz_return / norm
	
def get_phi(z, lum_params, survey):
	
	""" This function outputs the Schechter luminosity function with parameters fit in Loveday 2012, following the same procedure as Krause et al. 2015, as a function of z and L 
	The output is L[z][l], list of vectors of luminosity values in z, different at each z due to the different lowe luminosity limit, and phi[z][l], a list of luminosity functions at these luminosity vectors, at each z
	lum_params are the parameters of the luminosity function that are different for different samples, e.g. red vs all. lumparams = [Mr_s, Q, alpha_lum, phi_0, P]
	Note that the luminosity function is output both normalized (for getting Ai and ah) and unnormalized (for the red fraction)."""
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
		
	[Mr_s, Q, alpha_lum, phi_0, P ] = lum_params
	
	# Get the amplitude of the Schechter luminosity function as a function of redshift.
	phi_s = phi_0 * 10.**(0.4 * P * z)
	
	# Get M_* (magnitude), then convert to L_*
	Ms = Mr_s - Q * (z - 0.1)
	Ls = 10**(-0.4 * (Ms - pa.Mp))
	
	# Import the kcorr and ecorr correction from Poggianti (assumes elliptical galaxies)
	# No data for sources beyon z = 3, so we keep the same value at higher z as z=3
	(z_k, kcorr, x,x,x) = np.loadtxt('./txtfiles/kcorr.dat', unpack=True)
	(z_e, ecorr, x,x,x) = np.loadtxt('./txtfiles/ecorr.dat', unpack=True)
	kcorr_interp = scipy.interpolate.interp1d(z_k, kcorr)
	ecorr_interp = scipy.interpolate.interp1d(z_e, ecorr)
	kcorr = kcorr_interp(z)
	ecorr = ecorr_interp(z)
	
	# Get the absolute magnitude and luminosity corresponding to limiting apparent magntiude (as a function of z)
	dl = com(z, survey) * (1. + z)
	Mlim = pa.mlim - (5. * np.log10(dl) + 25. + kcorr + ecorr)
	Llim = 10.**(-0.4 * (Mlim-pa.Mp))
	
	# Get the luminosity vectos - there will be a list of these, one for each redshift, because the limiting values are z-dependent
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
		phi_func_normed[zi] = np.zeros(len(L[zi]))
		norm[zi] = scipy.integrate.simps(phi_func[zi], L[zi])
		phi_func_normed[zi] = phi_func[zi] / norm[zi]

	return (L, phi_func_normed, phi_func)
	
def get_fred_ofz(z, survey):
	""" This function gets the red fraction as a function of the redshift, using the Schecter luminosity function as defined in get_phi"""

	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()

	(L_red, nothing, phi_red) = get_phi(z, pa.lumparams_red, survey)
	(L_all, nothing, phi_all) = get_phi(z, pa.lumparams_all, survey)
	
	# Integrate out in luminosity (phi is already normalized in luminosity
	phi_red_ofz = np.zeros(len(z))
	phi_all_ofz = np.zeros(len(z))
	for zi in range(0,len(z)):
		phi_red_ofz[zi] = scipy.integrate.simps(phi_red[zi], L_red[zi])
		phi_all_ofz[zi] = scipy.integrate.simps(phi_all[zi], L_all[zi])
		
	fred_ofz = phi_red_ofz / phi_all_ofz

	return fred_ofz
