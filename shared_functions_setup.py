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

	z = scipy.linspace(z_min+0.0001, z_max, zpts)
	
	if (dNdztype == 'Nakajima'):
		# dNdz takes form like in Nakajima et al. 2011 equation 3
		a = dNdzpar[0]
		zs = dNdzpar[1]
		nofz_ = (z / zs)**(a-1) * np.exp( -0.5 * (z / zs)**2)	
	elif (dNdztype == 'Smail'):
		# dNdz take form like in Smail et al. 1994
		alpha = dNdzpar[0]
		z0 = dNdzpar[1]
		beta = dNdzpar[2]
		nofz_ = z**alpha * np.exp( - (z / z0)**beta)
	else:
		print "dNdz type "+str(dNdztype)+" not yet supported; exiting."
		exit()
		
	plt.figure()
	plt.plot(z, nofz_)
	plt.savefig('./plots/dNdz_unnormed_LSST.pdf')
	plt.close()

	return (z, nofz_)
	
