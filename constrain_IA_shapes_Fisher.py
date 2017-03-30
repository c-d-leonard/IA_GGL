# This is a script which forecasts constraints on IA using multiple shape measurement methods.

import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import subprocess
import shutil

########## FUNCTIONS ##########

####### SET UP & BASICS #######

def setup_rp_bins(rmin, rmax, nbins):
	""" Sets up the edges of the bins of projected radius """
	
	bins = scipy.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
	
	return bins

def rp_bins_mid(rp_edges):
        """ Gets the middle of each projected radius bin."""

        logedges=np.log10(rp_edges)
        bin_centers=np.zeros(len(rp_edges)-1)
        for ri in range(0,len(rp_edges)-1):
                bin_centers[ri]    =       10**((logedges[ri+1] - logedges[ri])/2. +logedges[ri])

        return bin_centers

def com(z_):
	""" Gets the comoving distance in units of Mpc/h at a redshift or a set of redshifts. """

	OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN

	def chi_int(z):
		return 1. / (pa.H0 * ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )**(0.5))

	if hasattr(z_, "__len__"):
		chi=np.zeros((len(z_)))
		for zi in range(0,len(z_)):
			chi[zi] = scipy.integrate.quad(chi_int,0,z_[zi])[0]
	else:
		chi = scipy.integrate.quad(chi_int, 0, z_)[0]

	return chi

def z_interpof_com():
	""" Returns an interpolating function which can give z as a function of comoving distance. """

	z_vec = scipy.linspace(0., 10., 1000) # This hardcodes that we don't care about anything over z=10.

	com_vec = com(z_vec)

	z_of_com = scipy.interpolate.interp1d(com_vec, z_vec)

	return	z_of_com

def get_z_close(z_l, cut_MPc_h):
	""" Gets the z above z_l which is the highest z at which we expect IA to be present for that lens. cut_Mpc_h is that separation in Mpc/h."""

	com_l = com(z_l) # Comoving distance to z_l, in Mpc/h

	tot_com_high = com_l + cut_MPc_h
	tot_com_low = com_l - cut_MPc_h

	# Convert tot_com back to a redshift.

	z_cl_high = z_of_com(tot_com_high)
	z_cl_low = z_of_com(tot_com_low)

	return (z_cl_high, z_cl_low)

def get_areas(bins, z_eff):
        """ Gets the area of each projected radial bin, in square arcminutes. z_eff = effective lens redshift. """

        # Areas in units (Mpc/h)^2
        areas_mpch = np.zeros(len(bins)-1)
        for i in range(0, len(bins)-1):
                areas_mpch[i] =  np.pi * (bins[i+1]**2 - bins[i]**2)

        #Comoving distance out to effective lens redshift in Mpc/h
        chi_eff = com(z_eff)

        # Areas in square arcminutes (466560000 / pi = sqAM in a sphere)
        areas_sqAM = areas_mpch * (466560000. / np.pi) / (3 * np.pi * chi_eff**2)

        return areas_sqAM
	
def get_NofZ_unnormed(a, zs, z_min, z_max, zpts):
	""" Returns the dNdz of the sources as a function of photometric redshift, as well as the z points at which it is evaluated."""
	z = scipy.linspace(z_min, z_max, zpts)
	
	nofz_ = (z / zs)**(a-1) * np.exp(-0.5 * ( z / zs)**2)

	return (z, nofz_)

def get_z_frac(rp_cents_):
        """ Gets the fraction of sources of the full survey which are within our photo-z cut (excess and smooth background)"""
        
        # The normalization factor should be the norm over the whole range of z for which the number density of the survey is given, i.e. z min to zmax.       
        (z, dNdz) = N_of_zph_weighted(pa.zmin, pa.zmax, pa.zmin, pa.zmax, z_close_low, z_close_high, pa.zmin_ph, pa.zmax_ph, pa.e_rms_mean)
        
        frac_rand = scipy.integrate.simps(dNdz, z)
        
        boost_samp = get_boost(rp_cents_, pa.boost_samp)
        boost_tot = get_boost(rp_cents_, pa.boost_tot)
        
        frac_total = boost_samp / boost_tot * frac_rand

        return frac_total

def get_perbin_N_ls(rp_bins_, zeff_, ns_, nl_, A, rp_cents_):
	""" Gets the number of lens/source pairs relevant to each bin of projected radius """
	""" zeff_ is the effective redshift of the lenses. frac_ is the fraction of sources in the sample. ns_ is the number density of sources per square arcminute. nl_ is the number density of lenses per square degree. A is the survey area in square degrees."""
        
	frac_		=	get_z_frac(rp_cents_)

	# Get the area of each projected bin in square arcminutes
	bin_areas       =       get_areas(rp_bins_, zeff_)

	N_ls_pbin = nl_ * A * ns_ * bin_areas * frac_

	return N_ls_pbin

###### LINEAR  / NONLINEAR ALIGNMENT MODEL + 1 HALO TERMS FOR FIDUCIAL GAMMA_IA ######

def window(tag):
	""" Get window function for projected correlations in linear alignment model. Tag = '+g' or 'gg', determines dndz's."""
	
	sigz = pa.sigz_gwin
	z = scipy.linspace(pa.zeff-5.*sigz, pa.zeff+5.*sigz, 50)
	dNdz_1 = 1. / np.sqrt(2. * np.pi) / sigz * np.exp(-(z-pa.zeff)**2 / (2. *sigz**2))
	
	chi = com(z)
	OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
	dzdchi = pa.H0 * ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )**(0.5)
	
	if (tag =='gg'):
		dNdz_2 = dNdz_1
	elif ((tag=='+g') or (tag=='g+')):
		(z, dNdz_2) = get_NofZ_unnormed(pa.alpha, pa.zs, pa.zeff-5.*sigz, pa.zeff+5.*sigz, 50)
		
	norm = scipy.integrate.simps(dNdz_1*dNdz_2 / chi**2 * dzdchi, z)
	
	win = dNdz_1*dNdz_2 / chi**2 * dzdchi / norm
	
	return (z, win )
	
def growth(z_):
	""" Returns the growth factor, normalized to 1 at z=0"""
	
	def int_(z):
		OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
		return (1.+z) / ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )**(1.5)
	
	norm = scipy.integrate.quad(int_, 0, 1000.)[0]
	
	ans = np.zeros(len(z_))
	for zi in range(0,len(z_)):
		ans[zi] = scipy.integrate.quad(int_, z_[zi], 1000)[0]
	
	D = ans / norm
	
	return D
	
def get_Pk(z_, tag):
	""" Calls camb and returns the nonlinear power spectrum (from camb) for the current cosmological parameters and at the redshifts of interest. Returns a 2D array in k and z."""
	
	# The redshifts have to be in a certain order to make camb happy:
	z_list = list(z_)
	z_list.reverse()
	z_rev = np.asarray(z_list)
	
	cambfolderpath = '/home/danielle/Documents/CMU/camb'
	if (tag=='gg'):
		param_string='output_root=IA_shapes_gg\nombh2='+str(pa.OmB * (pa.HH0/100.)**2)+'\nomch2='+str(pa.OmC* (pa.HH0/100.)**2)+'\nhubble='+str(pa.HH0)+'\ntransfer_num_redshifts='+str(len(z_))
	elif ((tag=='gp') or (tag=='pg')):
		param_string='output_root=IA_shapes_gp\nombh2='+str(pa.OmB * (pa.HH0/100.)**2)+'\nomch2='+str(pa.OmC* (pa.HH0/100.)**2)+'\nhubble='+str(pa.HH0)+'\ntransfer_num_redshifts='+str(len(z_))
			
	for zi in range(0, len(z_rev)):
		param_string += '\ntransfer_redshift('+str(zi+1)+') = '+str(z_rev[zi])+ '\ntransfer_matterpower('+str(zi+1)+') = matterpower_z='+str(z_rev[zi])+'.dat'	
		
	
	tempfile=open(cambfolderpath+'/params_string.dat','w')
	tempfile.write(param_string)
	tempfile.close()

	paramsnewfilename='/params_IA_fid.ini'
	
	params_new=open(cambfolderpath+paramsnewfilename, 'w')
	shutil.copyfileobj(open(cambfolderpath+'/params_base.dat', 'r'), params_new)
	shutil.copyfileobj(open(cambfolderpath+'/params_string.dat', 'r'), params_new)
	params_new.close()
	
	#Now write the script to run camb for this set of parameters:
	
	temp_camb=open('./runcamb_temp.sh', 'w')
	temp_camb.write('./camb params_IA_fid.ini')
	temp_camb.close()
	run_camb_now=open('./runcambnow.sh', 'w')
	shutil.copyfileobj(open('./runcamb_base.sh', 'r'), run_camb_now)
	shutil.copyfileobj(open('./runcamb_temp.sh', 'r'), run_camb_now)
	run_camb_now.close()
	
	subprocess.call('./fix_permission.sh')
	subprocess.call('./runcambnow.sh')
	
	# Load one to get the length in k:
	if (tag=='gg'):
		(k, P) = np.loadtxt(cambfolderpath+'/IA_shapes_gg_matterpower_z='+str(z_[0])+'.dat',unpack=True)
	elif ((tag=='gp') or (tag=='pg')):
		(k, P) = np.loadtxt(cambfolderpath+'/IA_shapes_gp_matterpower_z='+str(z_[0])+'.dat',unpack=True)
	
	Pofkz = np.zeros((len(z_),len(k))) 
	for zi in range(0,len(z_)):
		if (tag=='gg'):
			k, Pofkz[zi, :] = np.loadtxt(cambfolderpath+'/IA_shapes_gg_matterpower_z='+str(z_[zi])+'.dat', unpack=True)
		elif ((tag=='gp') or (tag=='pg')):
			k, Pofkz[zi, :] = np.loadtxt(cambfolderpath+'/IA_shapes_gp_matterpower_z='+str(z_[zi])+'.dat', unpack=True)
	
	return (k, Pofkz)

## 1 halo for IA (g+) (see eg Singh 2014): 

def get_pi(q1, q2, q3, z_):
	""" Returns the pi functions requires for the 1 halo term in wg+, at z_ """
	
	pi = q1 * np.exp(q2 * z_**q3)
	
	return pi
	
def get_P1haloIA(z, k):
	""" Returns the power spectrum required for the wg+ 1 halo term, at z and k_perpendicular ( = k) """
	
	p1 = get_pi(pa.q11, pa.q12, pa.q13, z)
	p2 = get_pi(pa.q21, pa.q22, pa.q23, z)
	p3 = get_pi(pa.q31, pa.q32, pa.q33, z)
	
	P1halo = pa.ah * ( k / p1 )**2 / ( 1. + ( k /p2 )** p3)
	
	return P1halo
	
def wgp_1halo(rp_c_):
	""" Returns the 1 halo term of wg+(rp) """
	
	#(z, w) = window('g+') # Same window function as for the NLA term.
	
	# Set up a k vector to integrate over:
	k = np.logspace(-5., 7., 100000)
	
	# Get the `power spectrum' term
	P1h = get_P1haloIA(pa.zeff, k)
	
	# First do the integral over z:
	#zint = np.zeros(len(k))
	#for ki in range(0,len(k)):
	#	zint[ki] = scipy.integrate.simps(P1h[ki, :] * w , z)
		
	#plot_quant_vs_quant(k, zint, './zint.png')
		
	# Now do the integral in k
	ans = np.zeros(len(rp_c_))
	for rpi in range(0,len(rp_c_)):
		#integrand = k * zint * scipy.special.j0(rp_c_[rpi] * k)
		ans[rpi] = scipy.integrate.simps(k * P1h * scipy.special.j0(rp_c_[rpi] * k), k)
		
	# Set this to zero above about 2 * virial radius (I've picked this value somewhat aposteriori, should do better). This is to not include 1halo contributions well outside the halo.
	Rvir = Rhalo(pa.Mvir)
	for ri in range(0,len(rp_c_)):
		if (rp_c_[ri]> 2.*Rvir):
			ans[ri] = 0.
		
	wgp1h = ans / (2. * np.pi)
	
	#plt.figure()
	#plt.loglog(rp_c_, wgp1h, 'bo')
	#plt.xlim(0.1, 200)
	#plt.ylim(0.01, 30)
	#plt.xlabel('$r_p$, Mpc/h com')
	#plt.ylabel('$w_{g+}$, Mpc/ h com')
	#plt.savefig('./plots/wg+_1h_check_shapes.pdf')
	#plt.close()
	
	wgp_save = np.column_stack((rp_c_, wgp1h))
	np.savetxt('./txtfiles/wgp_1halo_shapes.txt', wgp_save)
		
	return wgp1h

# 1 halo for DM with NFW profile (gg):
	
def Rhalo(M_insol):
	""" Get the radius of a halo in COMOVING Mpc/h given its mass."""
	
	#rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun * (pa. HH0 / 100.)) # Msol h^3 / Mpc^3, for use with M in Msol.
	rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h
	rho_m = rho_crit * pa.OmM
	Rvir = ( 3. * M_insol / (4. * np.pi * rho_m * 180.))**(1./3.)
	
	return Rvir
	
def cvir(M_insol):
	""" Returns the concentration parameter of the NFW profile, c_{vir}. """
	
	#cvi = pa.c14 / (1. + z) * (M_insol / 10**14)**(-0.11)
	
	# This is the mass concentration relation used in Singh 2014, for comparison
	#cvi = 5. * (M_insol * (pa.HH0 / 100.) / 10**14 ) ** (-0.1)
	cvi = 5. * (M_insol / 10**14)**(-0.1)
	
	return cvi 
	
def rho_s(cvi, Rvi, M_insol):
	""" Returns rho_s, the NFW parameter representing the density at the `scale radius', Rvir / cvir. Units: Mvir units * ( 1 / (Rvir units)**3), usualy Msol * h^3 / Mpc^3 with comoving distances. Sometimes also Msol h^2 / Mpc^3 (when Mvir is in Msol / h). """
	
	rhos = M_insol / (4. * np.pi) * ( cvi / Rvi)**3 * (np.log(1. + cvi) - (cvi / (1. + cvi)))**(-1)
	
	return rhos
	
def rho_NFW(r_, M_insol, z):
	""" Returns the density for an NFW profile in real space at distance r from the center (?). Units = units of rhos. (Usually Msol * h^3 / Mpc^3). r_ MUST be in the same units as Rv; usually Mpc / h."""
	
	Rv = Rhalo(M_insol)
	cv = cvir(M_insol)
	rhos = rho_s(cv, Rv, M_insol)
	
	rho_nfw = rhos  / ( (cv * r_ / Rv) * (1. + cv * r_ / Rv)**2)
	
	return rho_nfw
	
def wgg_1halo_Four(rp_cents_):
	""" Gets the 1halo term of wgg via Fourier space, to account for central-satelite pairs and satelite-satelite pairs. """
	
	Rvir = Rhalo(pa.Mvir)
	
	kvec_FT = np.logspace(-7, 4, 1000000)
	rvec_NFW = np.logspace(-7, np.log10(Rvir), 10000000)
	rvec_xi = np.logspace(-4, 4, 1000)
	Pivec = np.logspace(-7, np.log10(1000./np.sqrt(2.01)), 1000)
	rpvec = np.logspace(-7, np.log10(1000./np.sqrt(2.01)), 1000)
	
	Pk = get_Pkgg_1halo(rvec_NFW, kvec_FT) # Gets the 1halo galaxy power spectrum including c-s and s-s terms. Pass rvec because need to get rho_NFW in here.
	#(k_dummy, Pk) = np.loadtxt('./txtfiles/Pkgg_1h.txt', unpack=True)
	
	xi_gg_1h = get_xi_1h(rvec_xi, kvec_FT, Pk) # Function that gets the 1halo galaxy-galaxy correlation function term.
	
	plt.figure()
	plt.loglog(rvec_xi, xi_gg_1h, 'go')
	plt.ylabel('$\\xi(r)$')
	plt.xlabel('$r$, Mpc/h com')
	plt.xlim(10**(-7), 50)
	plt.savefig('./plots/xigg_1h_FFT.png')
	plt.close()
	
	# Set xi_gg_1h to zero above Rvir - we have made this assumption; anything else is Fourier transform noise.
	for ri in range(0, len(rvec_xi)):
		if (rvec_xi[ri]>Rvir):
			xi_gg_1h[ri] = 0.0
			
	plt.figure()
	plt.loglog(rvec_xi, xi_gg_1h, 'go')
	plt.ylabel('$\\xi(r)$')
	plt.xlabel('$r$, Mpc/h com')
	#plt.xlim(0.5, 0.6)
	plt.savefig('./plots/xigg_1h_FFT_mod.png')
	plt.close()
	
	xi_interp = scipy.interpolate.interp1d(rvec_xi, xi_gg_1h)
	
	xi_2D = np.zeros((len(rp_cents_), len(Pivec)))
	for ri in range(0, len(rp_cents_)):
		for pi in range(0, len(Pivec)):
			xi_2D[ri, pi] = xi_interp(np.sqrt(rp_cents_[ri]**2 + Pivec[pi]**2)) 
			
	# Only integrate out to the virial radius
	#Rvir = Rhalo(pa.Mvir)
	#indvir = next(j[0] for j in enumerate(Pivec) if j[1]>=(Rvir))
	
	wgg_1h = np.zeros(len(rp_cents_))
	for ri in range(0,len(rp_cents_)):
		wgg_1h[ri] = scipy.integrate.simps(xi_2D[ri, :], Pivec)
		
	plt.figure()
	plt.loglog(rp_cents_, wgg_1h, 'go')
	plt.xlim(0.05, 20.)
	plt.ylim(10**(-3), 10**(4))
	plt.xlabel('$r_p$, Mpc/h com')
	plt.ylabel('$w_{gg}$, Mpc/h com')
	plt.savefig('./plots/wgg_1h_shapes.png')
	plt.close()
	
	wgg_save = np.column_stack((rp_cents_, wgg_1h))
	np.savetxt('./txtfiles/wgg_1halo_mod_shapes.txt', wgg_save)
	
	return wgg_1h

def get_xi_1h(r, kvec, Pofk):
	""" Returns the 1 halo galaxy correlation function including cen-sat and sat-sat terms, from the power spectrum via Fourier transform."""
	
	xi = np.zeros(len(r))
	for ri in range(0,len(r)):
		xi[ri] = scipy.integrate.simps(Pofk * kvec**2 * np.sin(kvec*r[ri]) / (kvec * r[ri]) / 2. / np.pi**2 , kvec)

	return xi

def get_Pkgg_1halo(rvec_nfw, kvec_ft):
	""" Returns the 1halo galaxy power spectrum with c-s and s-s terms."""
	
	# Get ingredients we need here:
	y = gety(rvec_nfw, pa.Mvir, pa.zeff, kvec_ft) # Mass-averaged Fourier transform of the density profile

	alpha = get_alpha(pa.Mvir) # The number which accounts for deviation from Poisson statistics
	Ncenavg = 1. # We assume that every halo has a central galaxy, so the mean number of central galaxies / halo is 1.
	
	fcen = 1. - pa.fsat # fraction of central galaxies = 1 - fraction of satelite galaxies
	Nsatavg = pa.fsat / fcen # Mean number of satelite galaxies per halo

	NcNs = NcenNsat(alpha, Ncenavg, Nsatavg) # The average number of central-satelite pairs in a halo of mass M
	NsNs = NsatNsat(alpha, Nsatavg) # The average number of satelite-satelite pairs in a halo of mass M

	Pkgg = (1. - pa.fsat) / pa.ng * (NcNs * y + NsNs * y **2)

	#plt.figure()
	#plt.loglog(kvec_ft, Pkgg, 'b+')
	#plt.ylabel('$P_{gg}^{1h}(k)$, $(Mpc/h)^3$, com')
	#plt.xlabel('$k$, h/Mpc, com')
	#plt.savefig('./plots/Pkgg_1halo_shapes.png')
	#plt.close()
	
	Pkgg_save = np.column_stack((kvec_ft, Pkgg))
	np.savetxt('./txtfiles/Pkgg_1h.txt', Pkgg_save)

	return Pkgg

def gety(rvec, M, z, kvec):
	""" Fourier transforms the density profile to get the power spectrum. """
	
	# Get the nfw density profile at the correct mass and redshift and at a variety of r
	rho = rho_NFW(rvec, M, z)  # Units Msol h^2 / Mpc^3, comoving.
	print "in gety - got rho"
	
	# Use a downsampled kvec to speed computation
	kvec_gety = np.logspace(np.log10(kvec[0]), np.log10(kvec[-1]), 1000) 
	print "in gety - got downsampled kvec"
	
	u_ = np.zeros(len(kvec_gety))
	for ki in range(0,len(kvec_gety)):
		u_[ki] = 4. * np.pi / M * scipy.integrate.simps( rvec * np.sin(kvec_gety[ki]*rvec)/ kvec_gety[ki] * rho, rvec) # unitless / dimensionless.
		
	print "in gety - got u"
		
	# Interpolate in k and use the higher-sampled k to output
	u_interp = scipy.interpolate.interp1d(kvec_gety, u_)
	u_morepoints = u_interp(kvec) 
		
	#plt.figure()
	#plt.loglog(kvec, u_morepoints, 'b+')
	#plt.ylim(0.01, 2)
	#plt.xlim(0.002, 100000)
	#plt.ylabel('u')
	#plt.xlabel('$k$, h/Mpc, com')
	#plt.savefig('./plots/u_shapes.png')
	#plt.close()
	
	return u_morepoints

def get_alpha(M):
	""" Gets the parameter that accounts for the deviation from Poisson statistics. M is in Msol / h. """
	
	if (M<10**11):
		alpha = np.sqrt(np.log(M / 10**11))
	else:
		alpha = 1.
		
	return alpha

def NcenNsat(alpha, Ncen, Nsat):
	""" Returns the average number of pairs of central and satelite galaxies per halo of mass M. """
	
	NcNs = alpha**2 * Ncen * Nsat
	
	return NcNs
	
def NsatNsat(alpha, Nsat):
	""" Returns the average number of pairs of satelite galaxies per halo. """
	
	NsNs = alpha**2 * Nsat**2
	
	return NsNs	

# 2 halo terms:
	
def wgg_2halo(rp_cents_):
	""" Returns wgg for the 2-halo term only."""
	
	# Get the redshift window functions
	z_gg, win_gg = window('gg')
	
	# Get the NL DM power spectrum from camb (this is only for the 2-halo term)
	(k_gg, P_gg) = get_Pk(z_gg, 'gg')
	
	# First do the integral over z. Don't yet interpolate in k.
	zint_gg = np.zeros(len(k_gg))
	for ki in range(0,len(k_gg)):
		zint_gg[ki] = scipy.integrate.simps(win_gg * P_gg[:, ki], z_gg)
		
	# Define vectors of kp (kperpendicual) and kz. Must have sufficiently high sampling to get the right answer, especially at large scales.
	kp_gg = np.logspace(np.log10(k_gg[0]), np.log10(k_gg[-1]/ np.sqrt(2.01)), pa.kpts_wgg)
	kz_gg = np.logspace(np.log10(k_gg[0]), np.log10(k_gg[-1]/ np.sqrt(2.01)), pa.kpts_wgg)
	
	# Interpolate in terms of kperp and kz
	kinterp_gg = scipy.interpolate.interp1d(k_gg, zint_gg)
	
	# Get the result of the z integral in terms of kperp and kz
	kpkz_gg = np.zeros((len(kp_gg), len(kz_gg)))
	for kpi in range(0,len(kp_gg)):
		for kzi in range(0, len(kz_gg)):
			kpkz_gg[kpi, kzi] = kinterp_gg(np.sqrt(kp_gg[kpi]**2 + kz_gg[kzi]**2))
			
	# Do the integrals in kz
	kz_int_gg = np.zeros(len(kp_gg))
	for kpi in range(0,len(kp_gg)):
		kz_int_gg[kpi] = scipy.integrate.simps(kpkz_gg[kpi,:] * kp_gg[kpi] / kz_gg * np.sin(kz_gg*pa.ProjMax), kz_gg)
		
	# Do the integral in kperp
	kp_int_gg = np.zeros(len(rp_cents_))
	for rpi in range(0,len(rp_cents_)):
		kp_int_gg[rpi] = scipy.integrate.simps(scipy.special.j0(rp_cents_[rpi]* kp_gg) * kz_int_gg, kp_gg)
		
	wgg_2h = kp_int_gg * pa.bs * pa. bd / np.pi**2
	wgg_stack = np.column_stack((rp_cents_, wgg_2h))
	np.savetxt('./txtfiles/wgg_2halo_'+str(pa.kpts_wgg)+'pts.txt', wgg_stack)
	
	return wgg_2h
		
def wgp_2halo(rp_cents_):
	""" Returns wgp from the nonlinear alignment model (2-halo term only). """
	
	# Get the redshift window function
	z_gp, win_gp = window('g+')
	
	# Get the required matter power spectrum from camb 
	(k_gp, P_gp) = get_Pk(z_gp, 'gp')
	
	# Get the growth factor
	D_gp = growth(z_gp)
	
	# First do the integral over z. Don't yet interpolate in k.
	zint_gp = np.zeros(len(k_gp))
	for ki in range(0,len(k_gp)):
		zint_gp[ki] = scipy.integrate.simps(win_gp * P_gp[:, ki] / D_gp, z_gp)
		
	# Define vectors of kp (kperpendicual) and kz. 
	kp_gp = np.logspace(np.log10(k_gp[0]), np.log10(k_gp[-1]/ np.sqrt(2.01)), pa.kpts_wgp)
	kz_gp = np.logspace(np.log10(k_gp[0]), np.log10(k_gp[-1]/ np.sqrt(2.01)), pa.kpts_wgp)
	
	# Interpolate the answers to the z integral in k to get it in terms of kperp and kz
	kinterp_gp = scipy.interpolate.interp1d(k_gp, zint_gp)
	
	# Get the result of the z integral in terms of kperp and kz
	kpkz_gp = np.zeros((len(kp_gp), len(kz_gp)))
	for kpi in range(0,len(kp_gp)):
		for kzi in range(0, len(kz_gp)):
			kpkz_gp[kpi, kzi] = kinterp_gp(np.sqrt(kp_gp[kpi]**2 + kz_gp[kzi]**2))
			
	# g+: integral in kz	
	kz_int_gp = np.zeros(len(kp_gp))
	for kpi in range(0,len(kp_gp)):
		kz_int_gp[kpi] = scipy.integrate.simps(kpkz_gp[kpi,:] * kp_gp[kpi]**3 / ( (kp_gp[kpi]**2 + kz_gp**2)*kz_gp) * np.sin(kz_gp*pa.ProjMax), kz_gp)
			
	# Finally, do the integrals in kperpendicular
	kp_int_gp = np.zeros(len(rp_cents_))
	for rpi in range(0,len(rp_cents_)):
		kp_int_gp[rpi] = scipy.integrate.simps(scipy.special.jv(2, rp_cents_[rpi]* kp_gp) * kz_int_gp, kp_gp)
		
	wgp_NLA = kp_int_gp * pa.Ai * pa.bd * pa.C1rho * (pa.OmC + pa.OmB) / np.pi**2
	
	wgp_stack = np.column_stack((rp_cents_, wgp_NLA))
	np.savetxt('./txtfiles/wgp_2halo_'+str(pa.kpts_wgp)+'pts.txt', wgp_stack)
	
	return wgp_NLA

# Put 1 halo and 2 halo together

def wgg_full(rp_c):
	""" Combine 1 and 2 halo terms of wgg """
	
	#wgg_1h = wgg_1halo(rp_c, pa.zeff)
	#print "GETTING 1HALO WGG"
	#wgg_1h = wgg_1halo_Four(rp_c)
	(rp_cen, wgg_1h) = np.loadtxt('./txtfiles/wgg_1halo_mod_shapes.txt', unpack=True)
	#print "DONE WITH 1HALO WGG, GETTING 2HALO"
	#wgg_2h = wgg_2halo(rp_c)
	#print "DONE WITH 2HALO WGG"
	(rp_cen, wgg_2h) = np.loadtxt('./txtfiles/wgg_2halo_'+str(pa.kpts_wgg)+'pts.txt', unpack=True)
	
	wgg_tot = wgg_1h + wgg_2h 
	
	#plt.figure()
	#plt.semilogx(rp_c, rp_c* wgg_tot, 'go')
	#plt.xlim(0.01, 200.)
	#plt.ylim(0., 300.)
	#plt.ylabel('$r_p w_{gg}$, $(Mpc/h)^2$, com')
	#plt.xlabel('$r_p$, Mpc/h, com')
	#plt.savefig('./plots/Rwgg_tot_SatFrac_fixprefac_fixcovDS.png')
	#plt.close()
	
	#plt.figure()
	#plt.loglog(rp_c, wgg_tot, 'go')
	#plt.xlim(0.01, 200.)
	#plt.ylim(0., 300.)
	#plt.ylabel('$w_{gg}$, Mpc/h, com')
	#plt.xlabel('$r_p$, Mpc/h, com')
	#plt.savefig('./plots/wgg_tot_shapes_mod.png')
	#plt.close()
	
	return wgg_tot
		
def wgp_full(rp_c):
	""" Combine 1 and 2 halo terms of wgg """
	
	#print "GETTING WGP_1H"
	#wgp_1h = wgp_1halo(rp_c)
	(rp_cen, wgp_1h) = np.loadtxt('./txtfiles/wgp_1halo_shapes.txt', unpack=True)
	#print "DONE WITH WGP_1H, ON TO WGP 2H"
	#wgp_2h = wgp_2halo(rp_c)
	#print "DONE WITH 2 HALO WGP"
	(rp_cen, wgp_2h) = np.loadtxt('./txtfiles/wgp_2halo_'+str(pa.kpts_wgp)+'pts.txt', unpack=True)
	
	wgp_tot = wgp_1h + wgp_2h 
	
	#plt.figure()
	#plt.loglog(rp_c, wgp_1h, 'go')
	#plt.xlim(0.1, 200.)
	#plt.ylim(0.01, 30)
	#plt.ylabel('$w_{gp}$, Mpc/h, com')
	#plt.xlabel('$r_p$, Mpc/h')
	#plt.savefig('./plots/wgp_1h_shapes.png')
	#plt.close()
	
	#plt.figure()
	#plt.loglog(rp_c, wgp_2h, 'go')
	#plt.xlim(0.1, 200.)
	#plt.ylim(0.01, 30)
	#plt.ylabel('$w_{gp}$, Mpc/h, com')
	#plt.xlabel('$r_p$, Mpc/h')
	#plt.savefig('./plots/wgp_2h_shapes.png')
	#plt.close()
	
	#plt.figure()
	#plt.loglog(rp_c, wgp_tot, 'go')
	#plt.xlim(0.1, 200.)
	#plt.ylim(0.01, 30)
	#plt.ylabel('$w_{gp}$, Mpc/h, com')
	#plt.xlabel('$r_p$, Mpc/h')
	#plt.savefig('./plots/wgp_tot_shapes.png')
	#plt.close()
	
	return wgp_tot		
	
def gamma_fid(rp):
	""" Returns the fiducial gamma_IA from a combination of terms from different models which are valid at different scales """
	
	wgg_rp = wgg_full(rp)
	wgp_rp = wgp_full(rp)
	
	gammaIA = wgp_rp / (wgg_rp + 2. * pa.close_cut) 
	
	"""plt.figure()
	plt.loglog(rp, gammaIA, 'go')
	plt.ylim(10**(-4), 0.1)
	plt.ylabel('$\gamma_{IA}^{fid}$')
	plt.xlabel('$r_p$, Mpc/h')
	plt.savefig('./plots/gIA_fid_shapes.png')
	plt.close()"""
	
	return gammaIA
	

####### GETTING ERRORS #########

def get_boost(rp_cents_, propfact):
	""" Returns the boost factor in radial bins. propfact is a tunable parameter giving the proportionality constant by which boost goes like projected correlation function (= value at 1 Mpc/h). """

	Boost = propfact *(rp_cents_)**(-0.8) + np.ones((len(rp_cents_)))# Empirical power law fit to the boost, derived from the fact that the boost goes like projected correlation function.

	return Boost

def N_of_zph_unweighted(z_a_def, z_b_def, z_a_norm, z_b_norm, z_a_norm_ph, z_b_norm_ph):
	""" Returns dNdz_ph, the number density in terms of photometric redshift, defined and normalized over the photo-z range (z_a_norm_ph, z_b_norm_ph), normalized over the spec-z range (z_a_norm, z_b_norm), but defined on the spec-z range (z_a_def, z_b_def)"""
	
	(z, dNdZ) = get_NofZ_unnormed(pa.alpha, pa.zs, z_a_def, z_b_def, pa.zpts)
	(z_norm, dNdZ_norm) = get_NofZ_unnormed(pa.alpha, pa.zs, z_a_norm, z_b_norm, pa.zpts)
	
	z_ph_vec = scipy.linspace(z_a_norm_ph, z_b_norm_ph, 5000)
	
	int_dzs = np.zeros(len(z_ph_vec))
	int_dzs_norm = np.zeros(len(z_ph_vec))
	for i in range(0,len(z_ph_vec)):
		int_dzs[i] = scipy.integrate.simps(dNdZ*p_z(z_ph_vec[i], z, pa.sigz), z)
		int_dzs_norm[i] = scipy.integrate.simps(dNdZ_norm*p_z(z_ph_vec[i], z_norm, pa.sigz), z_norm)
		
	norm = scipy.integrate.simps(int_dzs_norm, z_ph_vec)
	
	return (z_ph_vec, int_dzs / norm)
	
def N_of_zph_weighted(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, erms):
	""" Returns dNdz_ph, the number density in terms of photometric redshift, defined over photo-z range (z_a_def_ph, z_b_def_ph), normalized over the photo-z range (z_a_norm_ph, z_b_norm_ph), normalized over the spec-z range (z_a_norm, z_b_norm), and defined on the spec-z range (z_a_def, z_b_def)"""
	
	(z, dNdZ) = get_NofZ_unnormed(pa.alpha, pa.zs, z_a_def_s, z_b_def_s, pa.zpts)
	(z_norm, dNdZ_norm) = get_NofZ_unnormed(pa.alpha, pa.zs, z_a_norm_s, z_b_norm_s, pa.zpts)
	
	z_ph_vec = scipy.linspace(z_a_def_ph, z_b_def_ph, 5000)
	z_ph_vec_norm = scipy.linspace(z_a_norm_ph, z_b_norm_ph, 5000)
	
	weights_ = weights(erms, z_ph_vec)
	weights_norm = weights(erms, z_ph_vec_norm)
	
	int_dzs = np.zeros(len(z_ph_vec))
	for i in range(0,len(z_ph_vec)):
		int_dzs[i] = scipy.integrate.simps(dNdZ*p_z(z_ph_vec[i], z, pa.sigz), z)
	
	int_dzs_norm = np.zeros(len(z_ph_vec_norm))
	for i in range(0,len(z_ph_vec_norm)):
		int_dzs_norm[i] = scipy.integrate.simps(dNdZ_norm*p_z(z_ph_vec_norm[i], z_norm, pa.sigz), z_norm)
		
	norm = scipy.integrate.simps(weights_norm*int_dzs_norm, z_ph_vec_norm)
	
	return (z_ph_vec, weights_*int_dzs / norm)

def N_in_samp(z_a, z_b, e_rms_weights):
	""" Number of galaxies in the photometric redshift range of the sample (assumed z_eff of lenses to z_close) from the SPECTROSCOPIC redshift range z_a to z_b """
	
	(z_ph, N_of_zp) = N_of_zph_weighted(z_a, z_b, pa.zmin, pa.zmax, z_close_low, z_close_high, z_close_low, z_close_high, pa.e_rms_mean)

	answer = scipy.integrate.simps(N_of_zp, z_ph)
	
	return (answer)

def N_corr(rp_cent):
	""" Computes the correction factor which accounts for the fact that some of the galaxies in the photo-z defined source sample are actually higher-z and therefore not expected to be affected by IA"""
	
	N_tot = N_in_samp(pa.zmin, pa.zmax, pa.e_rms_mean)
	N_close = N_in_samp(z_close_low, z_close_high, pa.e_rms_mean)
	boost = get_boost(rp_cent, pa.boost_samp)
	
	Corr_fac = 1. - (1. / boost) * ( 1. - (N_close / N_tot)) # fraction of the galaxies in the source sample which have spec-z in the photo-z range of interest.
	
	return Corr_fac

def N_corr_stat_err(rp_cents_, boost_error_file):
	""" Gets the error on N_corr from statistical error on the boost. """
	
	N_tot = N_in_samp(pa.zmin, pa.zmax, pa.e_rms_mean)
	N_close = N_in_samp(z_close_low, z_close_high, pa.e_rms_mean)
	boost = get_boost(rp_cents_, pa.boost_samp)
	boost_err = boost_errors(rp_cents_, boost_error_file)
	
	sig_Ncorr = (boost_err / boost**2) * np.sqrt(1. - N_close / N_tot)

	return sig_Ncorr

def boost_errors(rp_bins_c, filename):
	""" Imports a file with 2 columns, [rp (kpc/h), sigma(boost-1)]. Interpolates and returns the value of the error on the boost at the center of each bin. """
	
	(rp_kpc, boost_error_raw) = np.loadtxt(filename, unpack=True)
	
	# Convert the projected radius to Mpc/h
	rp_Mpc = rp_kpc / 1000.
	
	interpolate_boost_error = scipy.interpolate.interp1d(rp_Mpc, boost_error_raw)
	
	boost_error = interpolate_boost_error(rp_bins_c)
	
	return boost_error

def sigma_e(z_s_, s_to_n):
	""" Returns a value for the model for the per-galaxy noise as a function of source redshift"""

	sig_e = 2. / s_to_n * np.ones(len(z_s_))

	return sig_e

def weights(e_rms, z_):
	""" Returns the inverse variance weights as a function of redshift. """
	
	weights = (1./(sigma_e(z_, pa.S_to_N)**2 + e_rms**2)) * np.ones(len(z_))
	
	return weights
	
def p_z(z_ph, z_sp, sigz):
	""" Returns the probability of finding a photometric redshift z_ph given that the true redshift is z_sp. """
	
	# I'm going to use a Gaussian probability distribution here for now
	p_z_ = np.exp(-(z_ph - z_sp)**2 / (2.*(sigz*(1.+z_sp))**2)) / (np.sqrt(2.*np.pi)*(sigz * (1. + z_sp)))
	
	return p_z_

def setup_shapenoise_cov(e_rms, N_ls_pbin):
	""" Returns a diagonal covariance matrix in bins of projected radius for a measurement dominated by shape noise. Elements are e_{rms}^2 / (N_ls) where N_ls is the number of l/s pairs relevant to each projected radius bin.""" 

	cov = np.diag(e_rms**2 / N_ls_pbin)
	
	return cov

def get_cov_btw_methods(cov_a_, cov_b_):
	""" Get the covariance between the methods given their correlation """
	
	cov_btw_methods = pa.cov_perc * np.sqrt(np.diag(cov_a_)) * np.sqrt(np.diag(cov_b_))
	
	return cov_btw_methods

def subtract_var(var_1, var_2, covar):
	""" Takes the variance of two non-independent  Gaussian random variables, and their covariance, and returns the variance of their difference."""
	
	var_diff = var_1 + var_2 - 2 * covar

	return var_diff

def get_gammaIA_stat_cov(Cov_1, Cov_2, rp_cents_, gIA_fid):
	""" Takes the covariance matrices of the constituent elements of gamma_{IA} and combines them to get the covariance matrix of gamma_{IA} in projected radial bins."""
	
	# Get the covariance between the shear of the two shape measurement methods in each bin:
	covar = get_cov_btw_methods(Cov_1, Cov_2)
	
	corr_fac = N_corr(rp_cents_)  # factor correcting for galaxies which have higher spec-z than the sample but which end up in the sample.

	corr_fac_err = N_corr_stat_err(rp_cents_, pa.sigBF_a) # statistical error on that from the boost

	stat_mat = np.diag(np.zeros(len(np.diag(Cov_1))))
	boost_term = np.zeros(len(np.diag(Cov_1)))
	shape_term = np.zeros(len(np.diag(Cov_1)))
	for i in range(0,len(np.diag(Cov_1))):	
		stat_mat[i, i] = (1.-pa.a_con)**2 * gIA_fid[i]**2 * ( corr_fac_err[i]**2 / corr_fac[i]**2 + subtract_var(Cov_1[i,i], Cov_2[i,i], covar[i]) / (corr_fac[i]**2 * (1.-pa.a_con)**2 * gIA_fid[i]**2))
		#boost_term[i] = ( corr_fac_err[i]**2 / corr_fac[i]**2 )
		#shape_term[i]  = (subtract_var(Cov_1[i,i], Cov_2[i,i], covar[i]) / (corr_fac[i]**2 * (1.-pa.a_con)**2 * gIA_fid[i]**2))
		
	print "stat=",  np.sqrt(np.diag(stat_mat))
	
	save_variance = np.column_stack((rp_cents_, np.sqrt(np.diag(stat_mat)) / ((1.-pa.a_con) * gIA_fid)))
	np.savetxt('./txtfiles/fractional_error_shapes_sigz='+str(pa.sigz)+'_covperc='+str(pa.cov_perc)+'_a='+str(pa.a_con)+'.txt', save_variance)
	
	plt.figure()
	plt.loglog(rp_cents, np.sqrt(np.diag(stat_mat)), 'mo')
	plt.xlim(0.04, 20)
	plt.savefig('./plots/statvariance_alone_shapes_modwgg.pdf')
	plt.close()

	return stat_mat

def get_gammaIA_sys_cov(rp_cents_, sys_dN, sys_p, gIa_fid):
	""" Takes the centers of rp_bins and a systematic error sources from dNdz_s uncertainty (assumed to affect each r_p bin in the same way) and adds them to each other in quadrature."""
	
	corr_fac = N_corr(rp_cents_) # correction factor
	
	
	sys_mat = np.zeros((len(rp_cents_), len(rp_cents_)))
	
	for i in range(0,len(rp_cents_)):
		for j in range(0,len(rp_cents_)):
			sys_mat[i,j] = sys_dN[i]*sys_dN[j] * (1-pa.a_con)**2 * corr_fac[i] * corr_fac[j] * gIa_fid[i] * gIa_fid[j] + sys_p[i]*sys_p[j] * (1-pa.a_con)**2 * corr_fac[i] * corr_fac[j] * gIa_fid[i] * gIa_fid[j]
			
	print "sys=", np.sqrt(np.diag(sys_mat))

	return sys_mat
	
def get_gamma_tot_cov(sys_mat, stat_mat):
	""" Takes the covariance matrix from statistical error and systematic error, and adds them to get the total covariance matrix. Assumes stat and sys errors should be added in quadrature."""
	
	tot_cov = sys_mat+stat_mat
	
	print "tot=", np.sqrt(np.diag(tot_cov))
	
	return tot_cov

####### PLOTTING / OUTPUT #######

def plot_variance(cov_1, fidvalues_1, bin_centers):
	""" Takes a covariance matrix, a vector of the fiducial values of the object in question, and the edges of the projected radial bins, and makes a plot showing the fiducial values and 1-sigma error bars from the diagonal of the covariance matrix. Outputs this plot to location 'filename'."""

	fig_sub=plt.subplot(111)
	plt.rc('font', family='serif', size=20)
	#fig_sub=fig.add_subplot(111) #, aspect='equal')
	fig_sub.set_xscale("log")
	fig_sub.set_yscale("log")
	fig_sub.errorbar(bin_centers,fidvalues_1*(1-pa.a_con), yerr = np.sqrt(np.diag(cov_1)), fmt='mo')
	#fig_sub.errorbar(bin_centers,fidvalues_1, yerr = np.sqrt(np.diag(cov_1)), fmt='mo')
	fig_sub.set_xlabel('$r_p$')
	fig_sub.set_ylabel('$\gamma_{IA}(1-a)$')
	#fig_sub.set_ylabel('$\gamma_{IA}$')
	fig_sub.set_ylim(10**(-5), 0.05)
	#fig_sub.set_ylim(10**(-4), 0.1)
	fig_sub.tick_params(axis='both', which='major', labelsize=12)
	fig_sub.tick_params(axis='both', which='minor', labelsize=12)
	plt.tight_layout()
	plt.savefig('./plots/errorplot_statonly_shapes.pdf')

	return  

def plot_quant_vs_rp(quant, rp_cent, file):
	""" Plots any quantity vs the center of redshift bins"""

	plt.figure()
	plt.loglog(rp_cent, quant, 'ko')
	plt.xlabel('$r_p$')
	plt.tick_params(axis='both', which='major', labelsize=12)
	plt.tick_params(axis='both', which='minor', labelsize=12)
	plt.tight_layout()
	plt.savefig(file)

	return
	
def plot_quant_vs_quant(quant1, quant2, file):
	""" Plots any quantity vs any other."""

	plt.figure()
	plt.loglog(quant1, quant2, 'ko')
	plt.tick_params(axis='both', which='major', labelsize=12)
	plt.tick_params(axis='both', which='minor', labelsize=12)
	plt.tight_layout()
	plt.savefig(file)

	return

######### FISHER STUFF ##########

def par_derivs(params, rp_mid):
        """ Computes the derivatives of gamma_IA wrt the parameters of the IA model we care about constraining.
        Returns a matrix of dimensions (# r_p bins, # parameters)."""

	n_bins = len(rp_mid)

        derivs = np.zeros((n_bins, len(params)))

        # This is for a power law model gamma_IA = A * rp ** beta

        derivs[:, pa.A] = rp_mid**(pa.beta_fid)

        derivs[:, pa.beta] = pa.beta_fid * pa.A_fid * rp_mid**(pa.beta_fid-1) 

        return derivs

def get_Fisher(p_derivs, dat_cov):     
        """ Constructs the Fisher matrix, given a matrix of derivatives wrt parameters in each r_p bin and the data covariance matrix."""

	inv_dat_cov = np.linalg.inv(dat_cov)

        Fish = np.zeros((len(p_derivs[0,:]), len(p_derivs[0, :])))
        for a in range(0,len(p_derivs[0,:])):
                for b in range(0,len(p_derivs[0,:])):
                        Fish[a,b] = np.dot(p_derivs[:,a], np.dot(inv_dat_cov, p_derivs[:,b]))
        return Fish

def cut_Fisher(Fish, par_ignore ):
        """ Cuts the Fisher matrix to ignore any parameters we want to ignore. (par_ignore is a list of parameter names as defed in input file."""

        if (par_ignore!=None):
                Fish_cut = np.delete(np.delete(Fish, par_ignore, 0), par_ignore, 1)
        else:
                Fish_cut = Fish

        return Fish_cut

def get_par_Cov(Fish_, par_marg):
        """ Takes a Fisher matrix and returns a parameter covariance matrix, cutting out (after inversion) any parameters which will be marginalized over. Par_marg should either be None, if no parameters to be marginalised, or a list of parameters to be marginalised over by name from the input file."""

        par_Cov = np.linalg.inv(Fish_)

        if (par_marg != None):
                par_Cov_marg = np.delete(np.delete(par_Cov, par_marg, 0), par_marg, 1)
        else:
                par_Cov_marg = par_Cov

        return par_Cov_marg

def par_const_output(Fish_, par_Cov):
        """ Gunction to output  whatever information about parameters constaints we want given the parameter covariance matrix. The user should modify this function as desired."""

        # Put whatever you want to output here 

        print "1-sigma constraint on A=", np.sqrt(par_Cov[pa.A, pa.A])

        print "1-sigma constraint on beta=", np.sqrt(par_Cov[pa.beta, pa.beta])

        return


######## MAIN CALLS ##########

# Import the parameter file:
import IA_params_shapes as pa

# Set up projected radial bins
rp_bins 	= 	setup_rp_bins(pa.rp_min, pa.rp_max, pa.N_bins) # Edges
rp_cents	=	rp_bins_mid(rp_bins) # Centers

# Set up to get z as a function of comoving distance
z_of_com 	= 	z_interpof_com()

# Get the redshift corresponding to the maximum separation from the effective lens redshift at which we assume IA may be present (pa.close_cut is the separation in comoving Mpc/h)
(z_close_high, z_close_low)	= 	get_z_close(pa.zeff, pa.close_cut)

# Get the number of lens source pairs for the source sample in projected radial bins
N_ls_pbin	=	get_perbin_N_ls(rp_bins, pa.zeff, pa.n_s, pa.n_l, pa.Area, rp_cents)

# Get the fiducial value of gamma_IA in each projected radial bin (this takes a while so only do it once
fid_gIA		=	gamma_fid(rp_cents)

# Get the covariance matrix in projected radial bins of gamma_t for both shape measurement methods
Cov_a		=	setup_shapenoise_cov(pa.e_rms_a, N_ls_pbin)
Cov_b		=	setup_shapenoise_cov(pa.e_rms_b, N_ls_pbin)

# Combine the constituent covariance matrices to get the covariance matrix for gamma_IA in projected radial bins
Cov_stat	=	get_gammaIA_stat_cov(Cov_a, Cov_b, rp_cents, fid_gIA) 
#Cov_sys 	=	get_gammaIA_sys_cov(rp_cents, pa.sig_sys_dNdz, pa.sig_sys_dp, fid_gIA)

#Cov_tot		=	get_gamma_tot_cov(Cov_sys, Cov_stat)

Cov_tot = Cov_stat

print "PLOTTING ONLY STATISTICAL ERROR - NEED TO RECALIBRATE SYSTEMATIC ERROR."
# Output a plot showing the 1-sigma error bars on gamma_IA in projected radial bins
plot_variance(Cov_tot, fid_gIA, rp_cents)

exit() # Below is Fisher stuff, don't worry about this yet

# Get the parameter derivatives required to construct the Fisher matrix
ders            =       par_derivs(pa.par, rp_cents)

# Get the Fisher matrix
fish            =       get_Fisher(ders, Cov_gIA)

# If desired, cut parameters which you want to fix from Fisher matrix:
fish_cut        =       cut_Fisher(fish, None)

# Get the covariance matrix from either fish or fish_cut, and marginalise over any desired parameters
parCov          =       get_par_Cov(fish_cut, None)

# Output whatever we want to know about the parameters:
par_const_output(fish_cut, parCov)
