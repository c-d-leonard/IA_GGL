# This file contains functions for getting w_{l+} w_{ls} which are shared between the Blazek et al. 2012 method and the multiple shape measurements method.

import scipy.integrate
import matplotlib.pyplot as plt
import scipy.interpolate
import subprocess
import shutil
import numpy as np
import shared_functions_setup as setup
import os.path
import pyccl as ccl

# Functions shared between w_{l+} and w_{ls}

def window(survey):
	""" Get window function for w_{l+} and w_{ls} 2-halo terms."""
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	sigz = pa.sigz_gwin  # This is a very small value to basically set the lenses to a delta function.
	z = scipy.linspace(pa.zeff-5.*sigz, pa.zeff+5.*sigz, 50)
	dNdz_1 = 1. / np.sqrt(2. * np.pi) / sigz * np.exp(-(z-pa.zeff)**2 / (2. *sigz**2)) # Lens distribution - a very narrow Gaussian about the effective redshift.
	
	chi = setup.com(z, survey)
	OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
	dzdchi = pa.H0 * ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )**(0.5)
	
	(z, dNdz_2) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zeff-5.*sigz, pa.zeff+5.*sigz, 50)  
		
	norm = scipy.integrate.simps(dNdz_1*dNdz_2 / chi**2 * dzdchi, z)
	
	win = dNdz_1*dNdz_2 / chi**2 * dzdchi / norm
	
	return (z, win )

def get_Pk(z_,survey):
	""" Calls camb and returns the nonlinear halofit power spectrum for the current cosmological parameters and at the redshifts of interest. Returns a 2D array in k and z."""
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	# The redshifts have to be in a certain order to make camb happy:
	z_list = list(z_)
	z_list.reverse()
	z_rev = np.asarray(z_list)
	
	cambfolderpath = '/home/danielle/Documents/CMU/camb'
	param_string='output_root=IA_shapes\nombh2='+str(pa.OmB * (pa.HH0/100.)**2)+'\nomch2='+str(pa.OmC* (pa.HH0/100.)**2)+'\nhubble='+str(pa.HH0)+'\ntransfer_num_redshifts='+str(len(z_))
			
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
	(k, P) = np.loadtxt(cambfolderpath+'/IA_shapes_matterpower_z='+str(z_[0])+'.dat',unpack=True)
	
	Pofkz = np.zeros((len(z_),len(k))) 
	for zi in range(0,len(z_)):
		k, Pofkz[zi, :] = np.loadtxt(cambfolderpath+'/IA_shapes_matterpower_z='+str(z_[zi])+'.dat', unpack=True)
	
	return (k, Pofkz)

# Functions to get the 1halo term of w_{l+}

def get_pi(q1, q2, q3, z_):
	""" Returns the pi functions requires for the 1 halo term in wg+, at z_ """
	
	pi = q1 * np.exp(q2 * z_**q3)
	
	return pi

def get_P1haloIA(z, k, ah, q11, q12, q13, q21, q22, q23, q31, q32, q33):
	""" Returns the power spectrum required for the wg+ 1 halo term, at z and k_perpendicular ( = k) """
	
	p1 = get_pi(q11, q12, q13, z)
	p2 = get_pi(q21, q22, q23, z)
	p3 = get_pi(q31, q32, q33, z)
	
	#P1halo = ah * ( k / p1 )**2 / ( 1. + ( k /p2 )** p3)
	
	#plt.figure()
	#plt.loglog(k, P1halo, 'm+')
	#plt.ylim(10**(-18), 10**3)
	#plt.xlim(10**(-3), 10**3)
	#plt.savefig('./plots/P1halo_g+.png')
	#plt.close()
	
	P1halo = np.zeros((len(k), len(z)))
	for ki in range(0,len(k)):
		for zi in range(0,len(z)):
			P1halo[ki, zi]  = ah * ( k[ki] / p1[zi] )**2 / ( 1. + ( k[ki] /p2[zi] )** p3[zi])
	
	return P1halo

def growth(z_,survey):
	""" Returns the growth factor, normalized to 1 at z=0"""
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	def int_(z):
		OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
		return (1.+z) / ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )**(1.5)
	
	norm = scipy.integrate.quad(int_, 0, 1000.)[0]
	
	ans = np.zeros(len(z_))
	for zi in range(0,len(z_)):
		ans[zi] = scipy.integrate.quad(int_, z_[zi], 1000)[0]
	
	D = ans / norm
	
	return D

def wgp_1halo(rp_c_, bd, Ai, ah, q11, q12, q13, q21, q22, q23, q31, q32, q33, savefile, survey):
	""" Returns the 1 halo term of wg+(rp) """
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	(z, w) = window(survey) 
	
	# Set up a k vector to integrate over:
	k = np.logspace(-5., 7., 100000)
	
	# Get the `power spectrum' term
	P1h = get_P1haloIA(z, k, ah, q11, q12, q13, q21, q22, q23, q31, q32, q33)
	
	# First do the integral over z:
	zint = np.zeros(len(k))
	for ki in range(0,len(k)):
		zint[ki] = scipy.integrate.simps(P1h[ki, :] * w , z)
		
	#plot_quant_vs_quant(k, zint, './zint.png')
		
	# Now do the integral in k
	ans = np.zeros(len(rp_c_))
	for rpi in range(0,len(rp_c_)):
		integrand = k * zint * scipy.special.j0(rp_c_[rpi] * k)
		ans[rpi] = scipy.integrate.simps(integrand, k)
		#ans[rpi] = scipy.integrate.simps(k * P1h * scipy.special.j0(rp_c_[rpi] * k), k)
		
	# Set this to zero above about 2 * virial radius (I've picked this value somewhat aposteriori, should do better). This is to not include 1halo contributions well outside the halo.
	Rvir = Rhalo(10**16, survey)
	print "ASSUMING 10**16 FOR RMAX IN WGP1H"

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
	#plt.savefig('./plots/wg+_1h_ah=1.pdf')
	#plt.close()
	
	wgp_save = np.column_stack((rp_c_, wgp1h))
	np.savetxt(savefile, wgp_save)
		
	return wgp1h

def wgp_2halo(rp_cents_, bd, Ai, savefile, survey):
	""" Returns wgp from the nonlinear alignment model (2-halo term only). """
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	# Get the redshift window function
	z_gp, win_gp = window(survey)
	
	# Get the required matter power spectrum from camb 
	(k_gp, P_gp) = get_Pk(z_gp, survey)
	
	# Get the growth factor
	D_gp = growth(z_gp, survey)
	
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
		kz_int_gp[kpi] = scipy.integrate.simps(kpkz_gp[kpi,:] * kp_gp[kpi]**3 / ( (kp_gp[kpi]**2 + kz_gp**2)*kz_gp) * np.sin(kz_gp*pa.close_cut), kz_gp)
			
	# Finally, do the integrals in kperpendicular
	kp_int_gp = np.zeros(len(rp_cents_))
	for rpi in range(0,len(rp_cents_)):
		kp_int_gp[rpi] = scipy.integrate.simps(scipy.special.jv(2, rp_cents_[rpi]* kp_gp) * kz_int_gp, kp_gp)
		
	wgp_NLA = kp_int_gp * Ai * bd * pa.C1rho * (pa.OmC + pa.OmB) / np.pi**2
	
	wgp_stack = np.column_stack((rp_cents_, wgp_NLA))
	np.savetxt(savefile, wgp_stack)
	
	return wgp_NLA

def wgp_full(rp_c, bd, Ai, ah, q11, q12, q13, q21, q22, q23, q31, q32, q33, savefile_1h, savefile_2h, plotfile, survey):
	""" Combine 1 and 2 halo terms of wgg """
	
	# Check if savefile_1h exists, and if not, calculate 1 halo term.
	if (os.path.isfile(savefile_1h)):
		print "Loading wgp 1halo term from file"
		(rp_cen, wgp_1h) = np.loadtxt(savefile_1h, unpack=True)
	else:
		print "Computing wgp 1halo term"
		wgp_1h = wgp_1halo(rp_c, bd, Ai, ah, q11, q12, q13, q21, q22, q23, q31, q32, q33, savefile_1h, survey)
		
	# Check if savefile_2h exists, and if not, calculate 2 halo term.
	if (os.path.isfile(savefile_2h)):
		print "Loading wgp 2halo term from file "
		(rp_cen, wgp_2h) = np.loadtxt(savefile_2h, unpack=True)
	else:
		print "Computing wgp 2halo term"
		wgp_2h = wgp_2halo(rp_c, bd, Ai, savefile_2h, survey)
	
	wgp_tot = wgp_1h + wgp_2h 
	
	#plt.figure()
	#plt.loglog(rp_c, wgp_1h, 'go')
	#plt.xlim(0.05, 30.)
	#plt.ylim(0.01, 30)
	#plt.title('$w_{gp}$, 1halo')
	#plt.ylabel('$w_{gp}$, Mpc/h')
	#plt.xlabel('$r_p$, Mpc/h')
	#plt.savefig('./plots/wgp_1h_LRG.pdf')
	#plt.close()
	
	#plt.figure()
	#plt.loglog(rp_c, wgp_2h, 'go')
	#plt.xlim(0.1, 200.)
	#plt.ylim(0.01, 30)
	#plt.ylabel('$w_{gp}$, Mpc/h, com')
	#plt.xlabel('$r_p$, Mpc/h')
	#plt.title('$w_{gp}$, 2halo')
	#plt.savefig('./plots/wgp_2h_LRG.pdf')
	#plt.close()
	
	#plt.figure()
	#plt.loglog(rp_c, wgp_tot, 'go')
	#plt.xlim(0.05, 30.)
	#plt.ylabel('$w_{gp}$, Mpc/h')
	#plt.title('$w_{gp}$, 1halo+2halo')
	#plt.xlabel('$r_p$, Mpc/h')
	#plt.savefig(plotfile)
	#plt.close()
	
	return wgp_tot

	
# Functions to get the 1halo term of w_{ls}

def vol_dens(fsky, N,survey):
	""" Computes the volume density of galaxies given the fsky, minimum z, max z, and number of galaxies."""
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	V = fsky * 4. / 3. * np.pi * (setup.com(pa.zmax_dndz, survey)**3 - setup.com(pa.zmin_dndz, survey)**3)
	ndens = N / V
	return ndens

def Rhalo(M_insol, survey):
	""" Get the radius of a halo in COMOVING Mpc/h given its mass."""
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	#rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun * (pa. HH0 / 100.)) # Msol h^3 / Mpc^3, for use with M in Msol.
	rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h
	rho_m = rho_crit * pa.OmM
	Rvir = ( 3. * M_insol / (4. * np.pi * rho_m * 180.))**(1./3.)
	
	return Rvir

def cvir(M_insol):
	""" Returns the concentration parameter of the NFW profile, c_{vir}. """

	cvi = 5. * (M_insol / 10**14)**(-0.1)
	
	return cvi

def rho_s(cvi, Rvi, M_insol):
	""" Returns rho_s, the NFW parameter representing the density at the `scale radius', Rvir / cvir. Units: Mvir units * ( 1 / (Rvir units)**3), usualy Msol * h^3 / Mpc^3 with comoving distances. Sometimes also Msol h^2 / Mpc^3 (when Mvir is in Msol / h). """
	
	rhos = M_insol / (4. * np.pi) * ( cvi / Rvi)**3 * (np.log(1. + cvi) - (cvi / (1. + cvi)))**(-1)
	
	return rhos

def rho_NFW(r_, M_insol, survey):
	""" Returns the density for an NFW profile in real space at distance r from the center. Units = units of rhos. (Usually Msol * h^2 / Mpc^3 in comoving distances). r_ MUST be in the same units as Rv; usually Mpc / h."""

	Rv = Rhalo(M_insol, survey)
	cv = cvir(M_insol)
	rhos = rho_s(cv, Rv, M_insol)
	
	rho_nfw = rhos  / ( (cv * r_ / Rv) * (1. + cv * r_ / Rv)**2) 
	
	#plt.figure()
	#plt.loglog(r_ *1000, rho_nfw *  (pa.HH0/100.)**3 / (10**6)**3 )
	#plt.ylim(10**(-6), 10**4)
	#plt.xlim(0.01, 2000.)
	#plt.xlabel('$r$, comoving')
	#plt.savefig('./plots/nfw_vandeVen_lowmass.png')
	#plt.close()
	
	#plt.figure()
	#plt.loglog(r_ *1000 / (pa.HH0 /100.), rho_nfw *  (pa.HH0/100.)**3 / (10**6)**3 )
	#plt.ylim(10**(-6), 10**4)
	#plt.xlim(0.03, 2000.)
	#plt.savefig('./plots/nfw_Mandelbaun2006.png')
	#plt.close()
	
	#plt.figure()
	#plt.loglog(r_ , rho_nfw  )
	#plt.ylim(10**(11), 2.*10**18)
	#plt.xlim(0.001, 3)
	#plt.savefig('./plots/nfw_Singh2014_z=0.28.png')
	#plt.close()
	
	return rho_nfw

def wgg_1halo_Four(rp_cents_, fsat, fsky, savefile, survey):
	""" Gets the 1halo term of wgg via Fourier space, to account for central-satelite pairs and satelite-satelite pairs. """
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	logkmin = -6; kpts =40000; logkmax = 5; Mmax = 16;
	# Compute P_{gg}^{1h}(k)
	kvec_FT = np.logspace(logkmin, logkmax, kpts)
	
	#print "Before getting P_{gg}(k)"
	#Pk = get_Pkgg_1halo(kvec_FT, fsat, fsky, Mmax, survey) # Gets the 1halo galaxy power spectrum including c-s and s-s terms. Pass rvec because need to get rho_NFW in here.
	#(k_dummy, Pk) = np.loadtxt('./txtfiles/Pkgg_1h_dndM.txt', unpack=True)
	#print "We've now computed Pkgg - exiting now, get the fourier transform via external fft log and load result."
	#exit()
	
	# This function loads the xi_{gg}1h function computed from FFTlog externally
	(rvec_xi, xi_gg_1h) = get_xi_1h(pa.zeff)
	
	#plt.figure()
	#plt.loglog(rvec_xi, xi_gg_1h, 'm+')
	#plt.xlim(10**(-4), 10**3)
	#plt.savefig('./plots/xigg_dndMtest_FFTlog_logkmin='+str(logkmin)+'_logkmax='+str(logkmax)+'_kpts='+str(kpts)+'rpts1e6_loginterp.pdf')
	#plt.close()
	
	# Get the max R associated to our max M
	Rmax = Rhalo(10**Mmax, survey)
	# Set xi_gg_1h to zero above Rmax Mpc/h.
	for ri in range(0, len(rvec_xi)):
		if (rvec_xi[ri]>Rmax):
			xi_gg_1h[ri] = 0.0
			
	#plt.figure()
	#plt.loglog(rvec_xi, xi_gg_1h)
	#plt.savefig('./plots/xigg_dndMtest_cut_survey='+survey+'.pdf')
	#plt.close()
	
	#save_xi_gg = np.column_stack((rvec_xi, xi_gg_1h))		
	#np.savetxt('./txtfiles/xi_gg_1halo_testdndM.txt', save_xi_gg)
	
	xi_interp = scipy.interpolate.interp1d(rvec_xi, xi_gg_1h)
	
	# Get xi_{gg}1h as a function of rp and Pi
	Pivec = np.logspace(-7, np.log10(1000./np.sqrt(2.01)), 1000)
	
	xi_2D = np.zeros((len(rp_cents_), len(Pivec)))
	for ri in range(0, len(rp_cents_)):
		for pi in range(0, len(Pivec)):
			xi_2D[ri, pi] = xi_interp(np.sqrt(rp_cents_[ri]**2 + Pivec[pi]**2)) 
	
	wgg_1h = np.zeros(len(rp_cents_))
	for ri in range(0,len(rp_cents_)):
		wgg_1h[ri] = scipy.integrate.simps(xi_2D[ri, :], Pivec)
	
	wgg_save = np.column_stack((rp_cents_, wgg_1h))
	np.savetxt(savefile, wgg_save)
	
	return wgg_1h
	
def get_xi_1h(z):
	""" Returns the 1 halo galaxy correlation function including cen-sat and sat-sat terms, from the power spectrum via Fourier transform."""
	
	(r, xi) = np.loadtxt('./txtfiles/xi_z='+str(z)+'.txt', unpack=True)
	
	#xi = np.zeros(len(r))
	#for ri in range(0,len(r)):
	#	xi[ri] = scipy.integrate.simps(Pofk * kvec**2 * np.sin(kvec*r[ri]) / (kvec * r[ri]) / 2. / np.pi**2 , kvec)
	
	return (r, xi)
	
def get_Pkgg_1halo(kvec_ft, fsat, fsky, Mmax, survey):
	""" Returns the 1halo galaxy power spectrum with c-s and s-s terms."""
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	Mhalo = np.logspace(9., Mmax, 30)
	# Get the lower stellar mass cutoff for source satelites corresponding to the total empirical volume density of the source sample:
	tot_nsrc = vol_dens(pa.fsky, pa.N_shapes, survey)
	Mstarlow = get_Mstar_low(survey, tot_nsrc)
	
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = 2.1*10**(-9), n_s=0.96)
	cosmo = ccl.Cosmology(p)
	HMF = ccl.massfunction.massfunc(cosmo, Mhalo / (pa.HH0/100.), 1./ (1. + pa.zeff), odelta=200.)
	
	# We're going to use, for the centrals and satelite lenses, either the Reid & Spergel 2008 HOD (SDSS LRGs) or the CMASS More et al. 2014 HOD (DESI LRGS).
	# For the satellite sources, we use either the Zu & Mandelbaum 2015 model (SDSS shape sample) or the CMASS HOD again. We assume none of the sources are centrals in galaxies with satellites from the lenses.
	if (survey == 'SDSS'):
		Ncen_lens = get_Ncen_Reid(Mhalo, survey)  # Reid & Spergel
		Nsat_lens = get_Nsat_Reid(Mhalo, survey)  # Reid & Spergel 
		Nsat_src = get_Nsat(Mhalo, Mstarlow, survey)  # Zu & Mandelbaum 2015
		Ncen_src = get_Ncen(Mhalo, Mstarlow, survey)  # Zu & Mandelbaum 2015
	elif (survey== 'LSST_DESI'):
		Ncen_lens = get_Ncen(Mhalo, 'nonsense', survey) # CMASS
		Nsat_lens = get_Nsat(Mhalo, 'nonsense', survey) # CMASS for both
		Nsat_src = get_Nsat(Mhalo, 'nonsense', survey)  # CMASS for both
		
	
	
	# Get the number density predicted by the halo model (not the one for the actual survey)
	#tot_nsrc =  scipy.integrate.simps( (Nsat_src) * HMF, np.log10(Mhalo / (pa.HH0/100.) ) ) / (pa.HH0 / 100.)**3
	tot_ng = scipy.integrate.simps( ( Ncen_lens + Nsat_lens) * HMF, np.log10(Mhalo / (pa.HH0/100.) ) ) / (pa.HH0 / 100.)**3
	tot_nsrc_sat = scipy.integrate.simps(( Nsat_src ) * HMF, np.log10(Mhalo / (pa.HH0/100.) ) ) / (pa.HH0 / 100.)**3
	print "nsrc=", tot_nsrc, "nlens=", tot_ng, "tot nsrc check=", tot_nsrc_sat
	
	# Get the density of matter in comoving coordinates
	rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h (comoving distances)
	rho_m = pa.OmM * rho_crit # units of Msol h^2 / Mpc^3 (comoving distances)
	
	#alpha_sq = get_alpha_sq(Mhalo) # The number which accounts for deviation from Poisson statisticsal
	alpha_sq = np.ones(len(Mhalo))
	print "alpha=", alpha_sq
	NcNs = NcenNsat(alpha_sq, Ncen_lens, Nsat_src) # The average number of central-satelite pairs in a halo of mass M
	NsNs = NsatNsat(alpha_sq, Nsat_lens, Nsat_src) # The average number of satelite-satelite pairs in a halo of mass M
	
	# Get a downsampled kvec for these calculations:
	kvec_short = np.logspace(np.log10(kvec_ft[0]), np.log10(kvec_ft[-1]), 40)
	
	# Get ingredients we need here:
	y = gety(Mhalo, kvec_short, survey) # Mass-averaged Fourier transform of the density profile

	Pkgg = np.zeros(len(kvec_short))
	for ki in range(0,len(kvec_short)):
		Pkgg[ki] = scipy.integrate.simps( HMF * (NcNs * y[ki, :] + NsNs * y[ki, :]**2), np.log10(Mhalo / (pa.HH0/100.)  )) / (tot_nsrc_sat * tot_ng) / (pa.HH0 / 100.)**3
		print "k=", kvec_short[ki], "Pkgg=", Pkgg[ki]
	
	plt.figure()
	plt.loglog(kvec_short, 4* np.pi * kvec_short**3 * Pkgg / (2* np.pi)**3, 'mo')
	plt.ylim(0.001, 100000)
	plt.xlim(0.01, 10000)
	plt.ylabel('$4\pi k^3 P_{gg}^{1h}(k)$, $(Mpc/h)^3$, com')
	plt.xlabel('$k$, h/Mpc, com')
	plt.savefig('./plots/Pkgg_1halo_HMFint_Mmax15_CMASS.pdf')
	plt.close()
	
	Pkgg_interp = scipy.interpolate.interp1d(np.log(kvec_short), np.log(Pkgg))
	logPkgg = Pkgg_interp(np.log(kvec_ft))
	Pkgg = np.exp(logPkgg)
	
	Pkgg_save = np.column_stack((kvec_ft, Pkgg))
	np.savetxt('./txtfiles/Pkgg_1h_dndM_'+survey+'.txt', Pkgg_save)
	
	plt.figure()
	plt.loglog(kvec_ft, 4* np.pi * kvec_ft**3 * Pkgg / (2* np.pi)**3, 'mo')
	plt.ylim(0.001, 100000)
	plt.xlim(0.01, 10000)
	plt.ylabel('$4\pi k^3 P_{gg}^{1h}(k)$, $(Mpc/h)^3$, com')
	plt.xlabel('$k$, h/Mpc, com')
	plt.savefig('./plots/Pkgg_1halo_longerkvec_Mmax16.pdf')
	plt.close()
	
	return Pkgg

def get_Pkgg_ll_1halo_kz(kvec, zvec, survey):
	""" Returns the 1halo galaxy power spectrum with c-s and s-s terms for lenses x lenses (for the covariance). """

	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	# Define the downsampled k and z vector over which we will compute Pk_{gm}^{1h}
	kvec_short = np.logspace(np.log10(kvec[0]), np.log10(kvec[-1]), 40)
	zvec_short = np.linspace(zvec[0]-0.00000001, zvec[-1]+0.00000001, 40)
	Mhalo = np.logspace(7., 16., 30)
	
	# Get the halo mass function at each z (use CCL)
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = 2.1*10**(-9), n_s=0.96)
	cosmo = ccl.Cosmology(p)
	HMF = np.zeros((len(Mhalo), len(zvec_short)))
	for zi in range(0,len(zvec_short)):
		HMF[:, zi]= ccl.massfunction.massfunc( cosmo, Mhalo / (pa.HH0/100.), 1./ (1. + zvec_short[zi]), odelta=200. )
	
	# We're going to use, for the centrals and satelite lenses, either the Reid & Spergel 2008 HOD (SDSS LRGs) or the CMASS More et al. 2014 HOD (DESI LRGS).
	if (survey == 'SDSS'):
		Ncen_lens = get_Ncen_Reid(Mhalo, survey)  # Reid & Spergel
		Nsat_lens = get_Nsat_Reid(Mhalo, survey)  # Reid & Spergel 
	elif (survey== 'LSST_DESI'):
		Ncen_lens = get_Ncen(Mhalo, 'nonsense', survey) # CMASS
		Nsat_lens = get_Nsat(Mhalo, 'nonsense', survey) # CMASS for both
	
	# Get the number density predicted by the halo model (not the one for the actual survey)
	tot_ng = np.zeros(len(zvec_short))
	for zi in range(0, len(zvec_short)):
		tot_ng[zi] = scipy.integrate.simps( ( Ncen_lens + Nsat_lens) * HMF[:, zi], np.log10(Mhalo / (pa.HH0/100.) ) ) / (pa.HH0 / 100.)**3
		print "nlens=", tot_ng[zi]
	
	# Get the density of matter in comoving coordinates
	rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h (comoving distances)
	rho_m = pa.OmM * rho_crit # units of Msol h^2 / Mpc^3 (comoving distances)
	
	#alpha_sq = get_alpha_sq(Mhalo) # The number which accounts for deviation from Poisson statisticsal
	alpha_sq = np.ones(len(Mhalo))
	NcNs = NcenNsat(alpha_sq, Ncen_lens, Nsat_lens) # The average number of central-satelite pairs in a halo of mass M
	NsNs = NsatNsat(alpha_sq, Nsat_lens, Nsat_lens) # The average number of satelite-satelite pairs in a halo of mass M
	
	y = gety(Mhalo, kvec_short, survey) # Mass-averaged Fourier transform of the density profile

	Pkgg = np.zeros((len(kvec_short), len(zvec_short)))
	for ki in range(0,len(kvec_short)):
		for zi in range(0, len(zvec_short)):
			Pkgg[ki, zi] = scipy.integrate.simps( HMF[:, zi] * (NcNs * y[ki, :] + NsNs * y[ki, :]**2), np.log10(Mhalo / (pa.HH0/100.)  )) / (tot_ng[zi]**2) / (pa.HH0 / 100.)**3
	
	# Get this in terms of the right k and z vectors:
	logPkgg_interp_atz = [0]*len(zvec_short)
	Pkgg_correctk_shortz = np.zeros((len(kvec), len(zvec_short)))
	for zi in range(0,len(zvec_short)):
		logPkgg_interp_atz[zi] = scipy.interpolate.interp1d(np.log(kvec_short), np.log(Pkgg[:,zi]))
		Pkgg_correctk_shortz[:, zi] = np.exp(logPkgg_interp_atz[zi](np.log(kvec)))
		
	Pkgg_interp_atk = [0]*len(kvec)
	Pkgg_correctkz = np.zeros((len(kvec), len(zvec)))
	for ki in range(0,len(kvec)):
		Pkgg_interp_atk[ki] = scipy.interpolate.interp1d(zvec_short, Pkgg_correctk_shortz[ki, :])
		Pkgg_correctkz[ki, :] = Pkgg_interp_atk[ki](zvec)
	
	#Pkgm_save = np.column_stack((kvec_FT, Pkgm))
	#np.savetxt('./txtfiles/Pkgm_1h_dndM_survey='+SURVEY+'.txt', Pkgm_save)
	return Pkgg_correctkz

def get_Pkgm_1halo_kz(kvec, zvec, survey):
	""" Returns the 1halo lens galaxies x dark matter power spectrum at the given k and z values """
	
	# Get the average halo mass:
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	# Define the downsampled k and z vector over which we will compute Pk_{gm}^{1h}
	kvec_short = np.logspace(np.log10(kvec[0]), np.log10(kvec[-1]), 40)
	zvec_short = np.linspace(zvec[0]-0.00000001, zvec[-1]+0.00000001, 40)
	Mhalo = np.logspace(7., 16., 30)
	
	# Get the halo mass function at each z (use CCL)
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = 2.1*10**(-9), n_s=0.96)
	cosmo = ccl.Cosmology(p)
	HMF = np.zeros((len(Mhalo), len(zvec_short)))
	for zi in range(0,len(zvec_short)):
		HMF[:, zi]= ccl.massfunction.massfunc( cosmo, Mhalo / (pa.HH0/100.), 1./ (1. + zvec_short[zi]), odelta=200. )
	
	if (survey=='SDSS'):
		Ncen_lens = get_Ncen_Reid(Mhalo, survey) # We use the LRG model for the lenses from Reid & Spergel 2008
		Nsat_lens = get_Nsat_Reid(Mhalo, survey) 
	elif (survey=='LSST_DESI'):
		Ncen_lens = get_Ncen(Mhalo, 'nonsense', survey)
		Nsat_lens = get_Nsat(Mhalo, 'nonsense', survey)
	else:
		print "We don't have support for that survey yet!"
		exit()
		
	# Get total number of galaxies (this is z-dependent) 
	tot_ng = np.zeros(len(zvec_short))
	for zi in range(0,len(zvec_short)):
		tot_ng[zi] = scipy.integrate.simps( ( Ncen_lens + Nsat_lens) * HMF[:, zi], np.log10(Mhalo / (pa.HH0/100.) ) ) / (pa.HH0 / 100.)**3
		# Because the number density comes out a little different than the actual case, especially for DESI, we are going to use this number to get the right normalization.
		print "tot=", tot_ng[zi]

	# Get the fourier space NFW profile equivalent
	y = gety(Mhalo, kvec_short, survey) 
	
	# Get the density of matter in comoving coordinates
	rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h (comoving distances)
	rho_m = pa.OmM * rho_crit # units of Msol h^2 / Mpc^3 (comoving distances)
	
	# Get Pk
	Pkgm = np.zeros((len(kvec_short), len(zvec_short)))
	for ki in range(0,len(kvec_short)):
		for zi in range(0,len(zvec_short)):
			Pkgm[ki, zi] = scipy.integrate.simps( HMF[:, zi] * (Mhalo / rho_m) * (Ncen_lens * y[ki, :] + Nsat_lens * y[ki, :]**2), np.log10(Mhalo / (pa.HH0/ 100.))) / (tot_ng[zi]) / (pa.HH0 / 100.)**3
	
	# Get this in terms of the right k and z vectors:
	logPkgm_interp_atz = [0]*len(zvec_short)
	Pkgm_correctk_shortz = np.zeros((len(kvec), len(zvec_short)))
	for zi in range(0,len(zvec_short)):
		logPkgm_interp_atz[zi] = scipy.interpolate.interp1d(np.log(kvec_short), np.log(Pkgm[:,zi]))
		Pkgm_correctk_shortz[:, zi] = np.exp(logPkgm_interp_atz[zi](np.log(kvec)))
		
	Pkgm_interp_atk = [0]*len(kvec)
	Pkgm_correctkz = np.zeros((len(kvec), len(zvec)))
	for ki in range(0,len(kvec)):
		Pkgm_interp_atk[ki] = scipy.interpolate.interp1d(zvec_short, Pkgm_correctk_shortz[ki, :])
		Pkgm_correctkz[ki, :] = Pkgm_interp_atk[ki](zvec)
	
	#Pkgm_save = np.column_stack((kvec_FT, Pkgm))
	#np.savetxt('./txtfiles/Pkgm_1h_dndM_survey='+SURVEY+'.txt', Pkgm_save)
	return Pkgm_correctkz
	
def get_Pkmm_1halo_kz(kvec, zvec, survey):
	""" Returns the 1halo lens galaxies x dark matter power spectrum at the given k and z values """
	
	# Get the average halo mass:
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	# Define the downsampled k and z vector over which we will compute Pk_{gm}^{1h}
	kvec_short = np.logspace(np.log10(kvec[0]), np.log10(kvec[-1]), 40)
	zvec_short = np.linspace(zvec[0]-0.00000001, zvec[-1]+0.00000001, 40)
	Mhalo = np.logspace(7., 16., 30)
	
	# Get the halo mass function at each z (use CCL)
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = 2.1*10**(-9), n_s=0.96)
	cosmo = ccl.Cosmology(p)
	HMF = np.zeros((len(Mhalo), len(zvec_short)))
	for zi in range(0,len(zvec_short)):
		HMF[:, zi]= ccl.massfunction.massfunc( cosmo, Mhalo / (pa.HH0/100.), 1./ (1. + zvec_short[zi]), odelta=200. )

	# Get the fourier space NFW profile equivalent
	y = gety(Mhalo, kvec_short, survey) 
	
	# Get the density of matter in comoving coordinates
	rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h (comoving distances)
	rho_m = pa.OmM * rho_crit # units of Msol h^2 / Mpc^3 (comoving distances)
	
	# Get Pk
	Pkmm = np.zeros((len(kvec_short), len(zvec_short)))
	for ki in range(0,len(kvec_short)):
		for zi in range(0,len(zvec_short)):
			Pkmm[ki, zi] = scipy.integrate.simps( HMF[:, zi] * (Mhalo / rho_m)**2 * y[ki, :]**2 , np.log10(Mhalo / (pa.HH0/ 100.))) / (pa.HH0 / 100.)**3
	
	# Get this in terms of the right k and z vectors:
	logPkmm_interp_atz = [0]*len(zvec_short)
	Pkmm_correctk_shortz = np.zeros((len(kvec), len(zvec_short)))
	for zi in range(0,len(zvec_short)):
		logPkmm_interp_atz[zi] = scipy.interpolate.interp1d(np.log(kvec_short), np.log(Pkmm[:,zi]))
		Pkmm_correctk_shortz[:, zi] = np.exp(logPkmm_interp_atz[zi](np.log(kvec)))
	
	Pkmm_interp_atk = [0]*len(kvec)
	Pkmm_correctkz = np.zeros((len(kvec), len(zvec)))
	for ki in range(0,len(kvec)):
		Pkmm_interp_atk[ki] = scipy.interpolate.interp1d(zvec_short, Pkmm_correctk_shortz[ki, :])
		Pkmm_correctkz[ki, :] = Pkmm_interp_atk[ki](zvec)
		
	#Pkgm_save = np.column_stack((kvec_FT, Pkgm))
	#np.savetxt('./txtfiles/Pkgm_1h_dndM_survey='+SURVEY+'.txt', Pkgm_save)
	return Pkmm_correctkz
	
def gety(Mvec, kvec_gety, survey):
	""" Fourier transforms the density profile to get the power spectrum. """
	
	# Get the nfw density profile at the correct mass and redshift and at a variety of r
	rvec = [0]*len(Mvec)
	rho = [0]*len(Mvec)
	for Mi in range(0,len(Mvec)):
		print "Mi=", Mi
		Rvir = Rhalo(Mvec[Mi], survey)
		print "M=", Mvec[Mi], "Rvir=", Rvir
		rvec[Mi] = np.logspace(-8, np.log10(Rvir), 10**6)
		rho[Mi] = rho_NFW(rvec[Mi], Mvec[Mi], survey)  # Units Msol h^2 / Mpc^3, comoving. 

	u_ = np.zeros((len(kvec_gety), len(Mvec)))
	for ki in range(0,len(kvec_gety)):
		print "k=", kvec_gety[ki]
		for mi in range(0,len(Mvec)):
			u_[ki, mi] = 4. * np.pi / Mvec[mi] * scipy.integrate.simps( rvec[mi] * np.sin(kvec_gety[ki]*rvec[mi])/ kvec_gety[ki] * rho[mi], rvec[mi]) # unitless / dimensionless.
	
	return u_
	
def get_alpha_sq(M):
	""" Gets the parameter that accounts for the deviation from Poisson statistics. M is in Msol / h. """
	
	alpha = np.zeros(len(M))
	for Mi in range(0,len(M)):
		if (M[Mi]<10**11):
			alpha[Mi] = np.log(M[Mi] / 10**11)
		else:
			alpha[Mi] = 1.
		
	return alpha

def get_Mh_avg(Mlow_star, ngvol, survey):
	""" This function gets the average halo mass for the sample """
	
	# Get the high and low edges of halo mass distribution for this sample
	Mh_low = get_Mhlow( Mlow_star, survey )
	Mh_high = 10**16 # This is the max value included in calculating M*; alternatively just use a very large value, large enough the HMF is practically 0.
	
	# Get the average halo mass:
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	Mh_test = np.logspace(np.log10(Mh_low), np.log10(Mh_high), 10000)
	# Get the halo mass function (from CCL) to integrate over (dn / dlog10M, Tinker 2010 I think)
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = 2.1*10**(-9), n_s=0.96)
	cosmo = ccl.Cosmology(p)
	HMF = ccl.massfunction.massfunc(cosmo, Mh_test / (pa.HH0/100.), 1./ (1. + pa.zeff), odelta=200.)
	#Ncen = get_Ncen_src(Mh_test, Mlow_star, survey)
	if (survey=='SDSS'):
		if ((pa.Use_Reid_HOD)==True):
			Ncen = get_Ncen_Reid(Mh_test, survey)
			print "Using Reid HOD for SDSS LRGs"
		else:
			Ncen= get_Ncen(Mh_test, Mlow_star, survey)
			print "Using Zu&Mandelbaum HOD"
	else:
		if ((pa.Use_Reid_HOD)==True):
			Ncen = get_Ncen_Reid(Mh_test, survey)
			print "Using Reid HOD for SDSS LRGs"
		else:
			Ncen = get_Ncen(Mh_test, Mlow_star, survey)
			print "Using Zu&Mandelbaum HOD"
	
	logMh_bins = np.linspace(np.log10(Mh_low), np.log10(Mh_high),20)
	
	n_in_bin = np.zeros(len(logMh_bins)-1) # Number density of halos with at least a central galaxy in this bin.
	for i in range(0, len(logMh_bins)-1):
		ind_low = next(j[0] for j in enumerate(Mh_test) if j[1]>=10**(logMh_bins[i]))
		ind_high = next(j[0] for j in enumerate(Mh_test) if j[1]>=10**(logMh_bins[i+1]))
		n_in_bin[i] = scipy.integrate.simps(HMF[ind_low:ind_high]*Ncen[ind_low:ind_high], np.log10(Mh_test)[ind_low:ind_high])
	
	logMcentres = np.zeros(len(logMh_bins)-1)
	for i in range(0,len(logMcentres)):
		logMcentres[i] = logMh_bins[i] + (logMh_bins[i+1] - logMh_bins[i])
		
	avg = 10**(np.sum( n_in_bin * logMcentres) / np.sum(n_in_bin))
	
	return avg

def get_Mhlow( Mlow_star, survey):
	""" This function gets the lowest value of the halo mass for which the halo is at all likely to host a galaxy, for this sample """
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	
	Mh_test = np.logspace(9., 16., 1000)
	if (survey=='SDSS'):
		if ((pa.Use_Reid_HOD)==True):
			Ncen_of_Mh = get_Ncen_Reid(Mh_test, survey)
			Nsat_of_Mh = get_Nsat_Reid(Mh_test, survey)
		else:
			Ncen_of_Mh = get_Ncen(Mh_test, Mlow_star, survey)
			Nsat_of_Mh = get_Nsat(Mh_test, Mlow_star, survey)
	else:
		Ncen_of_Mh = get_Ncen(Mh_test, Mlow_star, survey)
		Nsat_of_Mh = get_Nsat(Mh_test, Mlow_star, survey)
	
	not_zero = 0.0001
	ind = next(j[0] for j in enumerate(Ncen_of_Mh) if j[1]>=not_zero)
	
	Mhlow = Mh_test[ind]
	print "Mhalo lower bound=", Mhlow
	
	return Mhlow
	
def get_Mhhigh( Mh_low, Mlow_star, ngvol, survey ):
	""" This function gets the max value of the halos mass that should be included to match the galaxy density of the sample. """
	""" Turns out this is definitionally the highest value of the halo mass includes when computing the stellar mass cutoff, so I'm not using this function, but I'm going to keep hold of it for now in case I need it."""
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	Mh_test = np.logspace(np.log10(Mh_low), 16., 1000)
	if (survey=='SDSS'):
		if ((pa.Use_Reid_HOD)==True):
			Ncen_of_Mh = get_Ncen_Reid(Mh_test, survey)
			Nsat_of_Mh = get_Nsat_Reid(Mh_test, survey)
		else:
			Ncen_of_Mh = get_Ncen(Mh_test, Mlow_star, survey)
			Nsat_of_Mh = get_Nsat(Mh_test, Mlow_star, survey)
	else:
		Ncen_of_Mh = get_Ncen(Mh_test, Mlow_star, survey)
		Nsat_of_Mh = get_Nsat(Mh_test, Mlow_star, survey)
	
	# Get the halo mass function (from CCL) to integrate over (dn / dlog10M, Tinker 2010 I think)
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = 2.1*10**(-9), n_s=0.96)
	cosmo = ccl.Cosmology(p)
	HMF = ccl.massfunction.massfunc(cosmo, Mh_test / (pa.HH0/100.), 1./ (1. + pa.zeff), odelta=200.)
	
	ng_of_Mh = np.zeros(len(Mh_test))
	for i in range(1,len(Mh_test)):
		#ng_of_Mh[i] = scipy.integrate.simps(HMF[0:i] * (Ncen_of_Mh[0:i] + Nsat_of_Mh[0:i]), np.log10(Mh_test)[0:i])
		ng_of_Mh[i] = scipy.integrate.simps(HMF * (Ncen_of_Mh + Nsat_of_Mh), np.log10(Mh_test))
	print "ng=", ng_of_Mh
		
	ind = next(j[0] for j in enumerate(ng_of_Mh) if j[1]>=ngvol)
	
	Mhigh = Mh_test[ind]
	
	return Mhigh
	

def get_Mstar_low(survey, ngal):
	""" For a given number density of source galaxies (calculated in the vol_dens function), get the appropriate choice for the lower bound of Mstar """
	
	if (survey == 'SDSS'):
		import params as pa
	
		# Use the HOD model from Zu & Mandelbaum 2015
		
		# Define a vector of Mstar_low value to try
		Ms_low_vec = np.logspace(9., 12.,1000)
		# Define a vector of Mh values to integrate over
		Mh_vec = np.logspace(9., 16., 1000)
	
		# Get Nsat and Ncen as a function of the values of the two above arrays
		Nsat = get_Nsat(Mh_vec, Ms_low_vec, survey)
		Ncen = get_Ncen(Mh_vec, Ms_low_vec, survey)
	
		# Get the halo mass function (from CCL) to integrate over (dn / dlog10M, Tinker 2010 I think)
		p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = 2.1*10**(-9), n_s=0.96)
		cosmo = ccl.Cosmology(p)
		HMF = ccl.massfunction.massfunc(cosmo, Mh_vec / (pa.HH0/100.), 1./ (1. + pa.zeff), odelta=200.) 
	
		# Now get what nsrc should be for each Mstar_low cut 
		nsrc_of_Mstar = np.zeros(len(Ms_low_vec))
		for i in range(0,len(Ms_low_vec)):
			nsrc_of_Mstar[i] = scipy.integrate.simps(HMF * ( Nsat[i, :] + Ncen[i, :]), np.log10(Mh_vec / (pa.HH0 / 100.))) / (pa.HH0 / 100.)**3
	
		# Get the correct Mstar cut	
		ind = next(j[0] for j in enumerate(nsrc_of_Mstar) if j[1]<=ngal)
		
		Mstarlow = Ms_low_vec[ind]
		
	elif (survey == 'LSST_DESI'):
		# The HOD model we use for LSST_DESI, that from More et al. 2014 for CMASS, does not take an Mstarlow, so we return a dummy variable
		Mstarlow = 'nonsense'
		
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	return Mstarlow
	
def get_Nsat(M_h, Mstar, survey):
	""" Gets the fraction of source galaxies that are satelites in halos associated with the lens sample, for stellars masses AT OR ABOVE the average one for our source sample using the HOD model from Zu & Mandelbaum 2015."""
	
	if (survey == 'SDSS'):
		import params as pa
	
		f_Mh = fSHMR_inverse(Mstar, survey)
		Ncen_src = get_Ncen(M_h, Mstar,survey)
		Msat = get_Msat(f_Mh, survey)
		Mcut = get_Mcut(f_Mh, survey)
	
		if ((type(M_h)==float) or (type(Mstar)==float) or(type(M_h)==np.float64) or (type(Mstar)==np.float64) ):
			Nsat = Ncen_src * (M_h / Msat)**(pa.alpha_sat) * np.exp(-Mcut / M_h)
		elif(((type(Mstar)==list) or isinstance(Mstar, np.ndarray)) and ((type(M_h)==list) or isinstance(M_h, np.ndarray))):
			Nsat=np.zeros((len(M_h), len(Mstar)))
			for i in range(0,len(Mstar)):
				for j in range(0,len(M_h)):
					Nsat[i,j] = Ncen_src[i,j] * (M_h[j] / Msat[i])**(pa.alpha_sat) * np.exp(-Mcut[i] / M_h[j])			
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
		
		Ncen_src = get_Ncen(M_h, Mstar, survey)
		
		Nsat = np.zeros(len(M_h))
		for mi in range(0,len(M_h)):
			if ( M_h[mi]> ( pa.kappa_CMASS * pa.Mmin_CMASS ) ):
				Nsat[mi] = Ncen_src[mi] * ( ( M_h[mi] - pa.kappa_CMASS * pa.Mmin_CMASS) / pa.M1_CMASS)**pa.alpha_CMASS
			else:
				Nsat[mi] = 0.	
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
		
	return Nsat
	
def get_Ncen(Mh, Mstar, survey):
	""" Get the CUMULATIVE distribution of central galaxies for the sources from the HOD model from Zu & Mandelbaum 2015"""
	
	if (survey == 'SDSS'):
		import params as pa
		
		# This is for the Zu & Mandelbaum 2015 halo model.
		sigmaMstar = get_sigMs(Mh, survey)
		fshmr = get_fSHMR(Mh, survey)
	
		if ((type(Mstar)==float) or (type(Mh)==float) or(type(Mh)==np.float64) or (type(Mstar)==np.float64) ):
			Ncen_CDF = 0.5 * (1. - scipy.special.erf((np.log(Mstar) - np.log(fshmr)) / (np.sqrt(2.) * sigmaMstar)))
		elif(((type(Mstar)==list) or (isinstance(Mstar, np.ndarray))) and ((type(Mh)==list) or isinstance(Mh, np.ndarray))):
			Ncen_CDF = np.zeros((len(Mstar), len(Mh)))
			for i in range(0,len(Mstar)):
				for j in range(0, len(Mh)):
					Ncen_CDF[i,j] = 0.5 * (1. - scipy.special.erf((np.log(Mstar[i]) - np.log(fshmr[j])) / (np.sqrt(2.) * sigmaMstar[j])))
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
		# This is for the More et al. 2014 CMASS HOD
		Ncen_CDF = np.zeros(len(Mh))
		finc = np.zeros(len(Mh))
		for mi in range(0,len(Mh)):
			finc[mi] = max(0, min(1., 1. + pa.alphainc_CMASS * (np.log10(Mh[mi]) - np.log10(pa.Minc_CMASS))))
			Ncen_CDF[mi] = finc[mi] * 0.5 * (1. + scipy.special.erf( (np.log10(Mh[mi]) - np.log10(pa.Mmin_CMASS)) / pa.siglogM_CMASS))
			#print "finc=", finc
			
		plt.figure()
		plt.loglog(Mh, finc, 'go')
		plt.xlim(10**12, 2*10**15)
		plt.ylim(0.1, 10)
		plt.savefig('./plots/finc.pdf')
		plt.close()
			
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	return Ncen_CDF

def get_Ncen_Reid(Mh, survey):
	""" Get the cumulative distribution of central galaxies for the SDSS LRG sample from Reid & Spergel 2008. """
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	Ncen = 0.5 * (1. + scipy.special.erf((np.log10(Mh / (pa.HH0/100.)) - np.log10(pa.Mmin_reid)) / pa.sigLogM_reid))
	
	return Ncen 
	
def get_Nsat_Reid(Mh, survey):
	""" Get the number of satellite galaxies per halo. """
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	Ncen = get_Ncen_Reid(Mh, survey)
	
	Nsat = np.zeros(len(Mh))
	for i in range(0,len(Mh)):
		if ( (Mh[i] / (pa.HH0/100.)) >= pa.Mcut_reid):
			Nsat[i] = Ncen[i] * ((Mh[i] / (pa.HH0/100.) - pa.Mcut_reid) / pa.M1_reid)**(pa.alpha_reid)
		else:
			Nsat[i] = 0.
	
	return Nsat
	
def get_sigMs(Mh, survey):
	""" Get sigma_ln(M*) as a function of the halo mass."""
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	if (type(Mh)==float):
	
		if (Mh<pa.M1):
			sigM = pa.sigMs
		else:
			sigM = pa.sigMs + pa.eta * np.log10( Mh / pa.M1)
	elif ((type(Mh) == list) or (isinstance(Mh, np.ndarray))):
		sigM = np.zeros(len(Mh))
		for i in range(0,len(Mh)):
			if (Mh[i]<pa.M1):
				sigM[i] = pa.sigMs
			else:
				sigM[i] = pa.sigMs + pa.eta * np.log10( Mh[i] / pa.M1)

	return sigM
	
def get_fSHMR(Mh, survey):
	""" Get Mstar in terms of Mh using f_SHMR inverse relationship."""
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	Mstar = np.logspace(0, 14, 2000)
	
	Mh_vec = fSHMR_inverse(Mstar, survey)
	
	#plt.figure()
	#plt.loglog(Mh_vec, Mstar)
	#plt.xlim(10**11, 3*10**15)
	#plt.ylim(7*10**8, 6*10**11)
	#plt.savefig('./plots/infSHMR.pdf')
	#plt.close()
	
	#plt.figure()
	#plt.loglog(Mstar, Mh_vec)
	#plt.ylim(10**11, 3*10**15)
	#plt.xlim(7*10**8, 6*10**11)
	#plt.savefig('./plots/infSHMR_inv.pdf')
	#plt.close()
	
	Mh_interp = scipy.interpolate.interp1d(Mh_vec, Mstar)
	
	Mstar_ans = Mh_interp(Mh)
	
	return Mstar_ans
	
def fSHMR_inverse(Ms, survey):
	""" Get Mh in terms of Mstar """
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	m = Ms / pa.Mso
	Mh = pa.M1 * m**(pa.beta) * np.exp( m**pa.delta / (1. + m**(-pa.gamma)) - 0.5)
	
	return Mh
	
def get_Msat(f_Mh, survey):
	""" Returns parameter representing the characteristic mass of a single-satelite hosting galaxy, Zu & Mandelbaum 2015."""
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	Msat = pa.Bsat * 10**12 * (f_Mh / 10**12)**pa.beta_sat
	
	return Msat
	
def get_Mcut(f_Mh, survey):
	""" Returns the parameter representing the cutoff mass scales """
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	Mcut = pa.Bcut * 10**12 * ( f_Mh / 10**12) ** pa.beta_cut
	
	return Mcut
	
def NcenNsat(alpha_sq, Ncen, Nsat):
	""" Returns the average number of pairs of central and satelite galaxies per halo of mass M. """
	
	NcNs = alpha_sq * Ncen * Nsat
	
	return NcNs
	
def NsatNsat(alpha_sq, Nsat_1, Nsat_2):
	""" Returns the average number of pairs of satelite galaxies per halo. """
	
	NsNs = alpha_sq * Nsat_1 * Nsat_2
	
	return NsNs
		
def wgg_2halo(rp_cents_, bd, bs, savefile, survey):
	""" Returns wgg for the 2-halo term only."""
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print "We don't have support for that survey yet; exiting."
		exit()
	
	# Get the redshift window functions
	z_gg, win_gg = window(survey)
	
	# Get the NL DM power spectrum from camb (this is only for the 2-halo term)
	(k_gg, P_gg) = get_Pk(z_gg, survey)
	
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
		kz_int_gg[kpi] = scipy.integrate.simps(kpkz_gg[kpi,:] * kp_gg[kpi] / kz_gg * np.sin(kz_gg*pa.close_cut), kz_gg)
		
	# Do the integral in kperp
	kp_int_gg = np.zeros(len(rp_cents_))
	for rpi in range(0,len(rp_cents_)):
		kp_int_gg[rpi] = scipy.integrate.simps(scipy.special.j0(rp_cents_[rpi]* kp_gg) * kz_int_gg, kp_gg)
		
	wgg_2h = kp_int_gg * bs * bd / np.pi**2
	wgg_stack = np.column_stack((rp_cents_, wgg_2h))
	np.savetxt(savefile, wgg_stack)
	
	return wgg_2h

def wgg_full(rp_c, fsat, fsky, bd, bs, savefile_1h, savefile_2h, plotfile,survey):
	""" Combine 1 and 2 halo terms of wgg """
	
	# Check if savefile_1h exists and if not compute the 1halo term.
	if (os.path.isfile(savefile_1h)):
		print "Loading wgg 1halo term from file."
		(rp_cen, wgg_1h) = np.loadtxt(savefile_1h, unpack=True)	
	else:
		print "Computing wgg 1halo term."
		wgg_1h = wgg_1halo_Four(rp_c, fsat, fsky, savefile_1h, survey)
		
	# Same for savefile_2h 
	if (os.path.isfile(savefile_2h)):
		print "Loading wgg 2halo term from file."
		(rp_cen, wgg_2h) = np.loadtxt(savefile_2h, unpack=True)
	else:	
		print "Computing wgg 2halo term."
		wgg_2h = wgg_2halo(rp_c, bd, bs, savefile_2h,survey)
	
	wgg_tot = wgg_1h + wgg_2h 
	
	plt.figure()
	plt.loglog(rp_c, wgg_1h, 'go')
	plt.hold(True)
	plt.loglog(rp_c, wgg_2h, 'mo')
	plt.hold(True)
	plt.loglog(rp_c, wgg_tot, 'ko')
	plt.xlim(0.01, 200.)
	#plt.ylim(0., 300.)
	plt.ylabel('$w_{gg}$, Mpc/h, com')
	plt.xlabel('$r_p$, Mpc/h, com')
	plt.savefig('./plots/wgg_tot_mod_dndM_survey='+survey+'.pdf')
	plt.close()
	
	plt.figure()
	plt.loglog(rp_c, wgg_1h, 'go')
	plt.xlim(0.05, 30.)
	#plt.ylim(0., 300.)
	plt.ylabel('$w_{gg}$, Mpc/h, com')
	plt.xlabel('$r_p$, Mpc/h, com')
	plt.title('$w_{gg}$, 1halo')
	plt.savefig('./plots/wgg_1h_dndM_survey='+survey+'.pdf')
	plt.close()
	
	
	#plt.figure()
	#plt.loglog(rp_c, wgg_2h, 'go')
	#plt.xlim(0.05, 30.)
	#plt.ylim(1, 5000.)
	#plt.ylabel('$w_{gg}$, Mpc/h, com')
	#plt.xlabel('$r_p$, Mpc/h, com')
	#plt.title('$w_{gg}$, 2halo')
	#plt.savefig('./plots/wgg_2h_LRG.pdf')
	#plt.close()
	
	#print "wgg=", zip(rp_c, wgg_tot)
	
	#plt.figure()
	#plt.loglog(rp_c, wgg_tot, 'mo')
	#plt.xlim(0.05, 30.)
	#plt.ylim(1., 5000.)
	#plt.ylabel('$w_{gg}$, Mpc/h, com')
	#plt.xlabel('$r_p$, Mpc/h, com')
	#plt.title('$w_{gg}$, 1halo + 2halo')
	#plt.savefig(plotfile)
	#plt.close()
	
	#plt.figure()
	#plt.semilogx(rp_c, rp_c * wgg_tot, 'mo')
	#plt.xlim(0.1, 200.)
	#plt.ylim(0., 320.)
	#plt.ylabel('$r_p w_{gg}$, Mpc/h, com')
	#plt.xlabel('$r_p$, Mpc/h, com')
	#plt.savefig('./plots/Rwgg_tot_LRG.png')
	#plt.close()
	
	return wgg_tot
