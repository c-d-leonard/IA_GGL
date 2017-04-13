# This file contains functions for getting w_{l+} w_{ls} which are shared between the Blazek et al. 2012 method and the multiple shape measurements method.

import params as pa
import scipy.integrate
import matplotlib.pyplot as plt
import scipy.interpolate
import subprocess
import shutil
import numpy as np
import shared_functions_setup as setup
import os.path

# Functions shared between w_{l+} and w_{ls}

def window():
	""" Get window function for w_{l+} and w_{ls} 2-halo terms."""
	
	sigz = pa.sigz_gwin  # This is a very small value to basically set the lenses to a delta function.
	z = scipy.linspace(pa.zeff-5.*sigz, pa.zeff+5.*sigz, 50)
	dNdz_1 = 1. / np.sqrt(2. * np.pi) / sigz * np.exp(-(z-pa.zeff)**2 / (2. *sigz**2)) # Lens distribution - a very narrow Gaussian about the effective redshift.
	
	chi = setup.com(z)
	OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
	dzdchi = pa.H0 * ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )**(0.5)
	
	(z, dNdz_2) = setup.get_NofZ_unnormed(pa.alpha_fid, pa.zs_fid, pa.zeff-5.*sigz, pa.zeff+5.*sigz, 50)
		
	norm = scipy.integrate.simps(dNdz_1*dNdz_2 / chi**2 * dzdchi, z)
	
	win = dNdz_1*dNdz_2 / chi**2 * dzdchi / norm
	
	return (z, win )

def get_Pk(z_):
	""" Calls camb and returns the nonlinear halofit power spectrum for the current cosmological parameters and at the redshifts of interest. Returns a 2D array in k and z."""
	
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

def wgp_1halo(rp_c_, bd, Ai, ah, q11, q12, q13, q21, q22, q23, q31, q32, q33, savefile):
	""" Returns the 1 halo term of wg+(rp) """
	
	(z, w) = window() 
	
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
	#plt.savefig('./plots/wg+_1h_ah=1.pdf')
	#plt.close()
	
	wgp_save = np.column_stack((rp_c_, wgp1h))
	np.savetxt(savefile, wgp_save)
		
	return wgp1h

def wgp_2halo(rp_cents_, bd, Ai, savefile):
	""" Returns wgp from the nonlinear alignment model (2-halo term only). """
	
	# Get the redshift window function
	z_gp, win_gp = window()
	
	# Get the required matter power spectrum from camb 
	(k_gp, P_gp) = get_Pk(z_gp)
	
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
		kz_int_gp[kpi] = scipy.integrate.simps(kpkz_gp[kpi,:] * kp_gp[kpi]**3 / ( (kp_gp[kpi]**2 + kz_gp**2)*kz_gp) * np.sin(kz_gp*pa.close_cut), kz_gp)
			
	# Finally, do the integrals in kperpendicular
	kp_int_gp = np.zeros(len(rp_cents_))
	for rpi in range(0,len(rp_cents_)):
		kp_int_gp[rpi] = scipy.integrate.simps(scipy.special.jv(2, rp_cents_[rpi]* kp_gp) * kz_int_gp, kp_gp)
		
	wgp_NLA = kp_int_gp * Ai * bd * pa.C1rho * (pa.OmC + pa.OmB) / np.pi**2
	
	wgp_stack = np.column_stack((rp_cents_, wgp_NLA))
	np.savetxt(savefile, wgp_stack)
	
	return wgp_NLA

def wgp_full(rp_c, bd, Ai, ah, q11, q12, q13, q21, q22, q23, q31, q32, q33, savefile_1h, savefile_2h, plotfile):
	""" Combine 1 and 2 halo terms of wgg """
	
	# Check if savefile_1h exists, and if not, calculate 1 halo term.
	if (os.path.isfile(savefile_1h)):
		print "Loading wgp 1halo term from file"
		(rp_cen, wgp_1h) = np.loadtxt(savefile_1h, unpack=True)
	else:
		print "Computing wgp 1halo term"
		wgp_1h = wgp_1halo(rp_c, bd, Ai, ah, q11, q12, q13, q21, q22, q23, q31, q32, q33, savefile_1h)
		
	# Check if savefile_2h exists, and if not, calculate 2 halo term.
	if (os.path.isfile(savefile_2h)):
		print "Loading wgp 2halo term from file "
		(rp_cen, wgp_2h) = np.loadtxt(savefile_2h, unpack=True)
	else:
		print "Computing wgp 2halo term"
		wgp_2h = wgp_2halo(rp_c, bd, Ai, savefile_2h)
	
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
	
	plt.figure()
	plt.loglog(rp_c, wgp_tot, 'go')
	plt.xlim(0.05, 30.)
	plt.ylabel('$w_{gp}$, Mpc/h')
	plt.title('$w_{gp}$, 1halo+2halo')
	plt.xlabel('$r_p$, Mpc/h')
	plt.savefig(plotfile)
	plt.close()
	
	return wgp_tot

	
# Functions to get the 1halo term of w_{ls}

def vol_dens(fsky, zmin_dndz, zmax_dndz, N):
	""" Computes the volume density of galaxies given the fsky, minimum z, max z, and number of galaxies."""
	
	V = fsky * 4. / 3. * np.pi * (setup.com(pa.zmax_dndz)**3 - setup.com(pa.zmin_dndz)**3)
	ndens = N / V
	return ndens

def Rhalo(M_insol):
	""" Get the radius of a halo in COMOVING Mpc/h given its mass."""
	
	#rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun * (pa. HH0 / 100.)) # Msol h^3 / Mpc^3, for use with M in Msol.
	rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h
	rho_m = rho_crit * pa.OmM
	Rvir = ( 3. * M_insol / (4. * np.pi * rho_m * 180.))**(1./3.)
	
	return Rvir

def cvir(M_insol, z):
	""" Returns the concentration parameter of the NFW profile, c_{vir}. """

	cvi = 5. * (M_insol / 10**14)**(-0.1)
	
	return cvi

def rho_s(cvi, Rvi, M_insol):
	""" Returns rho_s, the NFW parameter representing the density at the `scale radius', Rvir / cvir. Units: Mvir units * ( 1 / (Rvir units)**3), usualy Msol * h^3 / Mpc^3 with comoving distances. Sometimes also Msol h^2 / Mpc^3 (when Mvir is in Msol / h). """
	
	rhos = M_insol / (4. * np.pi) * ( cvi / Rvi)**3 * (np.log(1. + cvi) - (cvi / (1. + cvi)))**(-1)
	
	return rhos

def rho_NFW(r_, M_insol, z):
	""" Returns the density for an NFW profile in real space at distance r from the center. Units = units of rhos. (Usually Msol * h^2 / Mpc^3 in comoving distances). r_ MUST be in the same units as Rv; usually Mpc / h."""

	Rv = Rhalo(M_insol)
	cv = cvir(M_insol, z)
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

def f_near_1halo(zeff, Rvir):
	""" Computes the fraction of source galaxies within a reasonable 1-halo separation of a redshift."""
	
	z_of_com = setup.z_interpof_com()
	z_min = z_of_com(setup.com(pa.zeff) - Rvir) 
	z_max = z_of_com(setup.com(pa.zeff) + Rvir)
	
	(z, dNdz_unnorm) = setup.get_NofZ_unnormed(pa.alpha_fid, pa.zs_fid, z_min, z_max, 10000)
	
	(z_norm, dNdz_getnorm) = setup.get_NofZ_unnormed(pa.alpha_fid, pa.zs_fid, pa.zsmin, pa.zsmax, 10000)
	
	norm = scipy.integrate.simps(dNdz_getnorm, z_norm)
	
	f_near = scipy.integrate.simps(dNdz_unnorm, z) / norm
	
	return f_near

def wgg_1halo_Four(rp_cents_, fsat, fsky, savefile):
	""" Gets the 1halo term of wgg via Fourier space, to account for central-satelite pairs and satelite-satelite pairs. """
	
	Rvir = Rhalo(pa.Mvir)
	
	kvec_FT = np.logspace(-7, 5, 1000000)
	rvec_NFW = np.logspace(-8, np.log10(Rvir), 5000000)
	rvec_xi = np.logspace(-4, 4, 1000)
	Pivec = np.logspace(-7, np.log10(1000./np.sqrt(2.01)), 1000)
	rpvec = np.logspace(-7, np.log10(1000./np.sqrt(2.01)), 1000)
	
	#print "Before getting P_{gg}(k)"
	Pk = get_Pkgg_1halo(rvec_NFW, kvec_FT, fsat, fsky) # Gets the 1halo galaxy power spectrum including c-s and s-s terms. Pass rvec because need to get rho_NFW in here.
	#(k_dummy, Pk) = np.loadtxt('./txtfiles/Pkgg_1h.txt', unpack=True)
	#print "After getting P_{gg}(k), before getting xi_{gg}(r)."
	
	xi_gg_1h = get_xi_1h(rvec_xi, kvec_FT, Pk)
	
	#plt.figure()
	#plt.loglog(rvec_xi, xi_gg_1h, 'go')
	#plt.ylabel('$\\xi(r)$')
	#plt.xlabel('$r$, Mpc/h com')
	#plt.xlim(10**(-7), 50)
	#plt.savefig('./plots/xigg_1h_FFT_LRG.png')
	#plt.close()
	
	# Set xi_gg_1h to zero above Rvir - we have made this assumption; anything else is Fourier transform noise.
	for ri in range(0, len(rvec_xi)):
		if (rvec_xi[ri]>Rvir):
			xi_gg_1h[ri] = 0.0
			
	#plt.figure()
	#plt.loglog(rvec_xi, xi_gg_1h, 'go')
	#plt.ylabel('$\\xi(r)$')
	#plt.xlabel('$r$, Mpc/h com')
	#plt.xlim(0.5, 0.6)
	#plt.savefig('./plots/xigg_1h_FFT_LRG.png')
	#plt.close()
	
	#xi_gg_1h = get_xi_1h(rvec_xi, kvec_FT, Pk) # Function that gets the 1halo galaxy-galaxy correlation function term.
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
		
	#plt.figure()
	#plt.loglog(rp_cents_, wgg_1h, 'go')
	#plt.xlim(0.05, 20.)
	#plt.ylim(10**(-3), 10**(4))
	#plt.xlabel('$r_p$, Mpc/h com')
	#plt.ylabel('$w_{gg}$, Mpc/h com')
	#plt.savefig('./plots/wgg_1h_LRG.png')
	#plt.close()
	
	wgg_save = np.column_stack((rp_cents_, wgg_1h))
	np.savetxt(savefile, wgg_save)
	
	return wgg_1h
	
def get_xi_1h(r, kvec, Pofk):
	""" Returns the 1 halo galaxy correlation function including cen-sat and sat-sat terms, from the power spectrum via Fourier transform."""
	
	xi = np.zeros(len(r))
	for ri in range(0,len(r)):
		xi[ri] = scipy.integrate.simps(Pofk * kvec**2 * np.sin(kvec*r[ri]) / (kvec * r[ri]) / 2. / np.pi**2 , kvec)
	
	return xi
	
def get_Pkgg_1halo(rvec_nfw, kvec_ft, fsat, fsky):
	""" Returns the 1halo galaxy power spectrum with c-s and s-s terms."""
	
	# Get ingredients we need here:
	y = gety(rvec_nfw, pa.Mvir, pa.zeff, kvec_ft) # Mass-averaged Fourier transform of the density profile
	
	#(kvec_nothing, y) = np.loadtxt('./txtfiles/y.txt', unpack=True)

	alpha = get_alpha(pa.Mvir) # The number which accounts for deviation from Poisson statistics
	Ncenavg = 1. # We assume that every halo has a central galaxy, so the mean number of central galaxies / halo is 1.
	
	fcen = 1. - fsat # fraction of central galaxies = 1 - fraction of satelite galaxies
	Nsatavg = fsat / fcen # Mean number of satelite galaxies per halo

	NcNs = NcenNsat(alpha, Ncenavg, Nsatavg) # The average number of central-satelite pairs in a halo of mass M
	NsNs = NsatNsat(alpha, Nsatavg) # The average number of satelite-satelite pairs in a halo of mass M

	# Get the volume density of the shape galaxy sample:
	ns = vol_dens(fsky, pa.zmin_dndz, pa.zmax_dndz, pa.N_shapes)
	# Get the fraction of source galaxies that are near enough the lens redshift that they come into the 1halo term.
	Rv = Rhalo(pa.Mvir)
	fns = f_near_1halo(pa.zeff, Rv)

	#Pkgg = (1. - fsat) / ng * (NcNs * y + NsNs * y **2)
	Pkgg = (1. - fsat) / ns * ( ( fns * pa.N_shapes ) / ( ( 1. - fsat ) * pa.N_LRG ) * y + ( fsat / ( 1. - fsat ) ) * fns * pa.N_shapes / ( ( 1. - fsat ) * pa.N_LRG ) * y**2)
	
	#Pkgg_save = np.column_stack((kvec_ft, Pkgg))
	#np.savetxt('./txtfiles/Pkgg_1h.txt', Pkgg_save)
	
	#plt.figure()
	#plt.loglog(kvec_ft, 4* np.pi * kvec_ft**3 * Pkgg / (2* np.pi)**3, 'm')
	#plt.ylim(0.001, 100000)
	#plt.xlim(0.01, 100)
	#plt.ylabel('$4\pi k^3 P_{gg}^{1h}(k)$, $(Mpc/h)^3$, com')
	#plt.xlabel('$k$, h/Mpc, com')
	#plt.savefig('./plots/Pkgg_1halo_LRG.png')
	#plt.close()
	return Pkgg
	
def gety(rvec, M, z, kvec):
	""" Fourier transforms the density profile to get the power spectrum. """
	
	# Get the nfw density profile at the correct mass and redshift and at a variety of r
	rho = rho_NFW(rvec, M, z)  # Units Msol h^2 / Mpc^3, comoving.
	
	# Use a downsampled kvec to speed computation
	kvec_gety = np.logspace(np.log10(kvec[0]), np.log10(kvec[-1]), 100) 
	
	u_ = np.zeros(len(kvec_gety))
	for ki in range(0,len(kvec_gety)):
		u_[ki] = 4. * np.pi / M * scipy.integrate.simps( rvec * np.sin(kvec_gety[ki]*rvec)/ kvec_gety[ki] * rho, rvec) # unitless / dimensionless.
	
	#plt.figure()
	#plt.semilogx(kvec_gety, u_, 'm+')
	#plt.ylim(0.0, 1.05)
	#plt.xlim(0.0005,2000.)
	#plt.ylabel('u')
	#plt.xlabel('$k$, h/Mpc, com')
	#plt.savefig('./plots/y_notlog_LRG.png')
	#plt.close()
		
	# Interpolate in k and use the higher-sampled k to output
	u_interp = scipy.interpolate.interp1d(kvec_gety, u_)
	u_morepoints = u_interp(kvec) 
	
	#save_y = np.column_stack((kvec, u_morepoints))
	#np.savetxt('./txtfiles/y.txt', save_y)
	
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
		
def wgg_2halo(rp_cents_, bd, bs, savefile):
	""" Returns wgg for the 2-halo term only."""
	
	# Get the redshift window functions
	z_gg, win_gg = window()
	
	# Get the NL DM power spectrum from camb (this is only for the 2-halo term)
	(k_gg, P_gg) = get_Pk(z_gg)
	
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

def wgg_full(rp_c, fsat, fsky, bd, bs, savefile_1h, savefile_2h, plotfile):
	""" Combine 1 and 2 halo terms of wgg """
	
	# Check if savefile_1h exists and if not compute the 1halo term.
	if (os.path.isfile(savefile_1h)):
		print "Loading wgg 1halo term from file."
		(rp_cen, wgg_1h) = np.loadtxt(savefile_1h, unpack=True)	
	else:
		print "Computing wgg 1halo term."
		wgg_1h = wgg_1halo_Four(rp_c, fsat, fsky, savefile_1h)
		
	# Same for savefile_2h 
	if (os.path.isfile(savefile_2h)):
		print "Loading wgg 2halo term from file."
		(rp_cen, wgg_2h) = np.loadtxt(savefile_2h, unpack=True)
	else:	
		print "Computing wgg 2halo term."
		wgg_2h = wgg_2halo(rp_c, bd, bs, savefile_2h)
	
	wgg_tot = wgg_1h + wgg_2h 
	
	#plt.figure()
	#plt.loglog(rp_c, wgg_tot, 'go')
	#plt.xlim(0.01, 200.)
	#plt.ylim(0., 300.)
	#plt.ylabel('$w_{gg}$, Mpc/h, com')
	#plt.xlabel('$r_p$, Mpc/h, com')
	#plt.savefig('./plots/wgg_tot_mod.png')
	#plt.close()
	
	#plt.figure()
	#plt.loglog(rp_c, wgg_1h, 'go')
	#plt.xlim(0.05, 30.)
	#plt.ylim(0., 300.)
	#plt.ylabel('$w_{gg}$, Mpc/h, com')
	#plt.xlabel('$r_p$, Mpc/h, com')
	#plt.title('$w_{gg}$, 1halo')
	#plt.savefig('./plots/wgg_1h_LRG.pdf')
	#plt.close()
	
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
	
	plt.figure()
	plt.loglog(rp_c, wgg_tot, 'mo')
	plt.xlim(0.05, 30.)
	plt.ylim(1., 5000.)
	plt.ylabel('$w_{gg}$, Mpc/h, com')
	plt.xlabel('$r_p$, Mpc/h, com')
	plt.title('$w_{gg}$, 1halo + 2halo')
	plt.savefig(plotfile)
	plt.close()
	
	#plt.figure()
	#plt.semilogx(rp_c, rp_c * wgg_tot, 'mo')
	#plt.xlim(0.1, 200.)
	#plt.ylim(0., 320.)
	#plt.ylabel('$r_p w_{gg}$, Mpc/h, com')
	#plt.xlabel('$r_p$, Mpc/h, com')
	#plt.savefig('./plots/Rwgg_tot_LRG.png')
	#plt.close()
	
	return wgg_tot
