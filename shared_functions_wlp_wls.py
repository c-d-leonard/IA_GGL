# This file contains functions for getting w_{l+} w_{ls} which are shared between the Blazek et al. 2012 method and the multiple-shape-measurements method.

import scipy.integrate
import matplotlib.pyplot as plt
import scipy.interpolate
import subprocess
import shutil
import numpy as np
import shared_functions_setup as setup
import os.path
import pyccl as ccl
from halotools.empirical_models import PrebuiltHodModelFactory

# Functions shared between w_{l+} and w_{ls}

def get_ah(survey):
	""" Get the amplitude of the 1-halo part of w_{l+} """
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print("We don't have support for that survey yet; exiting.")
		exit()
		
	# Don't evaluate at any redshifts higher than the highest value for which we have kcorr and ecorr corrections
	# These high (>3) redshifts shouldn't matter anyway.
	(z_k, kcorr, x,x,x) = np.loadtxt('./txtfiles/kcorr.dat', unpack=True)
	(z_e, ecorr, x,x,x) = np.loadtxt('./txtfiles/ecorr.dat', unpack=True)
	zmaxke = min(max(z_k), max(z_e))	
	if (zmaxke<pa.zLmax):
		z = np.linspace(pa.zLmin, zmaxke, 1000)
	else:
		z = np.linspace(pa.zLmin, pa.zLmax, 1000)	

	# Get the luminosity function
	(L, phi_normed, phi) = setup.get_phi(z, pa.lumparams_red, survey)
	# Pivot luminosity:
	Lp = 1.
	
	# Get ah as a function of lens redshift.
	ah_ofzl = np.zeros(len(z))
	for zi in range(len(z)):
		ah_ofzl[zi] = scipy.integrate.simps(np.asarray(phi_normed[zi]) * 0.081 * (np.asarray(L[zi]) / Lp)**(2.1), np.asarray(L[zi]))
	
	# Integrate over lens redshift	
	dNdzl = setup.get_dNdzL(z, survey)
	
	ah = scipy.integrate.simps(ah_ofzl * dNdzl, z)
	
	return ah
	
def get_Ai(survey):
    """ Get the amplitude of the 2-halo part of w_{l+} """
	
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()
        
    (z_k, kcorr, x,x,x) = np.loadtxt('./txtfiles/kcorr.dat', unpack=True)
    (z_e, ecorr, x,x,x) = np.loadtxt('./txtfiles/ecorr.dat', unpack=True)
    zmaxke = min(max(z_k), max(z_e))
        
    if survey=='DESY1':
        z_l, dNdz_l = np.loadtxt('./txtfiles/'+pa.dNdzL_file, unpack=True)
        # Don't evaluate at any redshifts higher than the highest value for which we have kcorr and ecorr corrections
        # These high (>3) redshifts shouldn't matter anyway.	
        if (zmaxke<z_l[-1]):
            ind = next(j[0] for j in enumerate(z_l) if j[1]>zmaxke)
            z = z_l[0:ind]
            dNdzl = dNdz_l[0:ind]
        else:
            z = z_l
            dNdzl = dNdz_l
        
    else:
        if (zmaxke<pa.zLmax):
            z = np.linspace(pa.zLmin, zmaxke, 1000)
        else:
            z = np.linspace(pa.zLmin, pa.zLmax, 1000)
    
        dNdzl = setup.get_dNdzL(z, survey)

    # Get the luminosity function
    (L, phi_normed, phi) = setup.get_phi(z, pa.lumparams_red, survey)
    # Pivot luminosity:
    Lp = 1.
	
    # Get ah as a function of lens redshift.
    Ai_ofzl = np.zeros(len(z))
    for zi in range(len(z)):
        Ai_ofzl[zi] = scipy.integrate.simps(np.asarray(phi_normed[zi]) * pa.A_IA_amp * (np.asarray(L[zi]) / Lp)**(pa.beta_IA), np.asarray(L[zi]))
	
    # Integrate over lens redshift	
	
    Ai = scipy.integrate.simps(Ai_ofzl * dNdzl, z) / scipy.integrate.simps(dNdzl,z)
	
    return Ai

def window(survey, sample='null'):
    """ Get window function for w_{l+} and w_{ls} 2-halo terms. In both cases, this is the window functions for LENSES x SOURCES. """
    """ Old note: Note I am just going to use the standard cosmological parameters here because it's a pain and it shouldn't matter too much. """
	
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()
        
    if survey == 'DESY1':
        """
        # Load dNdzs from file for the appropriate source sample
        if(sample == 'B'):
            z_s = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_centres.dat')
            dNdz_2 = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_weighted')
        elif(sample=='A'):
            z_s = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin0_zmc_centres.dat')
            dNdz_2 = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin0_zmc_weighted')
            
        #print("Using perturbed source redshift distribution in window()")
        #z_s, dNdz_2 = setup.dNdz_perturbed(sample, pa.sigma, pa.del_z)"""
        dndz = np.load('./im3_full_n_values.npz')
        z_edges = dndz['bins']
        dNdz_2 = dndz['weighted_counts']
        z_s = np.zeros(len(z_edges)-1)
        for i in range(0,len(z_edges)-1):
            z_s[i] = (z_edges[i+1]-z_edges[i])/2. + z_edges[i]
        
    else:    	
        (z, dNdz_s_extend) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zLmin, pa.zLmax, 100, survey)
        
        
    if survey == 'DESY1':
        # Load dNdzL from file
        z_l, dNdz_1 = np.loadtxt('./txtfiles/'+pa.dNdzL_file, unpack=True)
        
        # this will not natively be over the same range as the source dNdz above - need to fix this
        # Get the segment of the source redshift before and after the lens redshift range:
        ind1 = next(j[0] for j in enumerate(z_s) if j[1]>min(z_l))
        ind2 = next(j[0] for j in enumerate(z_s) if j[1]>max(z_l))
        z_low_source = np.asarray(z_s[0:ind1])
        z_high_source = np.asarray(z_s[ind2:])
        
        
        z=np.append(np.append(z_low_source, z_l), z_high_source)
        
        # Okay, now we are going to set up the lens dNdz so it's zero everywhere outside its original range
        dNdz_l_extend = np.zeros(len(z))
        for zi in range(0, len(z)):
            if z[zi]<min(z_l):
                dNdz_l_extend[zi] = 0.
            elif z[zi]>max(z_l):
                dNdz_l_extend[zi]= 0.
            else:
                dNdz_l_extend[zi] = dNdz_1[zi-ind1]

        #plt.figure()
        #plt.plot(z, dNdz_l_extend)
        #plt.savefig('./text_dNdz_l_extend.png')
        #plt.close()
        
        # And now we are going to interpolate the source dNdz in this new z vector:
        dNdzs_interp = scipy.interpolate.interp1d(z_s, dNdz_2)
        dNdz_s_extend = dNdzs_interp(z)
        
        #plt.figure()
        #plt.plot(z, dNdz_s_extend)
        #plt.savefig('./text_dNdz_s_extend.png')
        #plt.close()
         
        
    else:
        z = np.linspace(pa.zLmin, pa.zLmax, 100)
        dNdz_l_extend = setup.get_dNdzL(z, survey)
	
    #chi = setup.com(z, survey, pa.cos_par_std)
    # 'true' cosmological parameters because this is a fiducial signal.
    cosmo_t = ccl.Cosmology(Omega_c = pa.OmC_t, Omega_b = pa.OmB, h = (pa.HH0_t/100.), sigma8 = pa.sigma8, n_s=pa.n_s)
    chi = ccl.comoving_radial_distance(cosmo_t, 1./(1.+z)) * (pa.HH0_t / 100.) # CCL returns in Mpc but we want Mpc/h
    OmL = 1. - pa.OmC_t - pa.OmB - pa.OmR_t - pa.OmN_t
    dzdchi = pa.H0 * ( ( pa.OmC_t + pa.OmB )*(1+z)**3 + OmL + (pa.OmR_t+pa.OmN_t) * (1+z)**4 )**(0.5) 
        
    #if max(z_s)!=max(z) or min(z_s)!=min(z):
    #    print("in window(), need redshifts loaded for sources and lenses to span same range")
    #    print("z_s=", z_s)
    #    print("z=", z)
    
    #norm = scipy.integrate.simps(dNdz_1*dNdz_2 / chi**2 * dzdchi, z)
    norm =  scipy.integrate.simps(dNdz_l_extend*dNdz_s_extend / chi**2 * dzdchi, z)
	
    win = dNdz_l_extend*dNdz_s_extend / chi**2 * dzdchi / norm
	
    return (z, win )

# Functions to get the 1halo term of w_{l+}

def get_pi(q1, q2, q3, z_):
	""" Returns the pi functions requires for the 1 halo term in wg+, at z_ """
	
	pi = q1 * np.exp(q2 * z_**q3)
	
	return pi

def get_P1haloIA(z, k, q11, q12, q13, q21, q22, q23, q31, q32, q33, survey):
	""" Returns the power spectrum required for the wg+ 1 halo term, at z and k_perpendicular ( = k) """
	
	p1 = get_pi(q11, q12, q13, z)
	p2 = get_pi(q21, q22, q23, z)
	p3 = get_pi(q31, q32, q33, z)
	
	# Get amplitude parameter (this is a function of limiting luminosity
	ah = get_ah(survey)
	print("ah=", ah)
	
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
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()
	
    def int_(z):
        OmL = 1. - pa.OmC_t - pa.OmB - pa.OmR_t - pa.OmN_t
        return (1.+z) / ( (pa.OmC_t+pa.OmB)*(1+z)**3 + OmL + (pa.OmR_t+pa.OmN_t) * (1+z)**4 )**(1.5)
	
    norm = scipy.integrate.quad(int_, 0, 1000.)[0]
	
    ans = np.zeros(len(z_))
    for zi in range(0,len(z_)):
        ans[zi] = scipy.integrate.quad(int_, z_[zi], 1000)[0]
	
    D = ans / norm
	
    return D

def wgp_1halo(rp_c_, q11, q12, q13, q21, q22, q23, q31, q32, q33, savefile, survey):
	""" Returns the 1 halo term of wg+(rp) """
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print("We don't have support for that survey yet; exiting.")
		exit()
	
	(z, w) = window(survey) 
	
	# Set up a k vector to integrate over:
	k = np.logspace(-5., 7., 100000)
	
	# Get the `power spectrum' term
	P1h = get_P1haloIA(z, k, q11, q12, q13, q21, q22, q23, q31, q32, q33, survey)
	
	# First do the integral over z:
	zint = np.zeros(len(k))
	for ki in range(0,len(k)):
		zint[ki] = scipy.integrate.simps(P1h[ki, :] * w , z)
		
	# Now do the integral in k
	ans = np.zeros(len(rp_c_))
	for rpi in range(0,len(rp_c_)):
		integrand = k * zint * scipy.special.j0(rp_c_[rpi] * k)
		ans[rpi] = scipy.integrate.simps(integrand, k)
		
	# Set this to zero above about 2 * virial radius (I've picked this value somewhat aposteriori, should do better). This is to not include 1halo contributions well outside the halo.
	Rvir = Rhalo(10**16, survey)

	for ri in range(0,len(rp_c_)):
		if (rp_c_[ri]> 2.*Rvir):
			ans[ri] = 0.
	
	wgp1h = ans / (2. * np.pi)
	
	wgp_save = np.column_stack((rp_c_, wgp1h))
	np.savetxt(savefile, wgp_save)
		
	return wgp1h

def wgp_2halo(rp_cents_, bd, savefile, survey, sample):
    """ Returns wgp from the nonlinear alignment model (2-halo term only). """
	
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()
	
    print("Getting window")	
    # Get the redshift window function
    z_gp, win_gp = window(survey, sample) # this is working april 15
    print("Got window")
	
    # Get the amplitude Ai (this depends on limiting luminosity)
    print("Getting Ai")
    Ai = get_Ai(survey) # this is working april 15
    print("Got Ai")
    print("Ai=", Ai)

    # Get the required matter power spectrum from CCL
    #p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), sigma8=pa.sigma8, n_s=pa.n_s_cosmo)
    #cosmo = ccl.Cosmology(p)
    # 'true' cosmological parameters because this is a fiducial signal.
    cosmo_t = ccl.Cosmology(Omega_c = pa.OmC_t, Omega_b = pa.OmB, h = (pa.HH0_t/100.), sigma8 = pa.sigma8, n_s=pa.n_s)
    #chi = ccl.comoving_radial_distance(cosmo_t, 1./(1.+z)) * (pa.HH0_t / 100.) # CCL returns in Mpc but we want Mpc/h

    print("Getting power spectra")	
    h = (pa.HH0_t/100.)
    k_gp = np.logspace(-5., 7., 100000)
    P_gp = np.zeros((len(z_gp), len(k_gp)))
    for zi in range(0,len(z_gp)):
        P_gp[zi, :] = h**3 * ccl.nonlin_matter_power(cosmo_t, k_gp * h , 1./(1.+z_gp[zi])) # CCL takes units without little h's, but we use little h units.
    print("Got power spectra")
	
    print("Getting growth")	
    # Get the growth factor
    D_gp = growth(z_gp, survey)
    print("Got growth")
	
    print("Getting redshift integrals")	
    # First do the integral over z. Don't yet interpolate in k.
    zint_gp = np.zeros(len(k_gp))
    for ki in range(0,len(k_gp)):
        zint_gp[ki] = scipy.integrate.simps(win_gp * P_gp[:, ki] / D_gp, z_gp)
    print("Got redshift integrals")
    
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

    print("Getting integral in kz")			
    # g+: integral in kz	
    kz_int_gp = np.zeros(len(kp_gp))
    for kpi in range(0,len(kp_gp)):
        kz_int_gp[kpi] = scipy.integrate.simps(kpkz_gp[kpi,:] * kp_gp[kpi]**3 / ( (kp_gp[kpi]**2 + kz_gp**2)*kz_gp) * np.sin(kz_gp*pa.close_cut), kz_gp)
    print("Got integral in kz")

    print("Getting integral in kperp")			
    # Finally, do the integrals in kperpendicular
    kp_int_gp = np.zeros(len(rp_cents_))
    for rpi in range(0,len(rp_cents_)):
        kp_int_gp[rpi] = scipy.integrate.simps(scipy.special.jv(2, rp_cents_[rpi]* kp_gp) * kz_int_gp, kp_gp)
    print("Got integral in kperp")
		
    wgp_NLA = kp_int_gp * Ai * bd * pa.C1rho * (pa.OmC_t + pa.OmB) / np.pi**2
	
    wgp_stack = np.column_stack((rp_cents_, wgp_NLA))
    np.savetxt(savefile, wgp_stack)
	
    return wgp_NLA

def wgp_full(rp_c, bd, q11, q12, q13, q21, q22, q23, q31, q32, q33, savefile_1h, savefile_2h, survey):
	""" Combine 1 and 2 halo terms of wgg """
	
	# Check if savefile_1h exists, and if not, calculate 1 halo term.
	if (os.path.isfile(savefile_1h)):
		print("Loading wgp 1halo term from file")
		(rp_cen, wgp_1h) = np.loadtxt(savefile_1h, unpack=True)
	else:
		print("Computing wgp 1halo term")
		wgp_1h = wgp_1halo(rp_c, q11, q12, q13, q21, q22, q23, q31, q32, q33, savefile_1h, survey)
		
	# Check if savefile_2h exists, and if not, calculate 2 halo term.
	if (os.path.isfile(savefile_2h)):
		print("Loading wgp 2halo term from file ")
		(rp_cen, wgp_2h) = np.loadtxt(savefile_2h, unpack=True)
	else:
		print("Computing wgp 2halo term")
		wgp_2h = wgp_2halo(rp_c, bd, savefile_2h, survey)
	
	wgp_tot = wgp_1h + wgp_2h 
	
	return wgp_tot

	
# Functions to get the 1halo term of w_{ls}

def Rhalo(M_insol, survey):
    """ Get the radius of a halo in COMOVING Mpc/h given its mass."""
    """ Note I'm fixing to source cosmological parameters here."""
	
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()
	
    #rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun * (pa. HH0 / 100.)) # Msol h^3 / Mpc^3, for use with M in Msol.
    rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h
    rho_m = rho_crit * (pa.OmC_s + pa.OmB_s)
    Rvir = ( 3. * M_insol / (4. * np.pi * rho_m * 200.))**(1./3.) # We use the 200 * rho_M overdensity definition. 
	
    return Rvir

def cvir_ls(M_insol):
	""" Returns the concentration parameter of the NFW profile, c_{vir}. """

	cvi = 5. * (M_insol / 10**14)**(-0.1) * 0.86 # 0.86 is the value fit from Zu & Mandelbaum 2015 which governs the shift in amplitude between dm concentration and satellite concentration.
	
	return cvi
	
def cvir_ldm(M_insol):
	""" Returns the concentration parameter of the NFW profile, c_{vir}. """

	cvi = 5. * (M_insol / 10**14)**(-0.1)
	
	return cvi

def rho_s(cvi, Rvi, M_insol):
	""" Returns rho_s, the NFW parameter representing the density at the `scale radius', Rvir / cvir. Units: Mvir units * ( 1 / (Rvir units)**3), usualy Msol * h^3 / Mpc^3 with comoving distances. Sometimes also Msol h^2 / Mpc^3 (when Mvir is in Msol / h). """
	
	rhos = M_insol / (4. * np.pi) * ( cvi / Rvi)**3 * (np.log(1. + cvi) - (cvi / (1. + cvi)))**(-1)
	
	return rhos

def rho_NFW_ls(r_, M_insol, survey):
	""" Returns the density for an NFW profile in real space at distance r from the center. Units = units of rhos. (Usually Msol * h^2 / Mpc^3 in comoving distances). r_ MUST be in the same units as Rv; usually Mpc / h."""

	Rv = Rhalo(M_insol, survey)
	cv = cvir_ls(M_insol)
	rhos = rho_s(cv, Rv, M_insol)
	
	rho_nfw = rhos  / ( (cv * r_ / Rv) * (1. + cv * r_ / Rv)**2) 
	
	return rho_nfw

def rho_NFW_ldm(r_, M_insol, survey):
	""" Returns the density for an NFW profile in real space at distance r from the center. Units = units of rhos. (Usually Msol * h^2 / Mpc^3 in comoving distances). r_ MUST be in the same units as Rv; usually Mpc / h."""

	Rv = Rhalo(M_insol, survey)
	cv = cvir_ldm(M_insol)
	rhos = rho_s(cv, Rv, M_insol)
	
	rho_nfw = rhos  / ( (cv * r_ / Rv) * (1. + cv * r_ / Rv)**2) 
	
	return rho_nfw

def wgg_1halo_Four(rp_cents_, fsky, savefile, endfile, survey):
	""" Gets the 1halo term of wgg via Fourier space, to account for central-satelite pairs and satelite-satelite pairs. """
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print("We don't have support for that survey yet; exiting.")
		exit()
	
	logkmin = -6; kpts =40000; logkmax = 5; Mmax = 16;
	# Compute P_{gg}^{1h}(k)
	kvec_FT = np.logspace(logkmin, logkmax, kpts)
	
	# This function loads the xi_{gg}1h function computed from FFTlog externally.
	(rvec_xi, xi_gg_1h) = get_xi_1h(endfile, survey)
	
	# Get the max R associated to our max M
	Rmax = Rhalo(10**Mmax, survey)
	# Set xi_gg_1h to zero above Rmax Mpc/h.
	for ri in range(0, len(rvec_xi)):
		if (rvec_xi[ri]>Rmax):
			xi_gg_1h[ri] = 0.0
	
	xi_interp = scipy.interpolate.interp1d(rvec_xi, xi_gg_1h)
	
	# Get xi_{gg}1h as a function of rp and Pi
	Pivec = np.logspace(-7, np.log10(1000./np.sqrt(2.01)), 1000)
	
	xi_2D = np.zeros((len(rp_cents_), len(Pivec)))
	for ri in range(0, len(rp_cents_)):
		for pi in range(0, len(Pivec)):
			xi_2D[ri, pi] = xi_interp(np.sqrt(rp_cents_[ri]**2 + Pivec[pi]**2)) 
	
	wgg_1h = np.zeros(len(rp_cents_))
	for ri in range(0,len(rp_cents_)):
		wgg_1h[ri] = 2.* scipy.integrate.simps(xi_2D[ri, :], Pivec)
	
	wgg_save = np.column_stack((rp_cents_, wgg_1h))
	np.savetxt(savefile, wgg_save)
	
	return wgg_1h
	
def get_xi_1h(endfile, survey):
	""" Returns the 1 halo galaxy correlation function including cen-sat and sat-sat terms, from the power spectrum via Fourier transform."""
	
	(r, xi) = np.loadtxt('./txtfiles/xi_1h_terms/xigg_1h_'+survey+'_'+endfile+'.txt', unpack=True)
	
	return (r, xi)

def get_Pkgg_2h_multiz(k, endfile, survey):
	""" Get the 2-halo term for the lens x source power spectrum """
	
	if (survey=='SDSS'):
		import params as pa
		z = np.asarray([0.16, 0.16202020202020204, 0.16404040404040404, 0.16606060606060608, 0.16808080808080808, 0.17010101010101011, 0.17212121212121212, 0.17414141414141415, 0.17616161616161616, 0.17818181818181819, 0.1802020202020202, 0.18222222222222223, 0.18424242424242424, 0.18626262626262627, 0.18828282828282827, 0.19030303030303031, 0.19232323232323234, 0.19434343434343435, 0.19636363636363635, 0.19838383838383838, 0.20040404040404042, 0.20242424242424242, 0.20444444444444446, 0.20646464646464646, 0.2084848484848485, 0.2105050505050505, 0.21252525252525253, 0.21454545454545454, 0.21656565656565657, 0.21858585858585861, 0.22060606060606061, 0.22262626262626262, 0.22464646464646465, 0.22666666666666668, 0.22868686868686869, 0.23070707070707069, 0.23272727272727273, 0.23474747474747476, 0.23676767676767677, 0.23878787878787877, 0.2408080808080808, 0.24282828282828284, 0.24484848484848487, 0.24686868686868688, 0.24888888888888888, 0.25090909090909091, 0.25292929292929295, 0.25494949494949493, 0.25696969696969696, 0.25898989898989899, 0.26101010101010103, 0.26303030303030306, 0.26505050505050504, 0.26707070707070707, 0.2690909090909091, 0.27111111111111108, 0.27313131313131311, 0.27515151515151515, 0.27717171717171718, 0.27919191919191921, 0.28121212121212125, 0.28323232323232322, 0.28525252525252526, 0.28727272727272724, 0.28929292929292927, 0.2913131313131313, 0.29333333333333333, 0.29535353535353537, 0.2973737373737374, 0.29939393939393943, 0.30141414141414141, 0.30343434343434345, 0.30545454545454542, 0.30747474747474746, 0.30949494949494949, 0.31151515151515152, 0.31353535353535356, 0.31555555555555559, 0.31757575757575757, 0.3195959595959596, 0.32161616161616158, 0.32363636363636361, 0.32565656565656564, 0.32767676767676768, 0.32969696969696971, 0.33171717171717174, 0.33373737373737378, 0.33575757575757575, 0.33777777777777779, 0.33979797979797977, 0.3418181818181818, 0.34383838383838383, 0.34585858585858587, 0.3478787878787879, 0.34989898989898993, 0.35191919191919191, 0.35393939393939394, 0.35595959595959598, 0.35797979797979795, 0.35999999999999999])
	elif (survey=='LSST_DESI'):
		import params_LSST_DESI as pa
		z = np.asarray([0.025000000000000001, 0.036616161616161616, 0.048232323232323238, 0.059848484848484852, 0.071464646464646481, 0.083080808080808088, 0.094696969696969696, 0.10631313131313133, 0.11792929292929294, 0.12954545454545457, 0.14116161616161618, 0.15277777777777779, 0.1643939393939394, 0.17601010101010103, 0.18762626262626264, 0.19924242424242428, 0.21085858585858588, 0.22247474747474749, 0.23409090909090913, 0.24570707070707073, 0.2573232323232324, 0.26893939393939398, 0.28055555555555561, 0.29217171717171725, 0.30378787878787883, 0.31540404040404046, 0.3270202020202021, 0.33863636363636374, 0.35025252525252532, 0.36186868686868695, 0.37348484848484859, 0.38510101010101017, 0.3967171717171718, 0.40833333333333344, 0.41994949494949502, 0.43156565656565665, 0.44318181818181829, 0.45479797979797987, 0.4664141414141415, 0.47803030303030314, 0.48964646464646477, 0.5012626262626263, 0.51287878787878793, 0.52449494949494957, 0.5361111111111112, 0.54772727272727284, 0.55934343434343448, 0.57095959595959611, 0.58257575757575764, 0.59419191919191927, 0.60580808080808091, 0.61742424242424254, 0.62904040404040418, 0.64065656565656581, 0.65227272727272745, 0.66388888888888897, 0.67550505050505061, 0.68712121212121224, 0.69873737373737388, 0.71035353535353551, 0.72196969696969715, 0.73358585858585867, 0.74520202020202031, 0.75681818181818195, 0.76843434343434358, 0.78005050505050522, 0.79166666666666685, 0.80328282828282849, 0.81489898989899001, 0.82651515151515165, 0.83813131313131328, 0.84974747474747492, 0.86136363636363655, 0.87297979797979819, 0.88459595959595971, 0.89621212121212135, 0.90782828282828298, 0.91944444444444462, 0.93106060606060626, 0.94267676767676789, 0.95429292929292953, 0.96590909090909105, 0.97752525252525269, 0.98914141414141432, 1.000757575757576, 1.0123737373737376, 1.0239898989898992, 1.0356060606060606, 1.0472222222222223, 1.0588383838383839, 1.0704545454545455, 1.0820707070707072, 1.0936868686868688, 1.1053030303030305, 1.1169191919191921, 1.1285353535353537, 1.1401515151515151, 1.1517676767676768, 1.1633838383838384, 1.175])
	elif (survey=='DESY1'):
		import params_DESY1_testpz as pa
		#z = np.loadtxt('/home/danielle/Research/IA_measurement_GGL/IA_GGL/txtfiles/DESY1_quantities_fromSara/lenz_subbin_cent.dat')
		z = np.loadtxt('/home/danielle/Research/IA_measurement_GGL/IA_GGL/txtfiles/DESY1_quantities_fromSara/lenz_cent.dat')
	else:
		print("We don't have support for that survey yet. Exiting.")
		exit()
		
	cosmo = ccl.Cosmology(Omega_c = pa.OmC_t, Omega_b = pa.OmB, h = (pa.HH0_t/100.), sigma8 = pa.sigma8, n_s=pa.n_s)
	
	h = (pa.HH0_t/100.)
	Pk = np.zeros((len(k),len(z)))
	#zsave_list = [0]*len(z)
	f = open("./txtfiles/z_list_"+survey+".txt", "w")
	for zi in range(0,len(z)):
		zsave=str('{:1.12f}'.format(z[zi]))
		#zsave_list[zi] = zsave
		f.write(zsave+'\n')
		if (os.path.isfile('./txtfiles/halofit_Pk/Pk_nonlin_z='+zsave+'_'+endfile+'.txt')):
			print("Warning: Pk 2halo at z="+zsave+" already exists.")
			k_dummy, Pk[:,zi] = np.loadtxt('./txtfiles/halofit_Pk/Pk_nonlin_z='+zsave+'_'+endfile+'.txt', unpack=True)
		else:	
			Pk[:,zi] = h**3 * ccl.nonlin_matter_power(cosmo, k * h , 1./(1.+z[zi])) # CCL takes units without little h's, but we use little h units.
			save_thing = np.column_stack((k, Pk[:,zi]))
			np.savetxt('./txtfiles/halofit_Pk/Pk_nonlin_z='+zsave+'_'+endfile+'.txt', save_thing)
	f.close()
		
	
	if (os.path.isfile('./txtfiles/halofit_Pk/Pk_zavg_'+survey+'_'+endfile+'.txt')):
		print("Warning: Pk 2halo integrated over z already exists.")
		return
	else:	
		dndzl = setup.get_dNdzL(z, survey)
	
		Pk_avgZ = np.zeros(len(k))
		for ki in range(0,len(k)):
			Pk_avgZ[ki] = scipy.integrate.simps(Pk[ki, :] * dndzl, z)
    
		save_Pk = np.column_stack((k, Pk_avgZ))
		np.savetxt('./txtfiles/halofit_Pk/Pk_zavg_'+survey+'_'+endfile+'.txt', save_Pk)
		
	return
	
def get_Pkgg_1halo_multiz(kvec_ft, fsky, Mhalo, kvec_short, y_src, y_lens, Mstarlow, endfile, survey):
    """ Return the 1halo galaxy power spectrum of lenses x sources at multiple redshift for calculating the boost. """
	
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()
	
    if (survey=='SDSS'):
        z = np.asarray([0.16, 0.16202020202020204, 0.16404040404040404, 0.16606060606060608, 0.16808080808080808, 0.17010101010101011, 0.17212121212121212, 0.17414141414141415, 0.17616161616161616, 0.17818181818181819, 0.1802020202020202, 0.18222222222222223, 0.18424242424242424, 0.18626262626262627, 0.18828282828282827, 0.19030303030303031, 0.19232323232323234, 0.19434343434343435, 0.19636363636363635, 0.19838383838383838, 0.20040404040404042, 0.20242424242424242, 0.20444444444444446, 0.20646464646464646, 0.2084848484848485, 0.2105050505050505, 0.21252525252525253, 0.21454545454545454, 0.21656565656565657, 0.21858585858585861, 0.22060606060606061, 0.22262626262626262, 0.22464646464646465, 0.22666666666666668, 0.22868686868686869, 0.23070707070707069, 0.23272727272727273, 0.23474747474747476, 0.23676767676767677, 0.23878787878787877, 0.2408080808080808, 0.24282828282828284, 0.24484848484848487, 0.24686868686868688, 0.24888888888888888, 0.25090909090909091, 0.25292929292929295, 0.25494949494949493, 0.25696969696969696, 0.25898989898989899, 0.26101010101010103, 0.26303030303030306, 0.26505050505050504, 0.26707070707070707, 0.2690909090909091, 0.27111111111111108, 0.27313131313131311, 0.27515151515151515, 0.27717171717171718, 0.27919191919191921, 0.28121212121212125, 0.28323232323232322, 0.28525252525252526, 0.28727272727272724, 0.28929292929292927, 0.2913131313131313, 0.29333333333333333, 0.29535353535353537, 0.2973737373737374, 0.29939393939393943, 0.30141414141414141, 0.30343434343434345, 0.30545454545454542, 0.30747474747474746, 0.30949494949494949, 0.31151515151515152, 0.31353535353535356, 0.31555555555555559, 0.31757575757575757, 0.3195959595959596, 0.32161616161616158, 0.32363636363636361, 0.32565656565656564, 0.32767676767676768, 0.32969696969696971, 0.33171717171717174, 0.33373737373737378, 0.33575757575757575, 0.33777777777777779, 0.33979797979797977, 0.3418181818181818, 0.34383838383838383, 0.34585858585858587, 0.3478787878787879, 0.34989898989898993, 0.35191919191919191, 0.35393939393939394, 0.35595959595959598, 0.35797979797979795, 0.35999999999999999])
    elif (survey=='LSST_DESI'):
        z = np.asarray([0.025000000000000001, 0.036616161616161616, 0.048232323232323238, 0.059848484848484852, 0.071464646464646481, 0.083080808080808088, 0.094696969696969696, 0.10631313131313133, 0.11792929292929294, 0.12954545454545457, 0.14116161616161618, 0.15277777777777779, 0.1643939393939394, 0.17601010101010103, 0.18762626262626264, 0.19924242424242428, 0.21085858585858588, 0.22247474747474749, 0.23409090909090913, 0.24570707070707073, 0.2573232323232324, 0.26893939393939398, 0.28055555555555561, 0.29217171717171725, 0.30378787878787883, 0.31540404040404046, 0.3270202020202021, 0.33863636363636374, 0.35025252525252532, 0.36186868686868695, 0.37348484848484859, 0.38510101010101017, 0.3967171717171718, 0.40833333333333344, 0.41994949494949502, 0.43156565656565665, 0.44318181818181829, 0.45479797979797987, 0.4664141414141415, 0.47803030303030314, 0.48964646464646477, 0.5012626262626263, 0.51287878787878793, 0.52449494949494957, 0.5361111111111112, 0.54772727272727284, 0.55934343434343448, 0.57095959595959611, 0.58257575757575764, 0.59419191919191927, 0.60580808080808091, 0.61742424242424254, 0.62904040404040418, 0.64065656565656581, 0.65227272727272745, 0.66388888888888897, 0.67550505050505061, 0.68712121212121224, 0.69873737373737388, 0.71035353535353551, 0.72196969696969715, 0.73358585858585867, 0.74520202020202031, 0.75681818181818195, 0.76843434343434358, 0.78005050505050522, 0.79166666666666685, 0.80328282828282849, 0.81489898989899001, 0.82651515151515165, 0.83813131313131328, 0.84974747474747492, 0.86136363636363655, 0.87297979797979819, 0.88459595959595971, 0.89621212121212135, 0.90782828282828298, 0.91944444444444462, 0.93106060606060626, 0.94267676767676789, 0.95429292929292953, 0.96590909090909105, 0.97752525252525269, 0.98914141414141432, 1.000757575757576, 1.0123737373737376, 1.0239898989898992, 1.0356060606060606, 1.0472222222222223, 1.0588383838383839, 1.0704545454545455, 1.0820707070707072, 1.0936868686868688, 1.1053030303030305, 1.1169191919191921, 1.1285353535353537, 1.1401515151515151, 1.1517676767676768, 1.1633838383838384, 1.175])
    elif(survey=='DESY1'):
        z = np.loadtxt('/home/danielle/Research/IA_measurement_GGL/IA_GGL/txtfiles/DESY1_quantities_fromSara/lenz_cent.dat')    
    else:
        print("We don't have support for that survey yet. Exiting.")
        exit()
		
    np.savetxt('./txtfiles/1halo_terms/z_list_'+survey+'.txt', z, fmt="%1.12f")
	
    zsave = [0]*len(z)
    for i in range(0,len(z)):
        zsave[i]=str('{:1.12f}'.format(z[i]))
		
    # Check if this has already been run for this file name:
    file_present = 0
    for zi in range(0,len(z)):
        if (os.path.isfile('./txtfiles/1halo_terms/Pk1h_ls_z='+zsave[zi]+'_'+endfile+'.txt')):
            file_present = 1
    if file_present==1:
        print("Pkgg 1halo multiz has already been run for this endfile.")
        return
		
    # For the cosmological parameters:
    # I'm going to use the SOURCE parameters here
    # For SDSS source and lens cosmological parameters are the same
    # For LSST+DESI, they are not too different, but DESI HOD is already weird because I have to use CMASS
    # So use the ones consistent with sources.
	
    # Get the halo mass function from CCL
    cosmo = ccl.Cosmology(Omega_c = pa.OmC_s, Omega_b = pa.OmB_s, h = (pa.HH0_s/100.), sigma8 = pa.sigma8_s, n_s=pa.n_s_s)
    HMF = np.zeros((len(Mhalo), len(z)))
    for zi in range(0,len(z)):
        HMF_class = ccl.halos.MassFuncTinker10(cosmo)
        HMF[:,zi] = HMF_class.get_mass_function(cosmo, Mhalo / (pa.HH0_s/100.), 1./ (1. + z[zi])) / (pa.HH0_s / 100.)**3
        #HMF[:, zi] = ccl.massfunction.massfunc(cosmo, Mhalo / (pa.HH0_s/100.), 1./ (1. + z[zi]), odelta=200.) / (pa.HH0_s / 100.)**3
		
    # Get the mean number of centrals and satelites for the appropriate HOD. We assume none of the sources are centrals in galaxies with satellites from the lenses.
    if (survey == 'SDSS'):
        Ncen_lens 	= 	get_Ncen_Reid(Mhalo, survey)  		# Reid & Spergel
        Nsat_lens 	= 	get_Nsat_Reid(Mhalo, survey)  		# Reid & Spergel 
        Ncen_src	= 	get_Ncen_Zu(Mhalo, Mstarlow, survey) # Zu & Mandelbaum 2015
        Nsat_src_tot 	= 	get_Nsat_Zu(Mhalo, Mstarlow, 'tot', survey)  	# Zu & Mandelbaum 2015 - the halo occupation including all satelite galaxies
        Nsat_src_wlens	= 	get_Nsat_Zu(Mhalo, Mstarlow, 'with_lens', survey)  	# Zu & Mandelbaum 2015 - the halo occupation including only those satelite galaxies which share halos with central lenses
		
    elif (survey== 'LSST_DESI'):
        Ncen_lens 	= 	get_Ncen_More(Mhalo, survey) # CMASS
        Nsat_lens 	= 	get_Nsat_More(Mhalo, survey) # CMASS 
        Ncen_src	= 	get_Ncen_Zu(Mhalo, Mstarlow, survey) # Zu & Mandelbaum 2015
        Nsat_src_tot 	= 	get_Nsat_Zu(Mhalo, Mstarlow, 'tot', survey)  	# Zu & Mandelbaum 2015 - the halo occupation including all satelite galaxies 
        Nsat_src_wlens	= 	get_Nsat_Zu(Mhalo, Mstarlow, 'with_lens', survey)  	# Zu & Mandelbaum 2015 - the halo occupation including only those satelite galaxies which share halos with central lenses
        
    elif (survey == 'DESY1'):
        Ncen_lens 	= 	get_Ncen_Reid(Mhalo, survey)  		# Reid & Spergel
        Nsat_lens 	= 	get_Nsat_Reid(Mhalo, survey)  		# Reid & Spergel 
        Ncen_src	= 	get_Ncen_Zu(Mhalo, Mstarlow, survey) # Zu & Mandelbaum 2015
        Nsat_src_tot 	= 	get_Nsat_Zu(Mhalo, Mstarlow, 'tot', survey)  	# Zu & Mandelbaum 2015 - the halo occupation including all satelite galaxies
        Nsat_src_wlens	= 	get_Nsat_Zu(Mhalo, Mstarlow, 'with_lens', survey)  	# Zu & Mandelbaum 2015 - the halo occupation including only those satelite galaxies which share halos with central lenses
		
    # Get the number density predicted by the halo model
    tot_ng = np.zeros(len(z)); tot_nsrc=np.zeros(len(z))
    for zi in range(0, len(z)):
        tot_ng[zi] = scipy.integrate.simps( ( Ncen_lens + Nsat_lens) * HMF[:, zi], np.log10(Mhalo / (pa.HH0_s/100.) ) ) 
        tot_nsrc[zi] = scipy.integrate.simps(( Ncen_src + Nsat_src_tot) * HMF[:, zi], np.log10(Mhalo / (pa.HH0_s/100.) ) )

    # We assume Poisson statistics because it doesn't make much difference for us..
    NcNs = Ncen_lens * Nsat_src_wlens  #NcenNsat(1., Ncen_lens, Nsat_src_wlens) # The average number of central-satelite pairs in a halo of mass M
    NsNs = Nsat_lens * Nsat_src_wlens #NsatNsat(1., Nsat_lens, Nsat_src_wlens) # The average number of satelite-satelite pairs in a halo of mass M
	
    # Get Pkgg in terms of z and k
    Pkgg = np.zeros((len(kvec_short), len(z)))
    for ki in range(0,len(kvec_short)):
        for zi in range(0,len(z)):
            Pkgg[ki, zi] = scipy.integrate.simps( HMF[:, zi] * (NcNs * y_src[ki, :] + NsNs * y_src[ki, :]*y_lens[ki,:]), np.log10(Mhalo / (pa.HH0_s/100.)  )) / (tot_nsrc[zi] * tot_ng[zi]) 
	
    for zi in range(0,len(z)):
        Pkgg_interp = scipy.interpolate.interp1d(np.log(kvec_short), np.log(Pkgg[:, zi]))
        logPkgg = Pkgg_interp(np.log(kvec_ft))
        Pkgg_longk = np.exp(logPkgg)
        save_P1h = np.column_stack((kvec_ft, Pkgg_longk))
        np.savetxt('./txtfiles/1halo_terms/Pk1h_ls_z='+zsave[zi]+'_'+endfile+'.txt', save_P1h)
	
    return

def get_Pkgg_1halo(kvec_ft, fsky, Mhalo, kvec_short, y_src, y_lens, Mstarlow, endfile, survey):
    """ Returns the 1halo galaxy power spectrum with c-s and s-s terms"""
	
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()
		
    if (os.path.isfile('./txtfiles/1halo_terms/Pkgg_1h_'+survey+'_'+endfile+'.txt')):
        print("Pkgg 1halo averaged over z already exists for this endfile.")
        return
		
    # Get the combined redshift window function of lens and source samples.
    (z, W_z) = window(survey) # Comment this out if getting 1-halo functions for the boost.
	
    # To produce 1-halo functions at each zL for getting the Boost, uncomment this:
    z = np.linspace(pa.zLmin, pa.zLmax, 100)
	
    # For the cosmological parameters:
    # I'm going to use the SOURCE parameters here
    # For SDSS source and lens cosmological parameters are the same
    # For LSST+DESI, they are not too different, but DESI HOD is already weird because I have to use CMASS
    # So use the ones consistent with sources.
	
    # Get the halo mass function from CCL
    cosmo = ccl.Cosmology(Omega_c = pa.OmC_s, Omega_b = pa.OmB_s, h = (pa.HH0_s/100.), sigma8 = pa.sigma8_s, n_s=pa.n_s_s)
    
    HMF = np.zeros((len(Mhalo), len(z)))
    for zi in range(0,len(z)):
        HMF_class = ccl.halos.MassFuncTinker10(cosmo)
        HMF[:,zi] = HMF_class.get_mass_function(cosmo, Mhalo / (pa.HH0_s/100.), 1./ (1. + z[zi])) / (pa.HH0_s / 100.)**3
        #HMF[:, zi] = ccl.massfunction.massfunc(cosmo, Mhalo / (pa.HH0_s/100.), 1./ (1. + z[zi]), odelta=200.) / (pa.HH0_s / 100.)**3
	
    # Get the mean number of centrals and satelites for the appropriate HOD. We assume none of the sources are centrals in galaxies with satellites from the lenses.
    if (survey == 'SDSS'):
        Ncen_lens 	= 	get_Ncen_Reid(Mhalo, survey)  		# Reid & Spergel
        Nsat_lens 	= 	get_Nsat_Reid(Mhalo, survey)  		# Reid & Spergel 
        Ncen_src	= 	get_Ncen_Zu(Mhalo, Mstarlow, survey) # Zu & Mandelbaum 2015
        Nsat_src_tot 	= 	get_Nsat_Zu(Mhalo, Mstarlow, 'tot', survey)  	# Zu & Mandelbaum 2015 - the halo occupation including all satelite galaxies
        Nsat_src_wlens	= 	get_Nsat_Zu(Mhalo, Mstarlow, 'with_lens', survey)  	# Zu & Mandelbaum 2015 - the halo occupation including only those satelite galaxies which share halos with central lenses
		
    elif (survey== 'LSST_DESI'):
        Ncen_lens 	= 	get_Ncen_More(Mhalo, survey) # CMASS
        Nsat_lens 	= 	get_Nsat_More(Mhalo, survey) # CMASS 
        Ncen_src	= 	get_Ncen_Zu(Mhalo, Mstarlow, survey) # Zu & Mandelbaum 2015
        Nsat_src_tot 	= 	get_Nsat_Zu(Mhalo, Mstarlow, 'tot', survey)  	# Zu & Mandelbaum 2015 - the halo occupation including all satelite galaxies
        Nsat_src_wlens	= 	get_Nsat_Zu(Mhalo, Mstarlow, 'with_lens', survey)  	# Zu & Mandelbaum 2015 - the halo occupation including only those satelite galaxies which share halos with central lenses
        
        print("Ncen_src=", Ncen_src)
        print("Nsat_src_tot=", Nsat_src_tot)
        print("Nsat_src_wlens=", Nsat_src_wlens)
		
    # Get the number density predicted by the halo model
    tot_ng = np.zeros(len(z)); tot_nsrc=np.zeros(len(z))
    tot_cen = np.zeros(len(z)); tot_sat = np.zeros(len(z))
    for zi in range(0, len(z)):
        tot_ng[zi] = scipy.integrate.simps( ( Ncen_lens + Nsat_lens) * HMF[:, zi], np.log10(Mhalo / (pa.HH0_s/100.) ) ) 
        tot_nsrc[zi] = scipy.integrate.simps(( Ncen_src + Nsat_src_tot) * HMF[:, zi], np.log10(Mhalo / (pa.HH0/100.) ) )  # all the sources in the sample 
        tot_cen[zi] = scipy.integrate.simps( ( Ncen_src) * HMF[:, zi], np.log10(Mhalo / (pa.HH0_s/100.) ) ) 
        tot_sat[zi] = scipy.integrate.simps( ( Nsat_src_tot) * HMF[:, zi], np.log10(Mhalo / (pa.HH0_s/100.) ) ) 
        
	
    # We assume Poisson statistics because it doesn't make much difference for us..
    NcNs = Ncen_lens * Nsat_src_wlens #NcenNsat(1., Ncen_lens, Nsat_src_wlens) # The average number of central-satelite pairs in a halo of mass M # Count only the sources that actually share halos with lenses. 
    NsNs = Nsat_lens * Nsat_src_wlens #NsatNsat(1., Nsat_lens, Nsat_src_wlens) # The average number of satelite-satelite pairs in a halo of mass M

    # Get Pkgg in terms of z and k
    Pkgg = np.zeros((len(kvec_short), len(z)))
    for ki in range(0,len(kvec_short)):
        for zi in range(0,len(z)):
            Pkgg[ki, zi] = scipy.integrate.simps( HMF[:, zi] * (NcNs * y_src[ki, :] + NsNs * y_src[ki, :]*y_lens[ki,:]), np.log10(Mhalo / (pa.HH0_s/100.)  )) / (tot_nsrc[zi] * tot_ng[zi]) 

    # Now integrate this over the window function
    Pkgg_zavg = np.zeros(len(kvec_short))
    for ki in range(0,len(kvec_short)):
        Pkgg_zavg[ki] = scipy.integrate.simps(W_z * Pkgg[ki, :], z)
    
    print("tot_cen=", scipy.integrate.simps(W_z*tot_cen, z))
    print("tot_sat=", scipy.integrate.simps(W_z*tot_sat, z))
	
    # Get the answer in terms of the full k vector for fourier transforming.
    Pkgg_interp = scipy.interpolate.interp1d(np.log(kvec_short), np.log(Pkgg_zavg))
    logPkgg = Pkgg_interp(np.log(kvec_ft))
    Pkgg_ft = np.exp(logPkgg)
	
    Pkgg_save = np.column_stack((kvec_ft, Pkgg_ft))
    np.savetxt('./txtfiles/1halo_terms/Pkgg_1h_'+survey+'_'+endfile+'.txt', Pkgg_save)
	
    """plt.figure()
    plt.loglog(kvec_short, 4* np.pi * kvec_short**3 * Pkgg_zavg / (2* np.pi)**3, 'm+')
    plt.ylim(0.001, 100000)
    plt.xlim(0.01, 100)
    plt.ylabel('$4\pi k^3 P_{gg}^{1h}(k)$, $(Mpc/h)^3$, com')
    plt.xlabel('$k$, h/Mpc, com')
    plt.savefig('./plots/Pkgg_1halo_'+survey+'.pdf')
    plt.close()"""
	
    return 

def get_Pkgm_1halo(kvec_FT, Mhalo, kvec_short, y, endfile, survey):
    """ Returns (and more usefully saves) the 1halo lens galaxies x dark matter power spectrum, for the calculation of Delta Sigma (theoretical) """
	
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()
		
    if (os.path.isfile('./txtfiles/1halo_terms/Pkgm_1h_'+survey+'_'+endfile+'.txt')):
        print("Pkgm 1halo averaged over z already exists.")
        return

    if survey == 'SDSS' or survey == 'LSST_DESI':
        # Define the vector of lens redshifts over which we will average.
        zLvec = np.linspace(pa.zLmin, pa.zLmax, 500)
        dndzl = setup.get_dNdzL(zLvec, survey)
    elif(survey=='DESY1'):
        zLvec, dndzl = np.loadtxt('./txtfiles/'+pa.dNdzL_file, unpack=True) 
	
    # Get the halo mass function
    cosmo = ccl.Cosmology(Omega_c = pa.OmC_l, Omega_b = pa.OmB_l, h = (pa.HH0_l/100.), sigma8=pa.sigma8_l, n_s=pa.n_s_l)
    HMF = np.zeros((len(Mhalo), len(zLvec)))
    for zi in range(0, len(zLvec)):
        HMF_class = ccl.halos.MassFuncTinker10(cosmo)
        HMF[:,zi] = HMF_class.get_mass_function(cosmo, Mhalo / (pa.HH0_l/100.), 1./ (1. + zLvec[zi]))
        #HMF[:, zi]= ccl.massfunction.massfunc( cosmo, Mhalo / (pa.HH0_l/100.), 1./ (1. + zLvec[zi]), odelta=200. )
	
    # Get HOD quantities we need
    if (survey=='SDSS'):
        Ncen_lens = get_Ncen_Reid(Mhalo, survey) # We use the LRG model for the lenses from Reid & Spergel 2008
        Nsat_lens = get_Nsat_Reid(Mhalo, survey)
    elif (survey=='LSST_DESI'):
        Ncen_lens = get_Ncen_More(Mhalo, survey)
        Nsat_lens = get_Nsat_More(Mhalo, survey)
    elif (survey == 'DESY1'):
        Ncen_lens = get_Ncen_Reid(Mhalo, survey) # We use the LRG model for the lenses from Reid & Spergel 2008
        Nsat_lens = get_Nsat_Reid(Mhalo, survey)
    else:
        print("We don't have support for that survey yet!")
        exit()
		
    # Check total number of galaxies:
    tot_ng= np.zeros(len(zLvec))
    for zi in range(0,len(zLvec)):
        tot_ng[zi] = scipy.integrate.simps( ( Ncen_lens + Nsat_lens) * HMF[:, zi], np.log10(Mhalo / (pa.HH0_l/100.) ) ) / (pa.HH0_l / 100.)**3
	
    # Get the density of matter in comoving coordinates
    rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h (comoving distances)
    rho_m = (pa.OmC_l + pa.OmB_l) * rho_crit # units of Msol h^2 / Mpc^3 (comoving distances)
	
    # Get Pk
    Pkgm = np.zeros((len(kvec_short), len(zLvec)))
    for ki in range(0,len(kvec_short)):
        for zi in range(0, len(zLvec)):
            Pkgm[ki, zi] = scipy.integrate.simps( HMF[:, zi] * (Mhalo / rho_m) * (Ncen_lens * y[ki, :] + Nsat_lens * y[ki, :]**2), np.log10(Mhalo / (pa.HH0_l/ 100.))) / (tot_ng[zi]) / (pa.HH0_l / 100.)**3
		
    # Now integrate this over the appropriate lens redshift distribution:

    Pk_zavg = np.zeros(len(kvec_short))
    for ki in range(0,len(kvec_short)):
        Pk_zavg[ki] = scipy.integrate.simps(dndzl * Pkgm[ki, :], zLvec)
	
    """plt.figure()
    plt.loglog(kvec_short, 4* np.pi * kvec_short**3 * Pk_zavg / (2* np.pi)**3, 'mo')
    plt.ylim(0.1, 100000)
    plt.xlim(0.05, 1000)
    plt.ylabel('$4\pi k^3 P_{gg}^{1h}(k)$, $(Mpc/h)^3$, com')
    plt.xlabel('$k$, h/Mpc, com')
    plt.savefig('./plots/Pkgm_1halo_survey='+SURVEY+'.pdf')
    plt.close()"""
	
    # Get this in terms of the more well-sampled k, for fourier transforming, and save.
    Pkgm_interp = scipy.interpolate.interp1d(np.log(kvec_short), np.log(Pk_zavg))
    logPkgm = Pkgm_interp(np.log(kvec_FT))
    Pkgm = np.exp(logPkgm)
    Pkgm_save = np.column_stack((kvec_FT, Pkgm))
    np.savetxt('./txtfiles/1halo_terms/Pkgm_1h_'+survey+'_'+endfile+'.txt', Pkgm_save)
	
    """plt.figure()
    plt.loglog(kvec_FT, 4* np.pi * kvec_FT**3 * Pkgm / (2* np.pi)**3, 'mo')
    plt.ylim(0.001, 100000)
    plt.xlim(0.01, 10000)
    plt.ylabel('$4\pi k^3 P_{gg}^{1h}(k)$, $(Mpc/h)^3$, com')
    plt.xlabel('$k$, h/Mpc, com')
    plt.savefig('./plots/Pkgm_1halo_longerkvec_survey='+SURVEY+'.pdf')
    plt.close()"""
	
    return 

def get_Pkgg_ll_1halo_kz(kvec, zvec, y, Mhalo, kvec_short, survey):
	""" Returns the 1halo galaxy power spectrum with c-s and s-s terms for lenses x lenses (for the covariance). """

	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print("We don't have support for that survey yet; exiting.")
		exit()
	
	# Define the downsampled k and z vector over which we will compute Pk_{gm}^{1h}
	#kvec_short = np.logspace(np.log10(kvec[0]), np.log10(kvec[-1]), 40)
	zvec_short = np.linspace(zvec[0]-0.00000001, zvec[-1]+0.00000001, 40)
	#Mhalo = np.logspace(7., 16., 30)
	
	# Get the halo mass function at each z (use CCL)
	p = ccl.Parameters(Omega_c = pa.OmC_l, Omega_b = pa.OmB_l, h = (pa.HH0_l/100.), sigma8=pa.sigma8_l, n_s=pa.n_s_l)
	cosmo = ccl.Cosmology(p)
	HMF = np.zeros((len(Mhalo), len(zvec_short)))
	for zi in range(0,len(zvec_short)):
		HMF[:, zi]= ccl.massfunction.massfunc( cosmo, Mhalo / (pa.HH0_l/100.), 1./ (1. + zvec_short[zi]), odelta=200. )
	
	# We're going to use, for the centrals and satelite lenses, either the Reid & Spergel 2008 HOD (SDSS LRGs) or the CMASS More et al. 2014 HOD (DESI LRGS).
	if (survey == 'SDSS'):
		Ncen_lens = get_Ncen_Reid(Mhalo, survey)  # Reid & Spergel
		Nsat_lens = get_Nsat_Reid(Mhalo, survey)  # Reid & Spergel 
	elif (survey== 'LSST_DESI'):
		Ncen_lens = get_Ncen_More(Mhalo, survey) # CMASS
		Nsat_lens = get_Nsat_More(Mhalo, survey) # CMASS 
	
	# Get the number density predicted by the halo model 
	tot_ng = np.zeros(len(zvec_short))
	for zi in range(0, len(zvec_short)):
		tot_ng[zi] = scipy.integrate.simps( ( Ncen_lens + Nsat_lens) * HMF[:, zi], np.log10(Mhalo / (pa.HH0_l/100.) ) ) / (pa.HH0_l / 100.)**3
	
	alpha_sq = np.ones(len(Mhalo)) # We assume Poisson statistics because it doesn't make much difference for us.
	NcNs = NcenNsat(alpha_sq, Ncen_lens, Nsat_lens) # The average number of central-satelite pairs in a halo of mass M
	NsNs = NsatNsat(alpha_sq, Nsat_lens, Nsat_lens) # The average number of satelite-satelite pairs in a halo of mass M
	
	#y = gety(Mhalo, kvec_short, survey) # Mass-averaged Fourier transform of the density profile

	Pkgg = np.zeros((len(kvec_short), len(zvec_short)))
	for ki in range(0,len(kvec_short)):
		for zi in range(0, len(zvec_short)):
			Pkgg[ki, zi] = scipy.integrate.simps( HMF[:, zi] * (NcNs * y[ki, :] + NsNs * y[ki, :]**2), np.log10(Mhalo / (pa.HH0_l/100.)  )) / (tot_ng[zi]**2) / (pa.HH0_l / 100.)**3
	
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

def get_Pkgm_1halo_kz(kvec, zvec, y, Mhalo, kvec_short, survey):
	""" Returns the 1halo lens galaxies x dark matter power spectrum at the given k and z values """
	
	# Get the average halo mass:
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print("We don't have support for that survey yet; exiting.")
		exit()
	
	# Define the downsampled k and z vector over which we will compute Pk_{gm}^{1h}
	#kvec_short = np.logspace(np.log10(kvec[0]), np.log10(kvec[-1]), 40)
	zvec_short = np.linspace(zvec[0]-0.00000001, zvec[-1]+0.00000001, 40)
	#Mhalo = np.logspace(7., 16., 30)
	
	# Get the halo mass function at each z (use CCL)
	p = ccl.Parameters(Omega_c = pa.OmC_l, Omega_b = pa.OmB_l, h = (pa.HH0_l/100.), sigma8 = pa.sigma8_l, n_s=pa.n_s_l)
	cosmo = ccl.Cosmology(p)
	HMF = np.zeros((len(Mhalo), len(zvec_short)))
	for zi in range(0,len(zvec_short)):
		HMF[:, zi]= ccl.massfunction.massfunc( cosmo, Mhalo / (pa.HH0_l/100.), 1./ (1. + zvec_short[zi]), odelta=200. )
	
	if (survey=='SDSS'):
		Ncen_lens = get_Ncen_Reid(Mhalo, survey) # We use the LRG model for the lenses from Reid & Spergel 2008
		Nsat_lens = get_Nsat_Reid(Mhalo, survey) 
	elif (survey=='LSST_DESI'):
		Ncen_lens = get_Ncen_More(Mhalo, survey)
		Nsat_lens = get_Nsat_More(Mhalo, survey)
	else:
		print("We don't have support for that survey yet!")
		exit()
		
	# Get total number of galaxies (this is z-dependent) 
	tot_ng = np.zeros(len(zvec_short))
	for zi in range(0,len(zvec_short)):
		tot_ng[zi] = scipy.integrate.simps( ( Ncen_lens + Nsat_lens) * HMF[:, zi], np.log10(Mhalo / (pa.HH0_l/100.) ) ) / (pa.HH0_l / 100.)**3

	# Get the fourier space NFW profile equivalent
	#y = gety(Mhalo, kvec_short, survey) 
	
	# Get the density of matter in comoving coordinates
	rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h (comoving distances)
	rho_m = (pa.OmC_l + pa.OmB_l) * rho_crit # units of Msol h^2 / Mpc^3 (comoving distances)
	
	# Get Pk
	Pkgm = np.zeros((len(kvec_short), len(zvec_short)))
	for ki in range(0,len(kvec_short)):
		for zi in range(0,len(zvec_short)):
			Pkgm[ki, zi] = scipy.integrate.simps( HMF[:, zi] * (Mhalo / rho_m) * (Ncen_lens * y[ki, :] + Nsat_lens * y[ki, :]**2), np.log10(Mhalo / (pa.HH0_l/ 100.))) / (tot_ng[zi]) / (pa.HH0_l / 100.)**3
	
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
	
def get_Pkmm_1halo_kz(kvec, zvec, y, Mhalo, kvec_short, survey):
	""" Returns the 1halo lens galaxies x dark matter power spectrum at the given k and z values """
	
	# Get the average halo mass:
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print("We don't have support for that survey yet; exiting.")
		exit()
	
	# Define the downsampled k and z vector over which we will compute Pk_{gm}^{1h}
	#kvec_short = np.logspace(np.log10(kvec[0]), np.log10(kvec[-1]), 40)
	zvec_short = np.linspace(zvec[0]-0.00000001, zvec[-1]+0.00000001, 40)
	#Mhalo = np.logspace(7., 16., 30)
	
	# Here use the standard cosmological parameters because there's no good reason not to - no HOD involved
	
	# Get the halo mass function at each z (use CCL)
	p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), A_s = pa.A_s, n_s=pa.n_s_cosmo)
	cosmo = ccl.Cosmology(p)
	HMF = np.zeros((len(Mhalo), len(zvec_short)))
	for zi in range(0,len(zvec_short)):
		HMF[:, zi]= ccl.massfunction.massfunc( cosmo, Mhalo / (pa.HH0/100.), 1./ (1. + zvec_short[zi]), odelta=200. )

	# Get the fourier space NFW profile equivalent
	#y = gety(Mhalo, kvec_short, survey) 
	
	# Get the density of matter in comoving coordinates
	rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h (comoving distances)
	rho_m = (pa.OmC + pa.OmB) * rho_crit # units of Msol h^2 / Mpc^3 (comoving distances)
	
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
	
def gety_ls(Mvec, kvec_gety, survey):
    """ Fourier transforms the density profile to get the power spectrum. """
	
    # Get the nfw density profile at the correct mass and redshift and at a variety of r
    rvec = [0]*len(Mvec)
    rho = [0]*len(Mvec)
    for Mi in range(0,len(Mvec)):
        Rvir = Rhalo(Mvec[Mi], survey)
        rvec[Mi] = np.logspace(-8, np.log10(Rvir), 10**6)
        rho[Mi] = rho_NFW_ls(rvec[Mi], Mvec[Mi], survey)  # Units Msol h^2 / Mpc^3, comoving. 

    # This should be an FFT
    u_ = np.zeros((len(kvec_gety), len(Mvec)))
    for ki in range(0,len(kvec_gety)):
        for mi in range(0,len(Mvec)):
            u_[ki, mi] = 4. * np.pi / Mvec[mi] * scipy.integrate.simps( rvec[mi] * np.sin(kvec_gety[ki]*rvec[mi])/ kvec_gety[ki] * rho[mi], rvec[mi]) # unitless / dimensionless.
	
    return u_
	
def gety_ldm(Mvec, kvec_gety, survey):
    """ Fourier transforms the density profile to get the power spectrum. """
	
    # Get the nfw density profile at the correct mass and redshift and at a variety of r
    rvec = [0]*len(Mvec)
    rho = [0]*len(Mvec)
    for Mi in range(0,len(Mvec)):
        Rvir = Rhalo(Mvec[Mi], survey)
        rvec[Mi] = np.logspace(-8, np.log10(Rvir), 10**6)
        rho[Mi] = rho_NFW_ldm(rvec[Mi], Mvec[Mi], survey)  # Units Msol h^2 / Mpc^3, comoving. 

    # This should be an FFT really.
    u_ = np.zeros((len(kvec_gety), len(Mvec)))
    for ki in range(0,len(kvec_gety)):
        for mi in range(0,len(Mvec)):
            u_[ki, mi] = 4. * np.pi / Mvec[mi] * scipy.integrate.simps( rvec[mi] * np.sin(kvec_gety[ki]*rvec[mi])/ kvec_gety[ki] * rho[mi], rvec[mi]) # unitless / dimensionless.
	
    return u_

###### Satelites occupation, Zu & Mandelbaum ######

def get_Nsat_Zu(M_h, Mstar, case, survey):
    """ Gets the number of source galaxies that are satelites in halos."""
	
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()
	
    # Uncomment the following to use halotools
    """if (case == 'tot'):
		if (( hasattr(Mstar, "__len__")==False) ):
			model = PrebuiltHodModelFactory('zu_mandelbaum15', threshold = np.log10(Mstar), prim_haloprop_key = 'halo_m200m')
			Nsat= model.mean_occupation_satellites(prim_haloprop=M_h)
			
		else:
			Nsat=np.zeros((len(Mstar), len(M_h)))
			for i in range(0,len(Mstar)):
				#print "Mstar in Nsat=", i
				for j in range(0,len(M_h)):
					model = PrebuiltHodModelFactory('zu_mandelbaum15', threshold = np.log10(Mstar[i]), prim_haloprop_key = 'halo_m200m')
					Nsat[i,j] = model.mean_occupation_satellites(prim_haloprop=M_h[j])
	elif (case =='with_lens'):
		if (survey=='SDSS'):
			if (( hasattr(Mstar, "__len__")==False) ):
				Ncen_Zu = get_Ncen_Zu(M_h, Mstar,survey) 
				Ncen_Reid = get_Ncen_Reid(M_h, survey)
				model = PrebuiltHodModelFactory('zu_mandelbaum15', threshold = np.log10(Mstar), prim_haloprop_key = 'halo_m200m')
				Nsat_tot = model.mean_occupation_satellites(prim_haloprop=M_h)
				Nsat = Nsat_tot / Ncen_Zu * Ncen_Reid
			else:
				print "Why are you calling Nsat with_lens for len(Mstar)!=1?"
				exit()
		elif (survey=='LSST_DESI'):
			if (( hasattr(Mstar, "__len__")==False) ):
				Ncen_Zu = get_Ncen_Zu(M_h, Mstar,survey) 
				Ncen_More = get_Ncen_More(M_h, survey)
				model = PrebuiltHodModelFactory('zu_mandelbaum15', threshold = np.log10(Mstar), prim_haloprop_key = 'halo_m200m')
				Nsat_tot = model.mean_occupation_satellites(prim_haloprop=M_h)
				Nsat = Nsat_tot / Ncen_Zu * Ncen_More
			else:
				print "Why are you calling Nsat with_lens for len(Mstar)!=1?"
				exit()
		else:
			print "We do not have support for that survey, exiting."
			exit()
	else:
		print "We don't have support for that case for Ncen in Nsat."
		exit()"""
	
    # Uncomment the following to use our own code
    Ncen_src = get_Ncen_Zu(M_h, Mstar,survey)
    if survey =='SDSS' or survey =='DESY1': 
        Ncen_lens = get_Ncen_Reid(M_h, survey)
    elif survey == 'LSST_DESI':
        Ncen_lens = get_Ncen_More(M_h, survey)
		
    f_Mh = get_inv_fSHMR(Mstar, survey)
    Msat = get_Msat(f_Mh, survey)
    Mcut = get_Mcut(f_Mh, survey)
		
    if (case == 'tot'):
        if (( hasattr(Mstar, "__len__")==False) ):
            Nsat = Ncen_src * (M_h / Msat)**(pa.alpha_sat) * np.exp(-Mcut / M_h)
        else:
            Nsat=np.zeros((len(Mstar), len(M_h)))
            for msi in range(0,len(Mstar)):
                #print "Mstar in Nsat=", i
                for mhi in range(0,len(M_h)):
                    Nsat[msi,mhi] = Ncen_src[msi,mhi] * (M_h[mhi] / Msat[msi])**(pa.alpha_sat) * np.exp(-Mcut[msi] / M_h[mhi])
    elif (case =='with_lens'):
        if (( hasattr(Mstar, "__len__")==False) ):
            Nsat = Ncen_lens * (M_h / Msat)**(pa.alpha_sat) * np.exp(-Mcut / M_h)
        else:
            print("Why are you calling Nsat with_lens for len(Mstar)!=1?")
            exit()
    else:
        print("We don't have support for that case for Ncen in Nsat.")
        exit()
	
    """#f_Mh = Mh_atfixed_Ms(Mstar)
	#Msat = get_Msat(f_Mh, survey)
	#Mcut = get_Mcut(f_Mh, survey)
		
	#Nsat = Ncen_src * (M_h / Msat)**(pa.alpha_sat) * np.exp(-Mcut / M_h)
	
	if ((type(M_h)==float) or (type(Mstar)==float) or(type(M_h)==np.float64) or (type(Mstar)==np.float64) ):
		Nsat = Ncen * (M_h / Msat)**(pa.alpha_sat) * np.exp(-Mcut / M_h)
	elif(((type(Mstar)==list) or isinstance(Mstar, np.ndarray)) and ((type(M_h)==list) or isinstance(M_h, np.ndarray))):
		Nsat=np.zeros((len(Mstar), len(M_h)))
		for i in range(0,len(Mstar)):
			for j in range(0,len(M_h)):
				Nsat[i,j] = Ncen[i,j] * (M_h[j] / Msat[i])**(pa.alpha_sat) * np.exp(-Mcut[i] / M_h[j])"""		
		
    return Nsat
	
def vol_dens(fsky, N,survey):
    """ Computes the volume density of galaxies given the fsky, minimum z, max z, and number of galaxies."""
	
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()

    if survey == 'DESY1':
        
        """# Load dNdzs from file for the appropriate source sample
        if(sample == 'B'):
            z = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_centres.dat')
            dNdz_unnormed = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin1_zmc_weighted')
        elif(sample=='A'):
            z = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin0_zmc_centres.dat')
            dNdz_unnormed = np.loadtxt('./txtfiles/DESY1_quantities_fromSara/bin0_zmc_weighted')
            
            #print("Using perturbed source redshift distribution in window()")
            #z_s, dNdz_2 = setup.dNdz_perturbed(sample, pa.sigma, pa.del_z)"""
        dndz = np.load('./im3_full_n_values.npz')
        z_edges = dndz['bins']
        dNdz_unnormed = dndz['weighted_counts']
        z = np.zeros(len(z_edges)-1)
        for i in range(0,len(z_edges)-1):
            z[i] = (z_edges[i+1]-z_edges[i])/2. + z_edges[i]
    else:    	
        (z, dNdz_unnormed) = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 500, survey)
	
    # Get dNdz, normalized to the number of source galaxies N
    norm = scipy.integrate.simps(dNdz_unnormed, z)
    dNdz_num = N * dNdz_unnormed / norm 
	
    # Get factors needed to change to n(z)
    # Use the source HOD cosmological parameters here to be consistent
    OmL = 1. - pa.OmC_s - pa.OmB_s - pa.OmR_t - pa.OmN_t
    H_over_c = pa.H0 * ( (pa.OmC_s+pa.OmB_s)*(1.+z)**3 + OmL + (pa.OmR_t+pa.OmN_t) * (1.+z)**4 )**(0.5)
	
    # volume density as a function of z
    # UPDATE CCL - use CCL comoving distance?
    cosmo_s = ccl.Cosmology(Omega_c = pa.OmC_s, Omega_b = pa.OmB_s, h = (pa.HH0_s/100.), sigma8 = pa.sigma8_s, n_s=pa.n_s_s)
    chi = ccl.comoving_radial_distance(cosmo_s, 1./(1.+z)) * (pa.HH0_s / 100.) # CCL returns in Mpc but we want Mpc/h
    ndens_ofz = dNdz_num * H_over_c / ( 4. * np.pi * fsky * chi**2 )
	
    # We want to integrate this over the window function of lenses x sources, because that's the redshift range on which we care about the number density:
    (z_win, win) = window(survey)
    interp_ndens = scipy.interpolate.interp1d(z, ndens_ofz)
    ndens_forwin = interp_ndens(z_win)
	
    ndens_avg = scipy.integrate.simps(ndens_forwin * win, z_win)
	
    return ndens_avg
	
def get_Mstar_low(survey, ngal):
    """ For a given number density of source galaxies (calculated in the vol_dens function), get the appropriate choice for the lower bound of Mstar """
	
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()
		
    # Get the window function for the lenses x sources
    (z, win) = window(survey)
	
    # Use the HOD model from Zu & Mandelbaum 2015
		
    # Define a vector of Mstar_low value to try
    Ms_low_vec = np.logspace(7., 10.,200)
    # Define a vector of Mh values to integrate over
    Mh_vec = np.logspace(9., 16., 100)
	
    # Get Nsat and Ncen as a function of the values of the two above arrays
    Nsat = get_Nsat_Zu(Mh_vec, Ms_low_vec, 'tot', survey) # Get the occupation number counting all sats in the sample.
    Ncen = get_Ncen_Zu(Mh_vec, Ms_low_vec, survey)
    
    # Get the halo mass function (from CCL) to integrate over (dn / dlog10M, Tinker 2010 )
    cosmo = ccl.Cosmology(Omega_c = pa.OmC_s, Omega_b = pa.OmB_s, h = (pa.HH0_s/100.), sigma8 = pa.sigma8_s, n_s=pa.n_s_s)
    HMF = np.zeros((len(Mh_vec), len(z)))
    #print("In Mstar low - check units of HMF!")
    for zi in range(0,len(z)):
        #print "zi=", zi
        HMF_class = ccl.halos.MassFuncTinker10(cosmo)
        HMF[:,zi] = HMF_class.get_mass_function(cosmo, Mh_vec / (pa.HH0_s/100.), 1./ (1. + z[zi]))
	
    # Now get what nsrc should be for each Mstar_low cut 
    nsrc_of_Mstar_z = np.zeros((len(Ms_low_vec), len(z)))
    for msi in range(0,len(Ms_low_vec)):
        #print "Msi=", i
        for zi in range(0,len(z)):
            nsrc_of_Mstar_z[msi, zi] = scipy.integrate.simps(HMF[:, zi] * ( Nsat[msi, :] + Ncen[msi, :]), np.log10(Mh_vec / (pa.HH0_s / 100.))) / (pa.HH0_s / 100.)**3
	
    # Integrate this over the z window function
    nsrc_of_Mstar = np.zeros(len(Ms_low_vec))
    for i in range(0,len(Ms_low_vec)):
        nsrc_of_Mstar[i] = scipy.integrate.simps(nsrc_of_Mstar_z[i, :] * win, z)
	
    # Get the correct Mstar cut	
    ind = next(j[0] for j in enumerate(nsrc_of_Mstar) if j[1]<=ngal)
		
    Mstarlow = Ms_low_vec[ind]
	
    return Mstarlow
		
def Mh_atfixed_Ms(Ms):
	""" Get Mh in terms of Mstar """
	
	# Using the relation given in equation 51 of Zu et al. 2015 to ensure we account for scatter.
	lgMh_fit = 4.41 * (1. + np.exp(-1.82* (np.log10(Ms) - 11.18)))**(-1) + 11.12 *np.sin(-0.12*(np.log10(Ms)-23.37))
	
	Mh = 10**(lgMh_fit)
	
	return Mh

def get_inv_fSHMR(Ms, survey):
    """ Get f_{SHMR}^{-1}(M*) as used in equation 23 and 24 of Zu et al. 2015. This is equation 19 directly."""
	
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()
	
    m = Ms / pa.Mso
	
    Mh = pa.M1 * m**(pa.beta) * 10.**( m**pa.delta / (1. + m**(-pa.gamma)) - 0.5)
	
    return Mh
	
def get_Msat(f_Mh, survey):
    """ Returns parameter representing the characteristic mass of a single-satelite hosting galaxy, Zu & Mandelbaum 2015."""
	
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()
	
    Msat = pa.Bsat * 10**12 * (f_Mh / 10**12)**pa.beta_sat
	
    return Msat
	
def get_Mcut(f_Mh, survey):
    """ Returns the parameter representing the cutoff mass scales """
	
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()
	
    Mcut = pa.Bcut * 10**12 * ( f_Mh / 10**12) ** pa.beta_cut
	
    return Mcut

#### For central occupation ####

def get_Ncen_Zu(Mh, Mstar, survey):
    """ Get the CUMULATIVE distribution of central galaxies for the sources from the HOD model from Zu & Mandelbaum 2015"""
	
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()
		
    # This is for the Zu & Mandelbaum 2015 halo model.
    sigmaMstar = get_sigMs(Mh, survey)
    fshmr = get_fSHMR(Mh, survey)
	
    if (( hasattr(Mstar, "__len__")==False) ):
        #model = PrebuiltHodModelFactory('zu_mandelbaum15', threshold = np.log10(Mstar), prim_haloprop_key = 'halo_m200m')
        #Ncen_CDF = model.mean_occupation_centrals(prim_haloprop=Mh)
        Ncen_CDF = 0.5 * (1. - scipy.special.erf((np.log(Mstar) - np.log(fshmr)) / (np.sqrt(2.) * sigmaMstar)))
    else:
        Ncen_CDF = np.zeros((len(Mstar), len(Mh)))
        for msi in range(0,len(Mstar)):
            #print("Mstar in Ncen=", i)
            for mhi in range(0, len(Mh)):
                #model = PrebuiltHodModelFactory('zu_mandelbaum15', threshold = np.log10(Mstar[i]), prim_haloprop_key = 'halo_m200m')
                #Ncen_CDF[i,j] = model.mean_occupation_centrals(prim_haloprop=Mh[j])
                Ncen_CDF[msi,mhi] = 0.5 * (1. - scipy.special.erf((np.log(Mstar[msi]) - np.log(fshmr[mhi])) / (np.sqrt(2.) * sigmaMstar[mhi])))
	
    return Ncen_CDF
	
def get_sigMs(Mh, survey):
    """ Get sigma_ln(M*) as a function of the halo mass."""
    
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()
	
    if (hasattr(Mh, "__len__")==False):
	
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
    """ Get the mean Mstar in terms of Mh using f_SHMR inverse relationship."""
	
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()
	
    Ms = np.logspace(1, 13, 5000)
	
    m = Ms / pa.Mso
    #Mh_vec = pa.M1 * m**(pa.beta) * np.exp( m**pa.delta / (1. + m**(-pa.gamma)) - 0.5)
    Mh_vec = pa.M1 * m**(pa.beta) * 10.**( m**pa.delta / (1. + m**(-pa.gamma)) - 0.5)
	
    Mh_interp = scipy.interpolate.interp1d(Mh_vec, Ms)
	
    Mstar_ans = Mh_interp(Mh)
	
    return Mstar_ans
	
	
##################################33
	
def get_Nsat_More(M_h, survey):
	""" Gets source galaxies in satelite halos from More et al. 2014 HOD"""
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print("We don't have support for that survey yet; exiting.")
		exit()
		
	Ncen = get_Ncen_More(M_h, survey)
		
	Nsat = np.zeros(len(M_h))
	for mi in range(0,len(M_h)):
		if ( M_h[mi]> ( pa.kappa_CMASS * pa.Mmin_CMASS ) ):
			Nsat[mi] = Ncen[mi] * ( ( M_h[mi] - pa.kappa_CMASS * pa.Mmin_CMASS) / pa.M1_CMASS)**pa.alpha_CMASS
		else:
			Nsat[mi] = 0.
				
	return Nsat

def get_Ncen_More(Mh, survey):
	""" Get central galaxy occupation number."""
	
	if (survey == 'SDSS'):
		import params as pa
	elif (survey == 'LSST_DESI'):
		import params_LSST_DESI as pa
	else:
		print("We don't have support for that survey yet; exiting.")
		exit()
	
	# This is for the More et al. 2014 CMASS HOD
	Ncen_CDF = np.zeros(len(Mh))
	finc = np.zeros(len(Mh))
	for mi in range(0,len(Mh)):
		finc[mi] = max(0, min(1., 1. + pa.alphainc_CMASS * (np.log10(Mh[mi]) - np.log10(pa.Minc_CMASS))))
		Ncen_CDF[mi] = finc[mi] * 0.5 * (1. + scipy.special.erf( (np.log10(Mh[mi]) - np.log10(pa.Mmin_CMASS)) / pa.siglogM_CMASS))
	
	return Ncen_CDF

def get_Ncen_Reid(Mh, survey):
    """ Get the cumulative distribution of central galaxies for the SDSS LRG sample from Reid & Spergel 2008. """
	
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()
	
    Ncen = 0.5 * (1. + scipy.special.erf((np.log10(Mh / (pa.HH0_l/100.)) - np.log10(pa.Mmin_reid)) / pa.sigLogM_reid))
	
    return Ncen 
	
def get_Nsat_Reid(Mh, survey):
    """ Get the number of satellite galaxies per halo. """
	
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()
	
    Ncen = get_Ncen_Reid(Mh, survey)
	
    Nsat = np.zeros(len(Mh))
    for i in range(0,len(Mh)):
        # This if-statement just sets annoying tiny numbers to 0.
        if ( (Mh[i] / (pa.HH0_l/100.)) >= pa.Mcut_reid):
            Nsat[i] = Ncen[i] * ((Mh[i] / (pa.HH0_l/100.) - pa.Mcut_reid) / pa.M1_reid)**(pa.alpha_reid)
        else:
            Nsat[i] = 0.
	
    return Nsat
	
	
def NcenNsat(alpha_sq, Ncen, Nsat):
	""" Returns the average number of pairs of central and satelite galaxies per halo of mass M. """
	
	NcNs = alpha_sq * Ncen * Nsat
	
	return NcNs
	
def alpha_sq(Mh):
	""" Returns alpha_sq from Scoccimarro et al 2000 """
	
	a_sq = np.zeros(len(Mh))
	for mi in range(0,len(Mh)):
		if Mh[mi]<10**(11):
			a_sq[mi] = np.log(np.sqrt(Mh[mi] / 10**11))**2
		else:
			a_sq[mi] = 1
	
	return a_sq
	
def NsatNsat(alpha_sq, Nsat_1, Nsat_2):
	""" Returns the average number of pairs of satelite galaxies per halo. """
	
	NsNs = alpha_sq * Nsat_1 * Nsat_2
	
	return NsNs
		
def wgg_2halo(rp_cents_, bd, bs, savefile, survey, sample):
    """ Returns wgg for the 2-halo term only."""
	
    if (survey == 'SDSS'):
        import params as pa
    elif (survey == 'LSST_DESI'):
        import params_LSST_DESI as pa
    elif (survey == 'DESY1'):
        import params_DESY1_testpz as pa
    else:
        print("We don't have support for that survey yet; exiting.")
        exit()

    print("Getting window")
    # Get the redshift window functions
    z_gg, win_gg = window(survey, sample)
    print("Got window")

    print("Getting matter power")
    # Get the required matter power spectrum from CCL
    # UPDATE CCL
    #p = ccl.Parameters(Omega_c = pa.OmC, Omega_b = pa.OmB, h = (pa.HH0/100.), sigma8= pa.sigma8, n_s=pa.n_s_cosmo)
    #cosmo = ccl.Cosmology(p)
    # 'true' cosmological parameters because this is a fiducial signal.
    cosmo_t = ccl.Cosmology(Omega_c = pa.OmC_t, Omega_b = pa.OmB, h = (pa.HH0_t/100.), sigma8 = pa.sigma8, n_s=pa.n_s)
    h = (pa.HH0_t/100.)
    k_gg = np.logspace(-5., 7., 100000)
    P_gg = np.zeros((len(z_gg), len(k_gg)))
    for zi in range(0,len(z_gg)):
        P_gg[zi, :] = h**3 * ccl.nonlin_matter_power(cosmo_t, k_gg * h , 1./(1.+z_gg[zi])) # CCL takes units without little h's, but we use little h unit
    print("Got matter power")
	
    print("Getting z intergrals")
    # First do the integral over z. Don't yet interpolate in k.
    zint_gg = np.zeros(len(k_gg))
    for ki in range(0,len(k_gg)):
        zint_gg[ki] = scipy.integrate.simps(win_gg * P_gg[:, ki], z_gg)
    print("Got z integrals")
		
    # Define vectors of kp (kperpendicual) and kz. Must have sufficiently high sampling to get the right answer, especially at large scales.
    kp_gg = np.logspace(np.log10(k_gg[0]), np.log10(k_gg[-1]/ np.sqrt(2.01)), pa.kpts_wgg)
    kz_gg = np.logspace(np.log10(k_gg[0]), np.log10(k_gg[-1]/ np.sqrt(2.01)), pa.kpts_wgg)
	
    # Interpolate in terms of kperp and kz
    kinterp_gg = scipy.interpolate.interp1d(k_gg, zint_gg)

    print("Interpolate in kperp and kz")
    # Get the result of the z integral in terms of kperp and kz
    kpkz_gg = np.zeros((len(kp_gg), len(kz_gg)))
    for kpi in range(0,len(kp_gg)):
        for kzi in range(0, len(kz_gg)):
            kpkz_gg[kpi, kzi] = kinterp_gg(np.sqrt(kp_gg[kpi]**2 + kz_gg[kzi]**2))
    print("done interpolating")
	
    print("getting kz integrals")			
    # Do the integrals in kz
    kz_int_gg = np.zeros(len(kp_gg))
    for kpi in range(0,len(kp_gg)):
        kz_int_gg[kpi] = scipy.integrate.simps(kpkz_gg[kpi,:] * kp_gg[kpi] / kz_gg * np.sin(kz_gg*pa.close_cut), kz_gg)
    print("got kz integrals")
    
    print("getting kperp integrals")
    # Do the integral in kperp
    kp_int_gg = np.zeros(len(rp_cents_))
    for rpi in range(0,len(rp_cents_)):
        kp_int_gg[rpi] = scipy.integrate.simps(scipy.special.j0(rp_cents_[rpi]* kp_gg) * kz_int_gg, kp_gg)
    print("got kperp integrals")
		
    wgg_2h = kp_int_gg * bs * bd / np.pi**2
    wgg_stack = np.column_stack((rp_cents_, wgg_2h))
    np.savetxt(savefile, wgg_stack)
	
    return wgg_2h

def wgg_full(rp_c, fsky, bd, bs, savefile_1h, savefile_2h, endfile, survey):
	""" Combine 1 and 2 halo terms of wgg """
	
	# Check if savefile_1h exists and if not compute the 1halo term.
	if (os.path.isfile(savefile_1h)):
		print("Loading wgg 1halo term from file.")
		(rp_cen, wgg_1h) = np.loadtxt(savefile_1h, unpack=True)	
	else:
		print("Computing wgg 1halo term.")
		wgg_1h = wgg_1halo_Four(rp_c, fsky,savefile_1h, endfile, survey)
		
	# Same for savefile_2h 
	if (os.path.isfile(savefile_2h)):
		print("Loading wgg 2halo term from file.")
		(rp_cen, wgg_2h) = np.loadtxt(savefile_2h, unpack=True)
	else:	
		print("Computing wgg 2halo term.")
		print(savefile_2h)
		wgg_2h = wgg_2halo(rp_c, bd, bs, savefile_2h,survey)
	
	wgg_tot = wgg_1h + wgg_2h 

	return wgg_tot
