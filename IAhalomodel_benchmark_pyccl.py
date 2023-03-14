# This is script is intended to provide code to benchmark the integration over the halo mass function and HOD quantities for the IA halo model in CCL

import numpy as np
import scipy.integrate
import pyccl as ccl
import matplotlib.pyplot as plt

def get_Pkgm_1halo(kvec, Mhalo, y, cosmo, z):
    """ Returns the 1halo lens galaxy position x intrinsic shape power spectrum
    kvec is the vector of wavenumbers to use in units of h/Mpc
    Mhalo is the vector of halo masses over which to integrate in units of Msol / h
    y is the Fourier-space NFW profile assuming a lens galaxies is at the center
    """
	
    # Get the halo mass function
    h = cosmo.__getitem__('h')
    omc = cosmo.__getitem__('Omega_c')
    omb = cosmo.__getitem__('Omega_b')
	
    #HMF = np.zeros((len(Mhalo), len(zLvec)))
    #for zi in range(0, len(zLvec)):
    #    HMF[:, zi]= ccl.massfunction.massfunc( cosmo, Mhalo / h, 1./ (1. + zLvec[zi]))
    HMF_setup = ccl.halos.MassFuncTinker10(cosmo)
    HMF = HMF_setup.get_mass_function(cosmo, Mhalo / h, 1./(1.+z))
	
    # Get HOD quantities we need
    Ncen_lens = get_Ncen_More(Mhalo)
    Nsat_lens = get_Nsat_More(Mhalo)
		
    # Check total number of galaxies:
    tot_ng = scipy.integrate.simps( ( Ncen_lens + Nsat_lens) * HMF, np.log10(Mhalo / h ) ) / h**3
	
    # Get the density of matter in comoving coordinates
    rho_crit = 3. * 10**10 * mperMpc / (8. * np.pi * Gnewt * Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h (comoving distances)
    rho_m = (omc + omb) * rho_crit # units of Msol h^2 / Mpc^3 (comoving distances)
	
    # Get Pk
    Pkgm = np.zeros(len(kvec))
    for ki in range(0,len(kvec)):
        Pkgm[ki] = scipy.integrate.simps( HMF * (Mhalo / rho_m) * (Ncen_lens * y[ki, :] + Nsat_lens * y[ki, :]**2), np.log10(Mhalo / h)) / (tot_ng) / h**3
		
    return (kvec, Pkgm)
	
def gety_ldm(Mvec, kvec):
    """ Fourier transforms the density profile to get the power spectrum. """
	
    # Get the nfw density profile at the correct mass and redshift and at a variety of r
    rvec = [0]*len(Mvec)
    rho = [0]*len(Mvec)
    for Mi in range(0,len(Mvec)): 
        Rvir = Rhalo(Mvec[Mi])
        rvec[Mi] = np.logspace(-8, np.log10(Rvir), 10**6)
        rho[Mi] = rho_NFW_ldm(rvec[Mi], Mvec[Mi])  # Units Msol h^2 / Mpc^3, comoving. 

    u_ = np.zeros((len(kvec), len(Mvec)))
    for ki in range(0,len(kvec)):
        for mi in range(0,len(Mvec)):
            u_[ki, mi] = 4. * np.pi / Mvec[mi] * scipy.integrate.simps( rvec[mi] * np.sin(kvec[ki]*rvec[mi])/ kvec[ki] * rho[mi], rvec[mi]) # unitless / dimensionless.
	
    return u_
	
def rho_NFW_ldm(r_, M_insol):
    """ Returns the density for an NFW profile in real space at distance r from the center. Units = units of rhos. (Usually Msol * h^2 / Mpc^3 in comoving distances). r_ MUST be in the same units as Rv; usually Mpc / h."""

    Rv = Rhalo(M_insol)
    cv = cvir_ldm(M_insol)
    rhos = rho_s(cv, Rv, M_insol)
	
    rho_nfw = rhos  / ( (cv * r_ / Rv) * (1. + cv * r_ / Rv)**2) 
	
    return rho_nfw
	
def Rhalo(M_insol):
    """ Get the radius of a halo in COMOVING Mpc/h given its mass."""
    """ Note I'm fixing to source cosmological parameters here."""

    import params_LSST_DESI as pa
	
    rho_crit = 3. * 10**10 * mperMpc / (8. * pi * Gnewt * Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h
    rho_m = rho_crit * (pa.OmC_s + pa.OmB_s)
    Rvir = ( 3. * M_insol / (4. * np.pi * rho_m * 200.))**(1./3.) # We use the 200 * rho_M overdensity definition. 
	
    return Rvir
	
def cvir_ldm(M_insol):
    """ Returns the concentration parameter of the NFW profile, c_{vir}. """

    cvi = 5. * (M_insol / 10**14)**(-0.1)
	
    return cvi
	
def rho_s(cvi, Rvi, M_insol):
    """ Returns rho_s, the NFW parameter representing the density at the `scale radius', Rvir / cvir. Units: Mvir units * ( 1 / (Rvir units)**3), usualy Msol * h^3 / Mpc^3 with comoving distances. Sometimes also Msol h^2 / Mpc^3 (when Mvir is in Msol / h). """
	
    rhos = M_insol / (4. * np.pi) * ( cvi / Rvi)**3 * (np.log(1. + cvi) - (cvi / (1. + cvi)))**(-1)
	
    return rhos
    
def get_Ncen_More(Mh):
    """ Get central galaxy occupation number."""
    
    import params_LSST_DESI as pa
    
    # This is for the More et al. 2014 CMASS HOD
    Ncen_CDF = np.zeros(len(Mh))
    finc = np.zeros(len(Mh))
    for mi in range(0,len(Mh)):
        finc[mi] = max(0, min(1., 1. + pa.alphainc_CMASS * (np.log10(Mh[mi]) - np.log10(pa.Minc_CMASS))))
        Ncen_CDF[mi] = finc[mi] * 0.5 * (1. + scipy.special.erf( (np.log10(Mh[mi]) - np.log10(pa.Mmin_CMASS)) / pa.siglogM_CMASS))
	
    return Ncen_CDF
	
def get_Nsat_More(M_h):
    """ Gets source galaxies in satelite halos from More et al. 2014 HOD"""
	
    import params_LSST_DESI as pa
		
    Ncen = get_Ncen_More(M_h)
		
    Nsat = np.zeros(len(M_h))
    for mi in range(0,len(M_h)):
        if ( M_h[mi]> ( pa.kappa_CMASS * pa.Mmin_CMASS ) ):
            Nsat[mi] = Ncen[mi] * ( ( M_h[mi] - pa.kappa_CMASS * pa.Mmin_CMASS) / pa.M1_CMASS)**pa.alpha_CMASS
        else:
            Nsat[mi] = 0.
				
    return Nsat
	
if (__name__ == "__main__"):

    logkmin = -4; kpts =100; logkmax = 2; Mmax = 15;
    kvec = np.logspace(logkmin, logkmax, kpts)
    Mhalo = np.logspace(10., Mmax, 30)
    
    z = 0.8
    
    cosmo = ccl.CosmologyVanillaLCDM()
    
    # Define some constants:
    mperMpc = 3.0856776*10**22
    Msun = 1.989*10**30 # in kg
    Gnewt = 6.67408*10**(-11)
    c=2.99792458*10**(8)

    y = gety_ldm(Mhalo, kvec)
    
    np.savetxt('./y_ldm.txt', y)
    y = np.loadtxt('./y_ldm.txt')
    
    (k, Pkgm) = get_Pkgm_1halo(kvec, Mhalo, y, cosmo, z)
    
    savePk = np.column_stack((k, Pkgm))
    np.savetxt('./Pkgm_1h_test.dat', savePk)
    
    plt.figure()
    plt.loglog(k, Pkgm)
    plt.savefig('./see_Pkgm.pdf')
    plt.close()
    
    

	

