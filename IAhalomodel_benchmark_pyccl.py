# This is script is intended to provide code to benchmark the integration over the halo mass function and HOD quantities for the IA halo model in CCL

import numpy as np
import scipy.integrate
import pyccl as ccl
import matplotlib.pyplot as plt

def get_Pkgm_1halo(kvec, Mhalo, y, cosmo, z):
    """ Returns the 1halo satellite lens galaxy position x intrinsic shape power spectrum.
    Assume that the alignment of the satellite galaxies have radially independent alignment with the amplitude set to 1 (just for comparison).s
    kvec is the vector of wavenumbers to use in units of h/Mpc
    Mhalo is the vector of halo masses over which to integrate in units of Msol / h
    y is the Fourier-space NFW profile assuming a lens galaxies is at the center
    """
	
    # Get the halo mass function
    h = cosmo.__getitem__('h')
    omc = cosmo.__getitem__('Omega_c')
    omb = cosmo.__getitem__('Omega_b')
	

    HMF_setup = ccl.halos.MassFuncTinker10(cosmo)
    HMF = HMF_setup.get_mass_function(cosmo, Mhalo / h, 1./(1.+z))
    
    #save_HMF = np.column_stack((Mhalo/h, HMF))
    #np.savetxt('./HMF.data', save_HMF)

    HODHProf = ccl.halos.HaloProfileHOD(conc)	
    # Get HOD quantities we need
    Ncen_lens = HODHProf._Nc(Mhalo, 1./(1.+z))
    Nsat_lens = HODHProf._Ns(Mhalo, 1./(1.+z))

		
    # Get total number of satellite galaxies
    tot_ns = scipy.integrate.simps( Nsat_lens * HMF, np.log10(Mhalo / h ) ) / h**3
    tot_all = scipy.integrate.simps( (Ncen_lens+Nsat_lens) * HMF, np.log10(Mhalo / h ) ) / h**3
    
    # Fraction of satellites:
    f_s = tot_ns / tot_all
	
    # Get the density of matter in comoving coordinates
    rho_crit = 3. * 10**10 * mperMpc / (8. * np.pi * Gnewt * Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h (comoving distances)
    rho_m = (omc + omb) * rho_crit # units of Msol h^2 / Mpc^3 (comoving distances)
    
    # Get Pk
    Pkgm = np.zeros(len(kvec))
    for ki in range(0,len(kvec)):
        Pkgm[ki] = scipy.integrate.simps( HMF * (Mhalo / rho_m) * f_s * (Nsat_lens * y[ki, :]**2), np.log10(Mhalo / h)) / (tot_ns) # output in Mpc^3
		
    return (kvec * h, Pkgm) # Ouptut in 1/Mpc not h/Mpc
	
def gety_ldm(Mvec, kvec,cosmo, z):
    """ Fourier transforms the density profile to get the power spectrum. """
	
    # Get the nfw density profile at the correct mass and redshift and at a variety of r
    rvec = [0]*len(Mvec)
    rho = [0]*len(Mvec)
    for Mi in range(0,len(Mvec)): 
        Rvir = Rhalo(Mvec[Mi],cosmo)
        rvec[Mi] = np.logspace(-8, np.log10(Rvir), 10**6)
        rho[Mi] = rho_NFW_ldm(rvec[Mi], Mvec[Mi],cosmo, z)  # Units Msol h^2 / Mpc^3, comoving. 

    u_ = np.zeros((len(kvec), len(Mvec)))
    for ki in range(0,len(kvec)):
        for mi in range(0,len(Mvec)):
            u_[ki, mi] = 4. * np.pi / Mvec[mi] * scipy.integrate.simps( rvec[mi] * np.sin(kvec[ki]*rvec[mi])/ kvec[ki] * rho[mi], rvec[mi]) # unitless / dimensionless.
	
    return u_
	
def rho_NFW_ldm(r_, M_insol,cosmo, z):
    """ Returns the density for an NFW profile in real space at distance r from the center. Units = units of rhos. (Usually Msol * h^2 / Mpc^3 in comoving distances). r_ MUST be in the same units as Rv; usually Mpc / h."""

    Rv = Rhalo(M_insol,cosmo)
    cv = conc.get_concentration(cosmo, M_insol, 1./(1.+z))
    rhos = rho_s(cv, Rv, M_insol)

    rho_nfw = rhos  / ( (cv * r_ / Rv) * (1. + cv * r_ / Rv)**2) 
    
    # Truncate at Rvir:
    for i in range(0,len(r_)):
        if r_[i]>Rv:
            rho_nfw[i]=0
	
    return rho_nfw
	
def Rhalo(M_insol, cosmo):
    """ Get the radius of a halo in COMOVING Mpc/h given its mass."""
    """ Note I'm fixing to source cosmological parameters here."""
    
    omc = cosmo.__getitem__('Omega_c')
    omb = cosmo.__getitem__('Omega_b')
	
    rho_crit = 3. * 10**10 * mperMpc / (8. * np.pi * Gnewt * Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h
    rho_m = rho_crit * (omc + omb)
    Rvir = ( 3. * M_insol / (4. * np.pi * rho_m * 200.))**(1./3.) # We use the 200 * rho_M overdensity definition. 
	
    return Rvir
	
	
def rho_s(cvi, Rvi, M_insol):
    """ Returns rho_s, the NFW parameter representing the density at the `scale radius', Rvir / cvir. Units: Mvir units * ( 1 / (Rvir units)**3), usualy Msol * h^3 / Mpc^3 with comoving distances. Sometimes also Msol h^2 / Mpc^3 (when Mvir is in Msol / h). """
	
    rhos = M_insol / (4. * np.pi) * ( cvi / Rvi)**3 * (np.log(1. + cvi) - (cvi / (1. + cvi)))**(-1)
	
    return rhos
   
	
if (__name__ == "__main__"):

    logkmin = -4; kpts =100; logkmax = 2; Mmax = 15;
    kvec = np.logspace(logkmin, logkmax, kpts)
    Mhalo = np.logspace(10., Mmax, 30)
    
    z = 0.0
    
    cosmo = ccl.CosmologyVanillaLCDM()
    
    # Define some constants:
    mperMpc = 3.0856776*10**22
    Msun = 1.989*10**30 # in kg
    Gnewt = 6.67408*10**(-11)
    c=2.99792458*10**(8)
    
    conc = ccl.halos.ConcentrationDuffy08()

    y = gety_ldm(Mhalo, kvec, cosmo, z)

    np.savetxt('./y_ldm.txt', y)
    #y = np.loadtxt('./y_ldm.txt')
    
    (k, Pkgm) = get_Pkgm_1halo(kvec, Mhalo, y, cosmo, z)
    
    savePk = np.column_stack((k, Pkgm))
    np.savetxt('./Pkgm_1h_test.dat', savePk)
    
    plt.figure()
    plt.loglog(k, Pkgm)
    plt.savefig('./see_Pkgm.pdf')
    plt.close()
    
    

	

