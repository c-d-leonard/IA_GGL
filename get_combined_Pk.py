""" Script to import a P(k) from CAMB, use the k values to get the 1-halo power spectrum, add CAMB P(k) to 1-halo P(k), and output combined P(k) """

import numpy as np
import scipy.integrate
import IA_params_Fisher as pa
import scipy.interpolate
import matplotlib.pyplot as plt

########## Functions ##########

def Delta_virial(z_):
	""" Get Delta_virial in terms of z, which is used to match the Mass / Radius relationship in Giocoli 2010. """
	
	# Follow Giocoli 2010 and simply read the points of Delta_virial as a function of Omega_tot (All omega except for OmegaLambda) off Figure 1 of Eke et al 2006
	OmTot = np.asarray([0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
	Dvir = np.asarray([107., 113., 126., 138., 148., 158., 169])
	
	# Get Omegatot (everything but Lambda) as a function of redshift from theory
	OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
	z = np.linspace(100000, 0., 1000000)
	Omtot_the = ( (pa.OmC+pa.OmB)*(1+z)**3 + (pa.OmR+pa.OmN) * (1+z)**4 ) / ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )
	
	# Interpolate
	z_of_Omtot = scipy.interpolate.interp1d(Omtot_the, z)
	
	# now, interpolate so that we can get Delta as a function of z:
	Dvir_of_z = scipy.interpolate.interp1d(z_of_Omtot(OmTot), Dvir)
	
	Dv = Dvir_of_z(z_)
	
	return Dv

def Rhalo(M_insol, z):
	""" Get the radius of a halo in Mpc/h given its mass IN SOLAR MASSES. Uses the Rvir / Mvir relationship from Giocolo 2010, which employs Delta_virial, from Eke et al 1996."""
	
	OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
	E_ofz = ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )**(0.5)#the dimensionless part of the hubble parameter 
	rho_crit = 4.126 * 10** 11 * E_ofz**2 # This is rho_crit in units of Msol h^3 / Mpc^3
	OmM = (pa.OmC+pa.OmB)*(1+z)**3 / ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )
	#Dv = Delta_virial(z)
	
	#Rvir = ( 3. * M_insol * OmM / (4. * np.pi * rho_crit * Dv))**(1./3.)
	Rvir = ( 3. * M_insol / (4. * np.pi * rho_crit * OmM * 180.))**(1./3.)
	
	return Rvir

def cvir(M_insol, z):
	""" Returns the concentration parameter of the NFW profile, c_{vir}. Uses the Neto 2007 definition, referenced in Giocoli 2010 """
	
	cvi = pa.c14 / (1. + z) * (M_insol / 10**14)**(-0.11)
	
	#cvi = 7.
	
	return cvi

def rho_s(cvi, Rvi, M_insol):
	""" Returns rho_s, the NFW parameter representing the density at the `scale radius', Rvir / cvir. Units: Msol * ( 1 / (Rvir units)**3), usualy Msol * h^3 / Mpc^3. """
	
	rhos = M_insol / (4. * np.pi) * ( cvi / Rvi)**3 * (np.log(1. + cvi) - (cvi / (1. + cvi)))**(-1)
	
	return rhos

def rho_NFW(r_, M_insol, z):
	""" Returns the density for an NFW profile in real space at distance r from the center (?). Units = units of rhos. (Usually Msol * h^3 / Mpc^3). r_ MUST be in the same units as Rv; usually Mpc / h."""
	
	Rv = Rhalo(M_insol, z)
	cv = cvir(M_insol, z)
	rhos = rho_s(cv, Rv, M_insol)
	
	rho_nfw = rhos  / ( (cv * r_ / Rv) * (1. + cv * r_ / Rv)**2)
	
	plt.figure()
	plt.loglog(r_ , rho_nfw *  (pa.HH0/100.) )
	plt.ylim(10**14, 2*10**18)
	plt.xlim(0.001, 3)
	plt.savefig('./plots/nfw_0605_6e13h.png')
	plt.close()
	

	
	return rho_nfw

def get_u(M_insol, z, kvec):
	""" Fourier transforms the density profile to get the power spectrum. """
	
	Rv = Rhalo(M_insol, z)
	cv = cvir(M_insol, z)
	
	# Get the nfw density profile at the correct mass and redshift and at a variety of r
	rvec = np.logspace(-10, np.log10(Rv), 10000) # Min r = 10^{-7} is sufficient for convergence. 4000 pts is more than sufficient.
	rho = rho_NFW(rvec, M_insol, z)
	
	u_ = np.zeros(len(kvec))
	for ki in range(0,len(kvec)):
		u_[ki] = 4. * np.pi / M_insol * scipy.integrate.simps( rvec**2 * np.sin(kvec[ki]*rvec)/ (kvec[ki]*rvec) * rho, rvec)
		
	plt.figure()
	plt.loglog(kvec, u_, 'b+')
	plt.ylim(0.01, 2)
	plt.xlim(0.002, 100000)
	plt.savefig('./plots/u_6e13h.png')
	plt.close()
	
	return u_
	
def get_Sigma(M_insol, z):
	""" Gets Sigma(R) by projecting rho for testing. """
	print "Enterring get_Sigma"
	rvec = np.logspace(-4, 100., 10000)
	rho = rho_NFW(rvec, M_insol, z)
	
	Rp = np.logspace(-4, np.log(100. / np.sqrt(2.)), 1000)
	Pi = np.logspace(-4, np.log(100./ np.sqrt(2.)), 1000)
	
	rho_interp = scipy.interpolate.interp1d(rvec, rho)
	
	print "rho 0.001=", rho_interp(0.001)
	print "Interpolating rho"
	rho_2D = np.zeros((len(Rp), len(Pi)))
	for ri in range(0, len(Rp)):
		for pi in range(0, len(Pi)):
			rho_2D[ri, pi] = rho_interp(np.sqrt(Rp[ri]**2 + Pi[pi]**2))
	
	print "Projecting rho"
	rho_proj = np.zeros(len(Rp))
	for ri in range(0,len(Rp)):
		rho_proj[ri] = scipy.integrate.simps(rho_2D[ri, :], Pi)
		
	# rho_proj has units Msol * h ^2 / Mpc^2. We want to plot in Msol h  / pc^2. 1 Mpc = 10^6 pc.
	Sig_plt = rho_proj * (pa.HH0/ 100.) / 10.**12
	
	#Sig_interp = scipy.interpolate.interp1d(Rp , Sig_plt)
	#print "Sig, 0.001=", Sig_interp(0.001)
		
	plt.figure()
	plt.loglog(Rp, Sig_plt)
	plt.xlim(0.0003, 8.)
	plt.ylim(0.3 , 20000)
	plt.savefig('./plots/Sigma_from_rho_6e13h.png')
	plt.close()
	
	return

def P1halo_DM(M_insol, z, k):
	""" Gets the 1 halo dark matter power spectrum as a function of z and k. """
	
	P1h = np.zeros(len(k))
	# Get u
	u = get_u(M_insol, z, k)
	
	# Need the average matter density  = rho_crit * OmegaM
	OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
	E_ofz = ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )**(0.5)#the dimensionless part of the hubble parameter 
	rho_crit = 4.126 * 10** 11 * E_ofz**2 # This is rho_crit in units of Msol h^3 / Mpc^3
	OmM = (pa.OmC + pa.OmB)* (1. + z)**(3) / ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )
	rho_m = OmM * rho_crit # units of Msol h^3 / Mpc^3
	
	#P1h = M_insol / rho_m * u**2 
	P1h = (M_insol / rho_m)**2 * 3.0*10**(-4) * u**2
	
	return P1h

############### Main script ##################

# Import CAMB nonlinear halofit power spectrum

get_Sigma(pa.Mvir, 0.3)

(k, P_HF) = np.loadtxt('./txtfiles/NL_kmax4000_z=0.32.dat', unpack=True)

P_1h = P1halo_DM(pa.Mvir, 0.32, k)

P_both = np.column_stack((k, P_1h + P_HF))

P_1h_stack = np.column_stack((k, P_1h))

np.savetxt('./txtfiles/1halo_only_Pk_z=0.32_M6e13h_fixRhoC.dat', P_1h_stack)

np.savetxt('./txtfiles/1halo_and_halofit_Pk_z=0.32_M6e13h_fixRhoC.dat', P_both)

plt.figure()
plt.loglog(k, P_HF, 'b+')
plt.hold(True)
plt.loglog(k, P_1h, 'm+')
plt.hold(True)
plt.loglog(k, P_HF + P_1h, 'k+')
plt.ylim(0.001, 50000)
plt.xlim(0.01, 100)
plt.savefig('./plots/P_1halo_Blazek_M6e13h_fixRhoC.png')
plt.close()

plt.figure()
plt.loglog(k, k**3* P_HF / (2. *np.pi**2), 'b+')
plt.hold(True)
plt.loglog(k, k**3* P_1h / (2. * np.pi**2), 'm+')
plt.hold(True)
plt.loglog(k, k**3*(P_HF + P_1h) / (2.*np.pi**2), 'k+')
plt.ylim(0.001, 50000)
plt.xlim(0.01, 100)
plt.savefig('./plots/Deltak_1h_M6e13h_fixRhoC.png')
plt.close()









