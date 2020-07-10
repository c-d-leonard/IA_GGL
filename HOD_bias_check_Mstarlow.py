
# coding: utf-8

# Check the large scale galaxy bias as implied by the HOD models we use for each source and lens sample.

# In[1]:

# Import modules
get_ipython().magic(u'matplotlib inline')
import numpy as np; import scipy.integrate; import scipy.interpolate; import matplotlib.pyplot as plt
import pyccl as ccl; import shared_functions_wlp_wls as shared; import shared_functions_setup as setup
from halotools.empirical_models import PrebuiltHodModelFactory


# In[2]:

# Set the survey
survey = 'SDSS'
if (survey == 'SDSS'):
    import params as pa
elif (survey == 'LSST_DESI'):
    import params_LSST_DESI as pa
    
# Also set whether we are looking at lenses or sources
gals = 'src'


# In[3]:

#Initialize the cosmology
p = ccl.Parameters(Omega_c = pa.OmC_s, Omega_b = pa.OmB_s, h = (pa.HH0_s/100.), sigma8=pa.sigma8_s, n_s=pa.n_s_s)
cosmo = ccl.Cosmology(p)

rho_crit = 3. * 10**10 * pa.mperMpc / (8. * np.pi * pa.Gnewt * pa.Msun)  # Msol h^2 / Mpc^3, for use with M in Msol / h (comoving distances)
rho_m = (pa.OmC_s + pa.OmB_s) * rho_crit # units of Msol h^2 / Mpc^3 (comoving distances)


# In[4]:

# Get the appropriate redshift distribution
#if (gals=='lens'):
#    z = np.linspace(pa.zLmin, pa.zLmax, 100)
#    dNdz = setup.get_dNdzL(z, survey)
#if (gals =='src'):
#    z, dNdz_unnormed = setup.get_NofZ_unnormed(pa.dNdzpar_fid, pa.dNdztype, pa.zsmin, pa.zsmax, 500, survey)
#    norm = scipy.integrate.simps(dNdz_unnormed, z)
#    dNdz = dNdz_unnormed / norm
 
# Get the window function of sources x lenses (this is the redshift range we care about)
(z, dNdz) = shared.window(survey)    
  
# Get the halo mass function and halo bias
Mhvec = np.logspace(9.,16,30) # In units Msol / h
HMF = np.zeros((len(Mhvec), len(z)))
bh = np.zeros((len(Mhvec), len(z)))
for zi in range(0,len(z)):
    HMF[:,zi] = ccl.massfunction.massfunc(cosmo, Mhvec / (pa.HH0_s/100.), 1./ (1. + z[zi]), odelta=200.) / (pa.HH0_s/ 100.)**3
    bh[:,zi] = ccl.massfunction.halo_bias(cosmo, Mhvec / (pa.HH0_s/100.), 1./(1.+z[zi]), odelta=200.)

#HMF= ccl.massfunction.massfunc(cosmo, Mhvec / (pa.HH0/100.), 1./ (1. + z), odelta=200.) / (pa.HH0/100.)**3
#bh = ccl.massfunction.halo_bias(cosmo, Mhvec / (pa.HH0/100.), 1./(1.+z), odelta=200.)
    
# Integrate bh over z just for ploting 
bh_M = np.zeros(len(Mhvec))
for mi in range(0,len(Mhvec)):
    bh_M[mi] = scipy.integrate.simps(bh[mi, :] * dNdz, z)
    
plt.figure()
plt.loglog(Mhvec, bh_M)
plt.xlim(10**9,10**15)
plt.ylim(0.1, 10)
plt.xlabel('Halo mass, $M_\odot / h$')
plt.ylabel('$b_h$')
plt.title("Halo bias, SDSS src dNdz")
plt.show()
#plt.tight_layout()
#plt.savefig('./plots/halobias_SDSS_src.pdf')


# In[5]:

# Get y(k,M) (the fourier transformed profile)
logkmin = -6; kpts =40000; logkmax = 5; 
kvec_FT = np.logspace(logkmin, logkmax, kpts)
# Actually we will use a downsampled version of this:
k = np.logspace(np.log10(kvec_FT[0]), np.log10(kvec_FT[-1]), 40)
y = shared.gety(Mhvec, k, survey)


# In[6]:

# Get the linear matter power spectrum from CCL to multiply through
Pklin = np.zeros((len(k), len(z)))
for zi in range(0,len(z)):
    Pklin[:, zi] = ccl.power.linear_matter_power(cosmo, k, 1. / (1. + z[zi]))

# We now have all the ingredients we require to get the 2-halo matter power spectrum 
# We don't use this, I'm just checking we get something reasonable at this intermediate step
#twoh_fact = np.zeros((len(k), len(z)))
#for ki in range(0,len(k)):
#    for zi in range(0,len(z)):
#        twoh_fact[ki, zi] = scipy.integrate.simps( Mhvec / rho_m * HMF[:,zi] * bh[:, zi] * y[ki, :], np.log10(Mhvec / (pa.HH0_s / 100.)))    

#Pk_2h = Pklin * (twoh_fact)**2

# Integrate over z
#Pk_2h_avgz = np.zeros(len(k))
#Pklin_avgz = np.zeros(len(k))
#for ki in range(0,len(k)):
#    Pk_2h_avgz[ki] = scipy.integrate.simps(dNdz * Pk_2h[ki,:], z)
#    Pklin_avgz[ki] = scipy.integrate.simps(dNdz * Pklin[ki,:], z)

#plt.figure()
#plt.loglog(k, k**3 * Pk_2h_avgz / 2. / np.pi**2, 'b')
#plt.hold(True)
#plt.loglog(k, k**3 * Pklin_avgz  / 2. / np.pi**2, 'm')
#plt.xlim(0.05,30)
#plt.ylim(0.01, 100)
#plt.show()

# This isn't exactly 1 at large scales because we aren't integrating down to all the masses where halos exist.
# This shouldn't matter in the end for galaxy bias because those mass halos won't host galaxies.
# When we get the galaxy bias we will compare to halofit ie the same as Pklin_avgz on large scales.


# In[7]:

# Now, we want to convert this to a 2-halo galaxy power spectrum using the various HOD's we use.

if (gals=='src'):
#    # We need Mstarlow for the Zu & Mandelbaum halo model
    tot_nsrc = shared.vol_dens(pa.fsky, pa.N_shapes, survey)
    Mstarlow = shared.get_Mstar_low(survey, tot_nsrc)
    print "tot_nsrc=", tot_nsrc
    print "Mstarlow=", Mstarlow
    
# Get occupation numbers as a function of mass
if (survey == 'SDSS'):
    if (gals=='lens'):
        Ncen = shared.get_Ncen_Reid(Mhvec, survey)  # Reid & Spergel
        Nsat = shared.get_Nsat_Reid(Mhvec, survey)  # Reid & Spergel 
    elif (gals=='src'):
        # Let's use HaloTools and see what happens
        model = PrebuiltHodModelFactory('zu_mandelbaum15', threshold = np.log10(Mstarlow), prim_haloprop_key = 'halo_m200m')
        Nsat = model.mean_occupation_satellites(prim_haloprop=Mhvec)
        Ncen = model.mean_occupation_centrals(prim_haloprop=Mhvec)
elif (survey== 'LSST_DESI'):
    if (gals =='lens'):
        Ncen = shared.get_Ncen_More(Mhvec, survey) # CMASS
        Nsat = shared.get_Nsat_More(Mhvec, survey) # CMASS 
    elif(gals=='src'):
        Nsat = shared.get_Nsat_Zu(Mhvec, Mstarlow, 'tot', survey)  	# Zu & Mandelbaum 2015
        Ncen = shared.get_Ncen_Zu(Mhvec, Mstarlow, survey)  	# Zu & Mandelbaum 2015

# Combine to get the total occupation at mass M
N_tot= Ncen + Nsat

# Get satelite fraction integrated over mass
Nsat_int_ofz = np.zeros(len(z))
Ntot_int_ofz = np.zeros(len(z))
for zi in range(0,len(z)):
    Nsat_int_ofz[zi] = scipy.integrate.simps(Nsat * HMF[:,zi], np.log10(Mhvec / pa.HH0_s/100.))
    Ntot_int_ofz[zi] = scipy.integrate.simps(N_tot * HMF[:,zi], np.log10(Mhvec / pa.HH0_s/100.))
    
Nsat_int = scipy.integrate.simps(Nsat_int_ofz * dNdz, z)
Ntot_int = scipy.integrate.simps(Ntot_int_ofz * dNdz, z)
#Nsat_int = scipy.integrate.simps(Nsat * HMF, np.log10(Mhvec / (pa.HH0/100.)))
#Ntot_int= scipy.integrate.simps(N_tot * HMF, np.log10(Mhvec / (pa.HH0/100.)))
satfrac = Nsat_int / Ntot_int
print "sat frac=", satfrac

# Get the numerator of the halo bias of each population
bcen_of_z = np.zeros(len(z))
bsat_of_z = np.zeros(len(z))
btot_of_z = np.zeros(len(z))
for zi in range(0,len(z)):
    bcen_of_z[zi] = scipy.integrate.simps(bh[:, zi] * HMF[:,zi] * Ncen, np.log10(Mhvec / pa.HH0_s/100.))
    bsat_of_z[zi] = scipy.integrate.simps(bh[:, zi] * HMF[:,zi] * Nsat, np.log10(Mhvec / pa.HH0_s/100.))
    btot_of_z[zi] = scipy.integrate.simps(bh[:,zi] * HMF[:,zi] * N_tot, np.log10(Mhvec / pa.HH0_s/100.))

bcen_int = scipy.integrate.simps(bcen_of_z * dNdz, z)
bsat_int = scipy.integrate.simps(bsat_of_z * dNdz, z)
btot_int = scipy.integrate.simps(btot_of_z * dNdz, z)
                                
#bcen_int = scipy.integrate.simps(bh * HMF * Ncen, np.log10(Mhvec / (pa.HH0/100.)))
#bsat_int = scipy.integrate.simps(bh * HMF * Nsat, np.log10(Mhvec / (pa.HH0/100.)))    
#btot_int = scipy.integrate.simps(bh * HMF * N_tot, np.log10(Mhvec / (pa.HH0/100.)))
    
# Integrate over the halo mass function to get total number density 
nbar = np.zeros(len(z))
nbar_sat = np.zeros(len(z))
nbar_cen = np.zeros(len(z))
for zi in range(0,len(z)):
    nbar[zi] = scipy.integrate.simps(HMF[:,zi] * N_tot, np.log10(Mhvec / (pa.HH0_s/100.)))
    nbar_sat[zi]= scipy.integrate.simps(HMF[:,zi] * Nsat, np.log10(Mhvec / (pa.HH0_s/100.)))
    nbar_cen[zi]= scipy.integrate.simps(HMF[:,zi] * Ncen, np.log10(Mhvec / (pa.HH0_s/100.)))
                                
#nbar_int= scipy.integrate.simps(HMF * N_tot, np.log10(Mhvec / ((pa.HH0/100.))))
#nbar_sat_int= scipy.integrate.simps(HMF * Nsat, np.log10(Mhvec / ((pa.HH0/100.))))
#nbar_cen_int= scipy.integrate.simps(HMF * Ncen, np.log10(Mhvec / ((pa.HH0/100.))))                                
    
nbar_int = scipy.integrate.simps(nbar *dNdz, z)
nbar_cen_int = scipy.integrate.simps(nbar_cen * dNdz, z)
nbar_sat_int = scipy.integrate.simps(nbar_sat*dNdz, z)

print "halo bias, centrals=", bcen_int / nbar_cen_int
print "halo bias, satelites =", bsat_int / nbar_sat_int
print "halo bias, all =", btot_int / nbar_int
print "nbar int=", nbar_int

#plt.figure()
#plt.semilogx(Mhvec, Ncen, 'mo')
#plt.xlim(10**12,10**17)
#plt.title('$N_{\\rm cen}$, SDSS sources')
#plt.ylabel('$N_{\\rm cen}$')
#plt.xlabel('$M_h$, $M_\odot / h$')
#plt.ylim(10**(-3), 10**(3))
#plt.show()
#plt.tight_layout()
#plt.savefig('./plots/Ncen_SDSS_src.pdf')


# In[ ]:

# Get the galaxy-galaxy 2-halo term

twoh_gg = np.zeros((len(k), len(z)))
for ki in range(0,len(k)):
    for zi in range(0,len(z)):
        twoh_gg[ki,zi] = scipy.integrate.simps(HMF[:,zi] * bh[:,zi] * y[ki, :] * N_tot, np.log10(Mhvec/(pa.HH0_s/100))) / nbar[zi]
       
#twoh_gg = np.zeros(len(k))
#for ki in range(0,len(k)):
#    twoh_gg[ki] = scipy.integrate.simps(HMF * bh* y[ki, :] * N_tot, np.log10(Mhvec/(pa.HH0_s/100))) / nbar_int

P_2h_gg = np.zeros((len(k), len(z)))
for ki in range(0,len(k)):
    for zi in range(0,len(z)):
        P_2h_gg[ki, zi] = twoh_gg[ki, zi]**2 * Pklin[ki,zi]

# Integrate over z
P_2h_gg_avgz = np.zeros(len(k))
Pklin_avgz = np.zeros(len(k))
for ki in range(0,len(k)):
    P_2h_gg_avgz[ki] = scipy.integrate.simps(dNdz * P_2h_gg[ki,:], z)
    Pklin_avgz[ki] = scipy.integrate.simps(dNdz * Pklin[ki, :], z)

plt.figure()
plt.loglog(k,P_2h_gg_avgz,  'b')
plt.hold(True)
plt.loglog(k, Pklin_avgz, 'm')
plt.xlim(0.0001,30)
plt.ylim(0.01, 10**7)
plt.show()

np.sqrt(P_2h_gg_avgz / Pklin_avgz)


# In[ ]:

# Now use this to get the scale-dependent bias

