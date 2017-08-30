""" This script computes the covariance matrix of Upsilon_{gm} in bins in R.
This version assumes an effective redshift for the lenses, parameterized by comoving distance chiLmean."""

import numpy as np
import scipy
import scipy.interpolate
import scipy.integrate
import subprocess
import shutil
import pylab
import time
import matplotlib.pyplot as plt

##########################################################################################
######################################## FUNCTIONS #######################################
##########################################################################################

def setup_vectors():
	""" This function sets up all the vectors of points we will need """
	
	chiSvec		=		scipy.linspace(chiSmin, chiSmax, chiSpts)
	chiLext		=		scipy.linspace(chiLext_min, chiLext_max, chiLextpts)
	lvec		=		scipy.logspace(np.log10(lmin), np.log10(lmax), lpts)
	Rvec		=		scipy.logspace(np.log10(Rmin), np.log10(Rmax), Rpts)
	Redges		=		scipy.logspace(np.log10(Rmin), np.log10(Rmax), numRbins+1)
	
	# Want to get the centres of the bins as well so we know where to plot. But, we need to get the centres in log space.
	logRedges=np.log10(Redges)
	Rcentres=np.zeros(numRbins)
	for ri in range(0,numRbins):
		Rcentres[ri]	=	10**((logRedges[ri+1] - logRedges[ri])/2. +logRedges[ri])
		
	np.savetxt('./txtfiles/Rcenters.txt', Rcentres)
	
	return (chiSvec, lvec, Rvec, Redges, chiLext, Rcentres)

def getwindowfunctions():
	""" This function computes the (unnormalised) redshift distribution of sources as an interpolating function in comoving distances.  """

	zvec 		=	 z_ofchi(chiSvec)

	# Get the index of the z which is closest to the effective lens redshift.
	indexzval	= 	next(j[0] for j in enumerate(zvec) if j[1]>zval)

	# Construct dndz.
	nofz = (zvec) ** alpha * np.exp( - (zvec / z0) ** beta )
	
	# Get the norm:	
	if (zval>minz):
		normWs = scipy.integrate.simps(nofz[indexzval:], zvec[indexzval:])
	else:
		normWs = scipy.integrate.simps(nofz, zvec)
		
	# Check how many source galaxies are ahead of the effective zval of the lenses:
	norm_zeff = scipy.integrate.simps(nofz[indexzval:], zvec[indexzval:])
	norm_total = scipy.integrate.simps(nofz, zvec)
	print "fractions of sources above z_eff_lens=", norm_zeff / norm_total
	
	# We also need the factor H(z)/ c = dz/d_z.	
	H = H0 * np.sqrt( (OmegaM+OmegaB) * (1+zvec)**3 + OmegaL + (OmegaR+OmegaN) * (1+ zvec)**4)

	if (chiLmean>minchival):
		normchi = scipy.integrate.simps(nofz[indexzval:]* H[indexzval:] / normWs, chiSvec[indexzval:])  
		print "check norm chi, chiL mean=", normchi
	else:
		normchi = scipy.integrate.simps(nofz* H / normWs, chiSvec)
		print "check norm chi=", normchi

	return nofz * H / normWs

def getwindowfunctions_Nakajima():
	""" This function computes the (unnormalised) redshift distribution of sources as an interpolating function in comoving distances, for the functional form of Nakajima 2011.  """

	zvec 		=	 z_ofchi(chiSvec)

	# Get the index of the z which is closest to the effective lens redshift.
	indexzval	= 	next(j[0] for j in enumerate(zvec) if j[1]>zval)

	# Construct dndz.
	print "WARNING: Using the Nakajima et al 2011 form for dNdz, not Smail et al."
	nofz = (zvec / zs) ** (alpha_-1.) * np.exp( - 0.5 * (zvec / zs) ** 2. )
	
	# Get the norm:	
	if (zval>minz):
		normWs = scipy.integrate.simps(nofz[indexzval:], zvec[indexzval:])
	else:
		normWs = scipy.integrate.simps(nofz, zvec)
		
	# Check how many source galaxies are ahead of the effective zval of the lenses:
	norm_zeff = scipy.integrate.simps(nofz[indexzval:], zvec[indexzval:])
	norm_total = scipy.integrate.simps(nofz, zvec)
	print "fractions of sources above z_eff_lens=", norm_zeff / norm_total
	
	# We also need the factor H(z)/ c = dz/d_z.	
	H = H0 * np.sqrt( (OmegaM+OmegaB) * (1+zvec)**3 + OmegaL + (OmegaR+OmegaN) * (1+ zvec)**4)

	if (chiLmean>minchival):
		normchi = scipy.integrate.simps(nofz[indexzval:]* H[indexzval:] / normWs, chiSvec[indexzval:])  
		print "check norm chi, chiL mean=", normchi
	else:
		normchi = scipy.integrate.simps(nofz* H / normWs, chiSvec)
		print "check norm chi=", normchi

	return nofz * H / normWs
	
####################### FUNCTIONS FOR GETTING POWER SPECTRA ##############################
	
def getzLvec():
	"""Constructs an interpolating function which gives z as a function of chi"""
    
	#Define a vector of z values that will encompass the required chi values (this is hard coded)
	zpts=10000
	zvec=scipy.linspace(0.0, 30., zpts) # This assumes we don't care about anything above z=30.

	#Define an integral which yields conformal distance
	def intchi(zpt):
		return 1./( ( OmegaM + OmegaB ) * ( 1. + zpt ) ** 3 + OmegaL  + (OmegaR+OmegaN) * (1. + zpt) ** 4 ) ** (0.5)
     
	chians=np.zeros(zpts)
	for zi in range(0,zpts):
		chians[zi]=-scipy.integrate.quad(intchi,zvec[zi],0.0)[0]/H0
        
	#Now interpolate z wrt chi
	z_ofchi=scipy.interpolate.interp1d(chians, zvec)

	return z_ofchi

def getHconf(xivec):
    """Returns the conformal Hubble constant as a function of zLvec."""
    
    #Get zLvec to correspond with chiLvec
    zLvec	=	z_ofchi(xivec)
    Hconf	=	H0 * ( (OmegaM+OmegaB)*(1+zLvec) + OmegaL / (1+zLvec)**2 + (OmegaR+OmegaN) * (1+zLvec)**2 )**(0.5)
    
    return Hconf
    
def getOmMx(xivec):
	"""Returns OmM(x) where OmM(x)=OmB(x)+OmC(x)"""
	#Get zLvec to correspond with chiLvec
	zLvec	=	z_ofchi(xivec)

	OmMx= ( OmegaM + OmegaB ) * (1+zLvec)**3 / ((OmegaM+OmegaB)*(1+zLvec)**3 + OmegaL + (OmegaN+OmegaR)*(1+zLvec)**4)
    
	return OmMx

def PofkGR(xivec):
	""" Returns the nonlinear matter power spectrum today as a 2 parameter function of l and x. Requires pre-computation from CAMB at the values of xivec. """
    
	#Load the matter power spectrum at each z we want (this has to be pre-computed, see get_Pk_NL.py in camb folder.
	zivec = z_ofchi(xivec)
	
	# Load one to get the length in k:
	(k, P) = np.loadtxt(cambfolderpath+'/Ygm_matterpower_z='+'{0:.8e}'.format(zivec[0])+'.dat',unpack=True)
	Pofkz = np.zeros((len(k),len(zivec))) 
	Pofkint = [0] * len(zivec)
	for zi in range(0,len(zivec)):
		k, Pofkz[:, zi] = np.loadtxt(cambfolderpath+'/Ygm_matterpower_z='+'{0:.8e}'.format(zivec[zi])+'.dat', unpack=True)
		Pofkint[zi]=scipy.interpolate.interp1d(k, Pofkz[:,zi])

	#Define vector to hold answer
	Poflandx=np.zeros((len(lvec),len(xivec)))
	for li in range(0,len(lvec)):
		for xi in range(0,len(xivec)):
			if (lvec[li]/xivec[xi]<k[-1] and lvec[li]/xivec[xi]>k[0]):
				Poflandx[li,xi]=Pofkint[xi](lvec[li]/xivec[xi])
				if (np.abs(Poflandx[li,xi])<10**(-15)): 
					Poflandx[li,xi]=0.0
			else:
				Poflandx[li,xi]=0.0

	return Poflandx
	
def PofkGR_chimean(xi):
	""" Returns the nonlinear (halofit) matter power spectrum today as a function of l at the comoving distance OF THE LENSES"""
	
	#Load the nonlinear matter power at zval and interpolate
	kvec, PofK=np.loadtxt(cambfolderpath+Fileroot+str(zval)+'.dat', unpack=True)
	Pofkint=scipy.interpolate.interp1d(kvec, PofK)

	#Define vector to hold answer
	Poflandx=np.zeros(len(lvec))

	for li in range(0,len(lvec)):
		if (lvec[li]/xi<kvec[-1] and lvec[li]/xi>kvec[0]):
			Poflandx[li]=Pofkint(lvec[li]/xi)
			if (np.abs(Poflandx[li]))<10**(-15): 
				Poflandx[li]=0.0
		else:
			Poflandx[li]=0.0

	return Poflandx

def growthfac_chimean(xi):
	""" Returns the growth factor D as a function of chi """
	#There are some factors of H0 floating around differently than expected to avoid numerical error - they all show up in the right place at the end.
    	# The integral in this function is analytic so we can use scipy.integrate.quad. 

	#Get zLvec
	zLvec	=	z_ofchi(xi)
	xv	=	np.log(1. / (1. + zLvec))
    
	xinf=np.log(1./(1.+100.))

	#Define the main integral
	def intD(x):
		Hofx=((OmegaM+OmegaB)*np.exp(-x)+OmegaL*np.exp(2.*x) + (OmegaR + OmegaN) * np.exp(-2.*x) )**(0.5) 
		return np.exp(x)/(Hofx**3)

	#Do the integral once to get the normalisation factor:
	norm=(scipy.integrate.quad(intD, xinf, 0.0, limit=500)[0])**(-1)

	ansD=scipy.integrate.quad(intD,xinf, xv, limit=500)[0]
	Dhold=norm*((OmegaM+OmegaB)*np.exp(-xv)+OmegaL*np.exp(2.*xv)+(OmegaR+OmegaN)*np.exp(-2.*xv))**(0.5)/(np.exp(xv))*ansD

	return Dhold
    
def growthfac(xivec):
	""" Returns the growth factor D as a function of chi """
	#There are some factors of H0 floating around differently than expected to avoid numerical error - rest assured they all show up in the right place at the end.
	# The integral here is analytic so we can use quad
    
	#Get zLvec
	zLvec	=	z_ofchi(xivec)
	xvec	=	np.log(1. / (1. + zLvec))
    
	xinf=np.log(1./(1.+100.))

	#Define the main integral
	def intD(x):
		Hofx=((OmegaM+OmegaB)*np.exp(-x)+OmegaL*np.exp(2.*x) + (OmegaR + OmegaN) * np.exp(-2.*x) )**(0.5) 
		return np.exp(x)/(Hofx**3)

	#Do the integral once to get the normalisation factor:
	norm=(scipy.integrate.quad(intD, xinf, 0.0, limit=500)[0])**(-1)

	#Vector to hold answer
	Dhold=np.zeros(len(xvec))

	for xi in range(0,len(xvec)):
		ansD=scipy.integrate.quad(intD,xinf, xvec[xi], limit=500)[0]
		Dhold[xi]=norm*((OmegaM+OmegaB)*np.exp(-xvec[xi])+OmegaL*np.exp(2.*xvec[xi])+(OmegaR+OmegaN)*np.exp(-2.*xvec[xi]))**(0.5)/(np.exp(xvec[xi]))*ansD

	return Dhold	
	
def get_Pgg():
	""" This function computes P_{gg}(l, chiLmean) """
	
	#Get the z value corresponding to our chiLmean:
	zL	=	z_ofchi(chiLmean)
	
	#Load the nonlinear matter power (from CAMB) and interpolate
	kvec_temp, PofK_temp=np.loadtxt(cambfolderpath+Fileroot+str(zval)+'.dat', unpack=True)
	print "In get_Pgg: the kmax is hard-coded to 2.9 h/Mpc to enable integration of spherical bessel function."
	index_cutk=next(j[0] for j in enumerate(kvec_temp) if j[1]>2.9)  # HARD CODED AND ARBITRARY MAX K TO GET BESSEL FUNCTION INTEGRATION TO RUN - maybe fixed in scipy 1.8.0?
	kvec=kvec_temp[0:index_cutk]
	PofK=PofK_temp[0:index_cutk]
	
	# Resample for better integration
	PofK_interp = scipy.interpolate.interp1d(kvec, PofK)
	k_resamp = np.logspace(np.log10(kvec[0]), np.log10(kvec[-1]), 10000)
	PofK_resamp = PofK_interp(k_resamp)
	
	#Now we need to compute an array in l and chiLvec of Pgg.	
	Clgg=np.zeros(len(lvec))

	for li in range(0,len(lvec)):
		print "li=",li
		Integrand  = np.zeros(len(k_resamp)) # have to do it this way because of spherical bessel implementation
		for ki in range(0,len(k_resamp)):	
			Integrand[ki] = k_resamp[ki] **2 * bias** 2 * PofK_resamp[ki] * (scipy.special.sph_jn(int(lvec[li]), k_resamp[ki] * chiLmean / (1.+zL) )[0][int(lvec[li])]) ** 2

		Clgg[li] = scipy.integrate.simps(Integrand, k_resamp)
		
	#Save answer:
	#np.savetxt(folderpath+outputfolder+'/Clgg_'+endfilename+'.txt', Clgg)
	
	#plt.figure()
	#plt.plot(lvec, Clgg)
	#plt.ylim(0, 0.00035)
	#plt.savefig('./plots/Clgg.png', unpack=True)
	#plt.close()
	
	return  Clgg
	
def get_Pgk():
	""" This function computes P_{gk}(l, chi_L, chi_S) """
	
	H=getHconf(chiLmean)
	Omz=getOmMx(chiLmean)
	Pof_lx=PofkGR_chimean(chiLmean)
	
	# Note that because it's so quick I'm computing this for all chiS values, even those below chiLmean, but we won't use this below chiLmean in the integral.
	Clgk=np.zeros((len(lvec), len(chiSvec)))
	for li in range(0, len(lvec)):
		for xiS in range(0, len(chiSvec)):
			Clgk[li, xiS] = 1.5 * bias * (chiSvec[xiS] - chiLmean) / chiLmean / chiSvec[xiS] * H**2 * Omz * Pof_lx[li] 
	
	return  Clgk

	
def get_Pkk():
	""" This function computes P_{kk}(l, chi_L, chi_L) """
	
	H=getHconf(chiLext)
	Omz=getOmMx(chiLext)
	Pof_lx=PofkGR(chiLext)
	
	# To reduce the number of integrals that must be done, it's best if we restrict to the region of chiSvec which is greater than chiLmean - this is the only part we use in any case.
	chiS_ind_chiL = next(j[0] for j in enumerate(chiSvec) if j[1]>=chiLmean)
	chiScut = chiSvec[chiS_ind_chiL:]
	#print "length=", len(chiScut)
	
		
	Clkk=np.zeros((len(lvec), len(chiScut), len(chiScut)))	
	for li in range(0,len(lvec)):
		#print "li in Clkk=", li
		for xiS in range(0,len(chiScut)):
			for bxiS in range(0,len(chiScut)):
				Clkk[li, xiS, bxiS]=  9./4. *H0**4 *scipy.integrate.simps((H/H0)**4 * Omz**2 * Pof_lx[li,:] *(chiScut[xiS] - chiLext) * (chiScut[bxiS]-chiLext) / (chiScut[xiS] * chiScut[bxiS]), chiLext)					
				
	# Save answer:
	#for li in range(0, len(lvec)):
	#	np.savetxt(folderpath+outputfolder+'/Clkk_'+endfilename_Clkk+'_l='+str(lvec[li])+'.txt', Clkk[li,:,:])
	
	return Clkk
	
############################# FUNCTIONS FOR DOING THE INTEGRALS #######################
	
def doints_Pgg(Clgg):
	""" This function does the integrals in dchiL, dbarchiL, dchiS, dbarchiS on the P_{gg} * gamma^2 / ns term"""

	#First do the integral over only SigmaC in chiS (the barchiS integral is totally the same, just two factors) (leaving off c^2 / 4 pi G factors - will add at the end)	
	
	# Only consider chiSvec above chiLmean:
	chiS_ind_chiL = next(j[0] for j in enumerate(chiSvec) if j[1]>=chiLmean)
	chiScut = chiSvec[chiS_ind_chiL:]
		
	barchiS_int=scipy.integrate.simps(Ws[chiS_ind_chiL:]* (chiSvec[chiS_ind_chiL:] - chiLmean) * chiLmean  * (1 + z_ofchi(chiLmean)) / chiSvec[chiS_ind_chiL:], chiSvec[chiS_ind_chiL:])

	# Now load Clgg
	#Clgg=np.loadtxt(folderpath+outputfolder+'/Clgg_'+endfilename+'.txt')
		
	#np.savetxt(folderpath+outputfolder+'/Pggterm_'+endfilename+'.txt', barchiS_int**2 * Clgg )
			
	return barchiS_int**2 * Clgg
	
def doints_Pgk(Clgk):
	""" This function does the integrals in dchiL, dbarchiL, dchiS, dbarchiS on the P_{gk} term"""
	
	# Do the integral in chiS - the bchiS one acts on the second factor of P_{gk} in exactly the same way, so just square it.	
	chiL_intans=np.zeros(len(lvec))
	
	# Only consider chiSvec above chiLmean:
	chiS_ind_chiL = next(j[0] for j in enumerate(chiSvec) if j[1]>=chiLmean)
	chiScut = chiSvec[chiS_ind_chiL:]
	
	for li in range(0, len(lvec)):
		chiL_intans[li] = scipy.integrate.simps(Ws[chiS_ind_chiL:] * Clgk[li,:][chiS_ind_chiL:] * (chiSvec[chiS_ind_chiL:] - chiLmean) * chiLmean * (1 + z_ofchi(chiLmean)) / chiSvec[chiS_ind_chiL:], chiSvec[chiS_ind_chiL:])
	
	#np.savetxt(folderpath+outputfolder+'/Pgkterm_'+endfilename+'.txt', chiL_intans**2 )
		
	return chiL_intans**2

	
def doints_Pkk(Clkk):
	""" This function does the integrals in dchiL, dbarchiL, dchiS, dbarchiS on the P_{kk} / nl term"""
	
	# Only consider chiSvec above chiLmean:
	chiS_ind_chiL = next(j[0] for j in enumerate(chiSvec) if j[1]>=chiLmean)
	chiScut = chiSvec[chiS_ind_chiL:]
	
	#First load Clkk
	#Clkk=np.zeros((len(lvec), len(chiSvec[chiS_ind_chiL:]), len(chiSvec[chiS_ind_chiL:])))
	#for li in range(0, len(lvec)):
	#	Clkk[li,:,:]=np.loadtxt(folderpath+outputfolder+'/Clkk_'+endfilename_Clkk+'_l='+str(lvec[li])+'.txt')	
			
	
	#Do the integral in chiS:
	chiS_intans=np.zeros((len(lvec), len(chiSvec[chiS_ind_chiL:])))		
	for li in range(0, len(lvec)):
		for bchiS in range(0,len(chiSvec[chiS_ind_chiL:])):
			chiS_intans[li, bchiS]= scipy.integrate.simps(Ws[chiS_ind_chiL:] * (chiSvec[chiS_ind_chiL:] - chiLmean) * chiLmean * (1 + z_ofchi(chiLmean)) / chiSvec[chiS_ind_chiL:] * Clkk[li, :, bchiS], chiSvec[chiS_ind_chiL:]) 
	
	# Now barchiS	
	bchiS_intans=np.zeros(len(lvec))
	for li in range(0,len(lvec)):
		bchiS_intans[li] = scipy.integrate.simps(Ws[chiS_ind_chiL:] * (chiSvec[chiS_ind_chiL:] - chiLmean) * chiLmean * (1 + z_ofchi(chiLmean)) / chiSvec[chiS_ind_chiL:] * chiS_intans[li,:], chiSvec[chiS_ind_chiL:])
		
	#np.savetxt(folderpath+outputfolder+'/Pkkterm_'+endfilename+'.txt', bchiS_intans )
	
	return bchiS_intans
	
def doints_PggPkk(Pkkterm, Clgg):
	""" This function constructs the Pgg*Pkk term from Pkk and Clgg"""
	
	#Load integrals over Pkk
	#Pkkterm = np.loadtxt(folderpath+outputfolder+'/Pkkterm_'+endfilename+'.txt')
	
	#And load Clgg
	#Clgg=np.loadtxt(folderpath+outputfolder+'/Clgg_'+endfilename+'.txt')
	
	#The PggPkk term is simply these two things multiplied in the effective lens redshift case:
	PggPkk = Clgg * Pkkterm
	
	np.savetxt(folderpath+outputfolder+'/PggPkkterm_'+endfilename+'.txt', PggPkk)
	
	return PggPkk
	
def doconstint():
	""" This function does the integrals in chiS, bchiS, chiL and bchiL for the constant term """
	
	# Only consider chiSvec above chiLmean:
	chiS_ind_chiL = next(j[0] for j in enumerate(chiSvec) if j[1]>=chiLmean)
	chiScut = chiSvec[chiS_ind_chiL:]

	chiSans = scipy.integrate.simps(Ws[chiS_ind_chiL:] * (chiSvec[chiS_ind_chiL:] - chiLmean) * chiLmean * (1 + z_ofchi(chiLmean))  / chiSvec[chiS_ind_chiL:], chiSvec[chiS_ind_chiL:])
	
	# The bchiL / bchiS integrals are the exact same so we don't need to do them again 
	
	#save=[0]
	#save[0]=chiSans ** 2 * gam ** 2 / ns
	
	#np.savetxt(folderpath+outputfolder+'/const_'+endfilename+'.txt', save)
	
	return chiSans ** 2 * gam ** 2 / ns / nl

def getwbar():
	""" This function computes wbar.  Note that we leave out the factor of (4piG)^2 / c^4 because it would cancel anyways.  """
	
	# Only consider chiSvec above chiLmean:
	chiS_ind_chiL = next(j[0] for j in enumerate(chiSvec) if j[1]>=chiLmean)
	chiScut = chiSvec[chiS_ind_chiL:]

	wbar = scipy.integrate.simps(Ws[chiS_ind_chiL:] * (chiSvec[chiS_ind_chiL:] - chiLmean)**2 * chiLmean**2 * (1.+z_ofchi(chiLmean))**2 / chiSvec[chiS_ind_chiL:]**2, chiSvec[chiS_ind_chiL:])

	#wbarsave=[0]
	#wbarsave[0]=wbar
	
	return wbar
	
def do_outsideints_SigR(i_Rbin, j_Rbin, Pgkterm, PggPkkterm, Pkkterm, Pggterm, constterm):
	""" This function does the integrals in l, R, and R' for the Delta Sigma(R) term """
	
	#Pgkterm		=	np.loadtxt(folderpath+outputfolder+'/Pgkterm_'+endfilename+'.txt')
	#PggPkkterm	=	np.loadtxt(folderpath+outputfolder+'/PggPkkterm_'+endfilename+'.txt')
	#Pkkterm		= 	np.loadtxt(folderpath+outputfolder+'/Pkkterm_'+endfilename+'.txt')
	#Pggterm		=	np.loadtxt(folderpath+outputfolder+'/Pggterm_'+endfilename+'.txt')
	#constterm	=	np.loadtxt(folderpath+outputfolder+'/const_'+endfilename+'.txt')
	
	# plot each thing to see what is dominating:
	plt.figure()
	plt.loglog(lvec, Pgkterm, 'b+', label='Pgk')
	plt.hold(True)
	plt.loglog(lvec, PggPkkterm, 'r+', label='PggPkk')
	plt.hold(True)
	plt.loglog(lvec, Pkkterm / nl, 'm+', label='Pkk')
	plt.hold(True)
	plt.loglog(lvec, Pggterm*gam**2 / ns, 'g+', label='Pgg')
	plt.hold(True)
	plt.loglog(lvec, constterm * np.ones(len(lvec)), 'k+', label='const')
	plt.hold(True)
	plt.loglog(lvec, ( Pgkterm + PggPkkterm + Pkkterm/nl + Pggterm*gam**2 / ns + constterm), 'k+', label='tot')
	plt.ylim(10**(-13), 10**(-3))
	plt.legend()
	plt.savefig('./plots/compareterms.png')
	plt.close()

	lint_ans=np.zeros((len(Rvec), len(Rvec)))
	
	for ri in range(0,len(Rvec)):
		#print "ri, outside ints R=", ri
		for rip in range(0,len(Rvec)):	
			#lint_ans[ri, rip] = scipy.integrate.simps(( Pgkterm + PggPkkterm + Pkkterm/nl + Pggterm*gam**2 / ns + constterm) * scipy.special.jv(2, Rvec[ri] * lvec) * scipy.special.jv(2, Rvec[rip] * lvec) * lvec, lvec)
			lint_ans[ri, rip] = scipy.integrate.simps(( Pgkterm + PggPkkterm + Pkkterm/nl + Pggterm*gam**2 / ns + constterm) * scipy.special.jv(2, Rvec[ri] * lvec / chiLmean) * scipy.special.jv(2, Rvec[rip] * lvec / chiLmean) * lvec, lvec)
		
	# Now do the Rprime integral.
	Rlowind_bini    =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[i_Rbin])
	Rhighind_bini   =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[i_Rbin+1])
		
	Rprime_intans=np.zeros(len(Rvec))	
	for ri in range(len(Rvec)):
		Rprime_intans[ri] = scipy.integrate.simps(lint_ans[ri,:][Rlowind_bini:Rhighind_bini], Rvec[Rlowind_bini:Rhighind_bini]) / (Rvec[Rhighind_bini] - Rvec[Rlowind_bini])

	# Now the r integral:
	Rlowind_binj    =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[j_Rbin])
	Rhighind_binj   =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[j_Rbin+1])
	Rintans = scipy.integrate.simps(Rprime_intans[Rlowind_binj:Rhighind_binj], Rvec[Rlowind_binj:Rhighind_binj]) / (Rvec[Rhighind_binj] - Rvec[Rlowind_binj])	

	# Add factors:
	ans_thisRbin	= Rintans  / (8. * np.pi**2) / fsky /wbar**2
	
	return ans_thisRbin
	
def do_outsideints_SigR0(i_Rbin, j_Rbin, Pgkterm, PggPkkterm, Pkkterm, Pggterm, constterm):
	""" This function does the integrals in R, and R' for the Delta Sigma(R0) term. This version uses a tiny bin around R0 to approximate the DeltaSigR0 part then averages over R for the bins themselves. """
	
	#Pgkterm		=	np.loadtxt(folderpath+outputfolder+'/Pgkterm_'+endfilename+'.txt')
	#PggPkkterm	=	np.loadtxt(folderpath+outputfolder+'/PggPkkterm_'+endfilename+'.txt')
	#Pkkterm		= 	np.loadtxt(folderpath+outputfolder+'/Pkkterm_'+endfilename+'.txt')
	#Pggterm		=	np.loadtxt(folderpath+outputfolder+'/Pggterm_'+endfilename+'.txt')
	#constterm	=	np.loadtxt(folderpath+outputfolder+'/const_'+endfilename+'.txt')
	
	lint_ans=np.zeros((len(Rvec), len(Rvec)))
	
	for ri in range(0,len(Rvec)):	
		for rip in range(0,len(Rvec)):				
			#lint_ans[ri, rip] = scipy.integrate.simps(( Pgkterm + PggPkkterm + Pkkterm/nl + Pggterm*gam*2 / ns + constterm) * scipy.special.jv(2, Rvec[ri] * lvec) * scipy.special.jv(2, Rvec[rip] * lvec) * lvec, lvec)
			lint_ans[ri, rip] = scipy.integrate.simps(( Pgkterm + PggPkkterm + Pkkterm/nl + Pggterm*gam*2 / ns + constterm) * scipy.special.jv(2, Rvec[ri] * lvec / chiLmean) * scipy.special.jv(2, Rvec[rip] * lvec / chiLmean) * lvec, lvec)
			
	
	Rlowind_R0=next(j[0] for j in enumerate(Rvec) if j[1]>=(R0-(R0/10.)))
	Rhighind_R0=next(j[0] for j in enumerate(Rvec) if j[1]>=(R0+(R0/10.)))

	# Now do the Rprime integral.
	Rprime_intans=np.zeros(len(Rvec))	
	for ri in range(len(Rvec)):
		Rprime_intans[ri] = scipy.integrate.simps(lint_ans[ri, :][Rlowind_R0:Rhighind_R0], Rvec[Rlowind_R0:Rhighind_R0]) / (Rvec[Rhighind_R0] - Rvec[Rlowind_R0])	
	# Now the r integral:
	Rintans = scipy.integrate.simps(Rprime_intans[Rlowind_R0:Rhighind_R0], Rvec[Rlowind_R0:Rhighind_R0]) / (Rvec[Rhighind_R0] - Rvec[Rlowind_R0])	

	Rlowind_bini    =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[i_Rbin])
	Rhighind_bini   =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[i_Rbin+1])

	# Now multiply this by R0^2 / R^2 and integrate in the current R bin:
	rint_ans = scipy.integrate.simps(R0**2 / Rvec[Rlowind_bini:Rhighind_bini]**2 * Rintans, Rvec[Rlowind_bini:Rhighind_bini]) / (Rvec[Rhighind_bini] - Rvec[Rlowind_bini])
	
	Rlowind_binj    =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[j_Rbin])
	Rhighind_binj   =       next(j[0] for j in enumerate(Rvec) if j[1]>=Redges[j_Rbin+1])

	# Now same again for R'
	rpint_ans = scipy.integrate.simps(R0**2 / Rvec[Rlowind_binj:Rhighind_binj]**2 * rint_ans, Rvec[Rlowind_binj:Rhighind_binj]) / (Rvec[Rhighind_binj] - Rvec[Rlowind_binj])	

	# Add factors:
	ans_thisRbin = rpint_ans / (8. * np.pi**2) / fsky / wbar**2 
	
	return ans_thisRbin
	
def assemble():
	""" This function puts the DeltaSigma(R) and DeltaSigma(R0) terms together."""

	DeltaSigR_term = np.loadtxt(folderpath+outputfolder+'/Rterm_'+endfilename+'_lpts='+str(lpts)+'.txt',unpack=True)
	DeltaSigR0_term = np.loadtxt(folderpath+outputfolder+'/R0term_'+endfilename+'_lpts='+str(lpts)+'.txt', unpack=True)
	
	covmat_total	=	 DeltaSigR_term + DeltaSigR0_term

	#This is just to test our Delta Sigma errors against those in Blake 2015. Debugging.
	# This factor accounts for the missing factor of c^2 / 4piG that we pulled out to make our calculations more numerically stable and which now should multiply the square-rooted variance. It also puts everything in units Msol * h / pc^2, because for some reason that's how Blake 2015 displays Delta Sigma
	Factor_Blake = c**2 / (4. * np.pi * G) *  3.086 * 10 **10 / (1.99 * 10 **30) 
	for i in range(0, numRbins):
		print Rcentres[i], "_", np.sqrt(DeltaSigR_term[i,i] + DeltaSigR0_term[i,i]) * Factor_Blake
	
	np.savetxt(folderpath+outputfolder+'/cov_Upgm_'+endfilename+'_lpts='+str(lpts)+'.txt', covmat_total)
	
	return


##########################################################################################
####################################### SET UP ###########################################
##########################################################################################

#Planck 2015 Parameters
Nnu=3.046
HH0=67.26
OmegaR=2.47*10**(-5)/(HH0/100.)**2
OmegaN=Nnu*(7./8.)*(4./11.)**(4./3.)*OmegaR
OmegaB=0.02222/(HH0/100.)**2
OmegaM=0.1199/(HH0/100.)**2
OmegaL=1.-OmegaM-OmegaB-OmegaR-OmegaN
OmegaK=0.0
h=HH0/100. #unitless
c=2.99792458*10**(8) # units m/s
H0=10**(5)/c # units h/MpC
MpCm=3.085678*10**22 #units m/MpC
G=6.67*10**(-11)

bias=1.0 # Assume constant bias for now

#Aminsq_in_steraidan	=	3600.*3282.8 # The number of square arcminutes in a steradian.

#Directory set up
folderpath 		= 	'/home/danielle/Dropbox/CMU/Research/EsubG_Horizon/Viability_Check/'
inputfolder		=	'/txtfiles/'
outputfolder		=	'/txtfiles/'
#cambfolder		=	'/camb/'
endfilename		=	'BOSS-Nakajima'
endfilename_Clkk	= 	'BOSS-Nakajima'
cambfolderpath = '/home/danielle/Documents/CMU/camb/'

#Parameters
R0				=	1.5

# Lenses:

#chiLmean		= 	883.    #z=0.32
# DESI
#zval			=	0.8
#chiLmean =  1937.41104112 #z=0.8

# SKA2
#chiLmean = 2293.35451023 #z=1.0
#zval = 1.0

# BOSS LOWZ
chiLmean 	=	884.
zval 		= 	0.32
nl			=	8.22 * 10**4 # surface density of lenses, number / steradian. See evernote for how I computed this.
#nl			= 10**2 		# THIS IS JUST TO TEST A SCENARIO WHERE SHAPE NOISE SHOULD HEAVILY DOMINATE.

# Sources:

# For the SDSS sources from Nakajima 2012 (See evernote for how I got these numbers
# Note this has a different functional form of dNdz than the typical Smail et al form
ns				=	1.4 * 10 **7 * 0.647242702668 #gal / steradian, where the multiplier is the fraction of sources which lie behind the effective lens redshift.
#ns 				= 10**3 # THIS IS JUST TO TEST A SCENARIO WHERE SHAPE NOISE SHOULD HEAVILY DOMINATE.
gam				=	0.36
fsky			=	0.218	# 9000 square degrees
alpha_			=	2.338
zs				=	0.303
maxchival 		=	4388.
minchival		=	293.
maxz			=	3.0
minz			=	0.1


# Euclid
#alpha=2.0
#beta=1.5
#z0=0.9/1.412
#minchival		=	1315.55184791 # z=0.5
#maxchival		=	3582.23968525 # z=2.0
#maxz			=	2.0
#minz			=	0.5

# LSST
#alpha			=	2.0
#beta			=	1.0
#z0				=	0.5
#minchival		=	292.606598364 # z=0.1
#maxchival		=	4944.99581468 # z=4.0
#maxz			=	4.0
#minz			=	0.1
#gam 			=   0.18

#SKA continuum source galaxies
#nofzfile = 'skacontinuum_mattjarvis.txt'
#minchival = 148.11597091  #z=0.05
#maxchival = 6474.24332819 # z=9.85
#maxchival = 6474.24

# For the survey considered in Reyes 2010
#nl				= 4.42 * 10 **5 		# surface density of lens galaxies per steradian
#gam				= 0.35			# rms shear
#ns				= 1.78 * 10 ** 7 		# surface density of source galaxies per steradian
#fsky				= 0.176

#For CFHTLenS in combination with BOSS LOWZ
#ns 				= 1.31*10**8
#gam				= 0.28
#fsky				= 0.0039
#nl				= 1.03 * 10 ** 5 # This is pretty sketchily derived but it will do for now I guess.

# For Euclid in combination with DESI
#ns = 3.55 * 10**8 #gal /steradian, Euclid
#gam  = 0.22 #Euclid
#fsky = 0.218 # DESI LRG's
#nl = 1.15 * 10**6 # DESI LRG's   # for 9000 deg survey    

# For Euclid in combination with SKA2
#ns = 3.55 * 10**8 #gal /steradian, Euclid
#gam  = 0.22 #Euclid
#fsky = 0.75 # SKA2
#nl = 1.18 * 10 ** 8 # galaxies per steradian for SKA 2
# Haven't accounted for the fact that some sources will be in front of lenses...

# For LSST and SKA2
#ns = 5.9 * 10 ** 8  * 0.67300016671 # gal / steradian LSST, accounting for overlap with SKA2 effective redshift.
#gam = 0.18  # LSST
#fsky =  0.75 # SKA2
#fsky = 0.485
#nl = 1.18 * 10 ** 8 # galaxies per steradian for SKA 2

# For SKA2 continuum and HI
#ns = 1.18 * 10 ** 8 * 0.764634133131 # gal /steraidan for SKA2 continuum (2 year survey, see 1501.03828), accounting for overlap with SKA HI galaxies.
#gam = 0.3  #SKA2 continuum
#fsky = 0.75 #SKA2
#nl = 9.68 * 10 ** 7 # galaxies per steradian for SKA2 HI (see 1501.03990)


#Vector set up
chiSmax			=	maxchival
chiSmin			=	minchival
chiSpts			=	100
Rpts			=	50
Rmin			=	R0 - R0/10.
Rmax			=	50
lpts			=	1000
lmin			=	1
lmax			=	2000
numRbins		=	8
chiLext_min		=	0.1
chiLext_max		=	chiSmax
chiLextpts		=	150

#CAMB P(k) files:
#Fileroot		=	'NL_matterpower_z='
#baseparamsfile		=	'errors_baseparams.ini'
#Fileroot = 'lin_kmax4000_matterpower_z='
Fileroot = 'NL_kmax4000_matterpower_z='

##########################################################################################
################################  MAIN FUNCTION CALLS ####################################
##########################################################################################

a = time.time()

# Set up
(chiSvec, lvec, Rvec, Redges, chiLext, Rcentres)		= 		setup_vectors()

#print "DeltaR1 =", Rcentres[1]-Rcentres[0]
#print "compare = ", np.sqrt(gam**2 / ns / nl / 4. * np.pi**2 * 

z_ofchi													=		getzLvec()
#print "zofchi=", list(z_ofchi(chiLext))
(Ws)													= 		getwindowfunctions_Nakajima()

# Get power spectra
Clgg = get_Pgg()
Clgk=get_Pgk()
Clkk = get_Pkk()

# Do the integrals on each term up to the l integral (so chiS, bchiS, chiL, bchiL)
Pggints = doints_Pgg(Clgg)
print "Done with Pgg integrals. Now do Pgk:"
Pgkints = doints_Pgk(Clgk)
print "Done with Pgk integrals. Now do Pkk:"
Pkkints = doints_Pkk(Clkk)
print "Done with Pkk integrals. Now do constant:"
constterm = doconstint()
print "Done with constant integrals. Now do PggPkk:"
PggPkkints = doints_PggPkk(Pkkints, Clgg)
print "Done with PggPkk integrals. Now getting wbar:"
wbar = getwbar()
print "wbar=", wbar
print "Done with getting wbar. Now doing integrals over R:"

#These ones must be done for each set of bins
DeltaSigR_term		=	np.zeros((numRbins, numRbins))
DeltaSigR0_term		=	np.zeros((numRbins, numRbins))

for i_R in range(0, numRbins):
	for j_R in range(0, numRbins):
		print "i bin=", i_R, "j bin=", j_R
		DeltaSigR_term[i_R, j_R]	=	do_outsideints_SigR(i_R, j_R, Pgkints, PggPkkints, Pkkints, Pggints, constterm)
		print "DeltaSigR=", DeltaSigR_term[i_R, j_R]
		DeltaSigR0_term[i_R, j_R]	=	do_outsideints_SigR0(i_R, j_R,Pgkints, PggPkkints, Pkkints, Pggints, constterm)
		print "Delta Sig R0=", DeltaSigR0_term[i_R, j_R]	

np.savetxt(folderpath+outputfolder+'/R0term_'+endfilename+'_lpts='+str(lpts)+'.txt', DeltaSigR0_term)
np.savetxt(folderpath+outputfolder+'/Rterm_'+endfilename+'_lpts='+str(lpts)+'.txt', DeltaSigR_term)

#DeltaSigR_term = np.loadtxt(folderpath+outputfolder+'/R0term_'+endfilename+'.txt')
#DeltaSigR0_term = np.loadtxt(folderpath+outputfolder+'/Rterm_'+endfilename+'.txt')
	
assemble()


print '\nTime for completion:', '%.1f' % (time.time() - a), 'seconds'
