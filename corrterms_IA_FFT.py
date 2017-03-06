# This file computes the correlation function at in R and Delta (sometimes called r_p and pi). 
# This is an updated version which imports the 1d correlation function from the FFTlog code, version by Anze Slosar, because the previous method of directly doing the Fourier transform myself was killing small scale behaviour.

import numpy as np
import scipy.interpolate
import scipy.integrate
import time
import matplotlib.pyplot as plt

###############################################################
################### FUNCTIONS DEFINITIONS #####################
###############################################################

def savedelta_R():
	""" This functions saves delta and R vectors"""
	np.savetxt(folderpath+outputfolder+deltafile, deltavec)
	np.savetxt(folderpath+outputfolder+Rfile, Rvec)

	return

def getcorrfunc():
	"""This function computes the 3D correlation function xi as a function of R and delta"""
	
	# Import the 1d correlation function as calculated in FFTlog
	(rvec, corrfunc1d) = np.loadtxt(folderpath+outputfolder+corrfunc1dfile,unpack=True)
	
	# Multiply by the desired galaxy bias amplitude - just one factor because we want e_gm
	
	corrfunc1d_gm = corrfunc1d * bias
	
	#plt.figure()
	#plt.loglog(rvec, corrfunc1d)
	#plt.ylim(0.01, 3000)
	#plt.xlim(0.05, 100)
	#plt.savefig('./plots/corrfunc_bothterms_M6e13h_fixRhoC.png')
	#plt.close()
	
	#exit()
	
	#Define an array to hold the correlation function as a 2D array in R and Delta 
	corrfunc2d_gm=np.zeros((len(deltavec),len(Rvec)))

	#Interpolate in rvec
	interp_corrfunc1d=scipy.interpolate.interp1d(rvec, corrfunc1d_gm)
	
	#Now get the value of this at every bigR and Delta val
	print 'Interpolate to get 2D corr fun'
	for ri in range(0,len(Rvec)):
		for di in range(0, len(deltavec)):
			corrfunc2d_gm[di, ri]=interp_corrfunc1d((deltavec[di]**2+Rvec[ri]**2)**(0.5))
	print "done getting 2d corr func" 
	
	#Save the correlation function as a function of delta, theta
	np.savetxt(folderpath+outputfolder+corrfunc2dfile, corrfunc2d_gm)
	
	exit()

	return

def getintoverR():
	""" This function computes the integral over R of the correlation function for the first term of the three-part correlation function term"""

	#Load correlation function as a function of R and Delta:
	corrfunc_gm=np.loadtxt(folderpath+outputfolder+corrfunc2dfile)
	
	Rintans=np.zeros((len(deltavec),len(Rvec)))
	#Rintans = np.zeros((len(Rvec)))
	for di in range(0, len(deltavec)):
		for ri in range(0,len(Rvec)):
			#print "di=", di, "ri=", ri
			Rintans[di,ri] = 2./Rvec[ri]**2 * scipy.integrate.simps(Rvec[0:ri+1]**2*corrfunc_gm[di,:][0:ri+1], np.log(Rvec[0:ri+1]))
			#Rintans[ri] = 2./Rvec[ri]**2 * scipy.integrate.trapz(Rvec[0:ri+1]**2, np.log(Rvec[0:ri+1]))
			
	#plt.figure()
	#plt.semilogx(Rvec, np.ones(len(Rvec)), 'r+')
	#plt.hold(True)
	#plt.semilogx(Rvec, Rintans, 'b+')
	#plt.savefig('./plots/test_Rint.png')
	#plt.close()
	
	#exit()
		

	#Save the value of the integral as a function of delta and theta
	np.savetxt(folderpath+outputfolder+Rintfile, Rintans)
	
	return 


###############################################################
########################## SET UP #############################
###############################################################

a = time.time()

folderpath 	= 	'/home/danielle/Dropbox/CMU/Research/Intrinsic_Alignments/' #This is the folder where everything is happening

# Set parameters which are used to create vectors in Delta and R
Rpts		=	400 	#Number of points in Rvec
Rmin 		= 	0.000246558
Rmax		=	200.
Dnegmax = 1000.
Dposmax = 1000.
Deltamin = 0.000246558
zval = 0.28
kmax=4000 #This is the max k in the power spectrum file.

bias = 1.77

outputfolder	= 	'/txtfiles/'
corrfunc2dfile	=	'/corr_2d_z='+str(zval)+'_Singh2014params.txt'
Rintfile	=	'/Rint_z='+str(zval)+'_kmax='+str(kmax)+'_M6e13h_fixRhoC.txt'
#corrfunc1dfile ='/corr_1d_NL_z='+str(zval)+'_kmax'+str(kmax)+'.txt'
corrfunc1dfile = '/corr_z=0.28_Singh2014params.txt'

##############################################################
############## Set up the Delta and R vectors ################
##############################################################

# These are not in a function because I want them as global variables

# Make two deltavecs and attach them to get more spacing near the delta=0 point:
delta_neg = -np.logspace(np.log10(Rmin), np.log10(Dnegmax), 2000)
delta_pos = np.logspace(np.log10(Rmin), np.log10(Dposmax), 2000)
delta_neg_list = list(delta_neg)
delta_neg_list.reverse()
delta_neg_rev = np.asarray(delta_neg_list)
deltavec = np.append(delta_neg_rev, delta_pos)

Rvec		=	np.logspace(np.log10(Rmin), np.log10(Rmax), Rpts)

deltafile	=	'corr_delta_z='+str(zval)+'.txt'
Rfile		=	'corr_rp_z='+str(zval)+'.txt'

###############################################################
#################### MAIN FUNCTION CALLS ######################
###############################################################

#Save delta and R vectors so I know what they are and can use them later:
savedelta_R()

print "Get Correlation Function"
# Get 3D correlation function xi(R,\Delta) (save instead of return, because this takes a while)
getcorrfunc()

exit()

print "Do R Integral"
# Perform the integral over R for the first term (again save instead of return)
getintoverR()

print '\nTime for completion:', '%.1f' % (time.time() - a), 'seconds'




