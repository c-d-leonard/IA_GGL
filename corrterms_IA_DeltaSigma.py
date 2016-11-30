# This file computes the correlation function "ADSD" term which is common between Upsilon_{gm} and Upsilon_{gg}
# This is an updated version for work on the E_G / Horizon AGN project, September 2016
# Updates: Introduced scipy.integrate.simps integration instead of sup-optimal scipy.integrate.quad

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
	
	print folderpath+outputfolder+deltafile
	np.savetxt(folderpath+outputfolder+deltafile, deltavec)
	np.savetxt(folderpath+outputfolder+Rfile, Rvec)

	return

def getcorrfunc():
	"""This function computes the 3D correlation function xi as a function of R and delta"""

	#Import P(k) at z=l, as well as k vector
	#kvec, Pofk=np.loadtxt(cambpath+cambinputfile, unpack=True)
	#Pofk_int=scipy.interpolate.interp1d(kvec, Pofk)
	#xvec = np.log(kvec)

	#Get the total number of calculations of the correlation function here - this will be the length of rvec
	#corrfunc1d=np.zeros(len(rvec))

	#Define an array to hold the correlation function as a 2D array in R and Delta (we'll get this afterwards)
	corrfunc2d=np.zeros((len(deltavec),len(Rvec)))
	
	#print 'do integral in k for 1d corr func'
	#for ri in range(0, len(rvec)):
		#corrfunc1d[ri] = scipy.integrate.simps(Pofk * kvec**2 * np.sin(kvec*rvec[ri]) / (kvec * rvec[ri]) / 2. / np.pi**2, kvec) #*np.exp(-0.005*kvec**2), kvec)
		#corrfunc1d[ri] = scipy.integrate.simps(Pofk * kvec**3 * np.sin(kvec*rvec[ri]) / (kvec * rvec[ri]) / 2. / np.pi**2, np.log(kvec)) #*np.exp(-0.005*kvec**2), kvec)

	(rvec, corrfunc1d) = np.loadtxt('./txtfiles/corr_1d_NL_z=0.32.txt',unpack=True)
	
	#plt.figure()
	#plt.loglog(rvec, corrfunc1d, '+')
	#plt.xlim(0.1,50)
	#plt.ylim(0.005,1000)
	#plt.savefig('./test_corrfunc1d_upload.png')
	#plt.close()

	#Interpolate in rvec
	interp_corrfunc1d=scipy.interpolate.interp1d(rvec, corrfunc1d)

	r_2d_test = np.zeros(len(Rvec)*len(deltavec))
	corr_1d_fr_2d = np.zeros(len(Rvec)*len(deltavec))
	#Now get the value of this at every bigR and Delta val
	i=0
	print 'Interpolate to get 2D corr fun'
	for di in range(0, len(deltavec)):
		for ri in range(0,len(Rvec)):
			corrfunc2d[di, ri]=interp_corrfunc1d((deltavec[di]**2+Rvec[ri]**2)**(0.5))
			r_2d_test[i] = (deltavec[di]**2+Rvec[ri]**2)**(0.5)
			corr_1d_fr_2d[i] = corrfunc2d[di, ri]
			i=i+1
			#print "corrfunc2d=", corrfunc2d[di,ri]
			
	#print "sorted r=", sorted(list(r_2d_test))
			
	#plt.figure()
	#plt.loglog(r_2d_test, corr_1d_fr_2d, '+')
	#plt.xlim(0.1,50)
	#plt.ylim(0.005,1000)
	#plt.savefig('./test_corrfunc2d.png')
	#plt.close()

	print "done getting 2d corr func" 

	#Save the correlation function as a function of r
	#save_1d = np.column_stack((rvec, corrfunc1d))
	#np.savetxt(folderpath+outputfolder+corrfunc1dfile, save_1d)

	#Save the correlation function as a function of delta, theta
	np.savetxt(folderpath+outputfolder+corrfunc2dfile, corrfunc2d)

	return

def getintoverR():
	""" This function computes the integral over R of the correlation function for the first term of the three-part correlation function term"""

	#Load correlation function as a function of R and Delta:
	corrfunc=np.loadtxt(folderpath+outputfolder+corrfunc2dfile)
        
	#Declare an array to hold the answer to the integral
	Rintans=np.zeros((len(deltavec), len(Rvec)))

	for di in range(len(deltavec)/2, len(deltavec)):
		#print "delta=", deltavec[di]
		for ri in range(0,len(Rvec)):
			Rintans[di,ri] = 2./Rvec[ri]**2 * scipy.integrate.simps(Rvec[0:ri+1]**2*corrfunc[di,:][0:ri+1], np.log(Rvec[0:ri+1]))
			
	for di in range(0,len(deltavec)/2):
		print "R Integral: delta (negative)=", deltavec[len(deltavec)/2-(di+1)], "delta positive=", deltavec[len(deltavec)/2+di]
		Rintans[len(deltavec)/2-(di+1),:] = Rintans[len(deltavec)/2+di,:]

	#Save the value of the integral as a function of delta and theta
	np.savetxt(folderpath+outputfolder+Rintfile, Rintans)
	
	return


###############################################################
########################## SET UP #############################
###############################################################

a = time.time()

folderpath 	= 	'/home/danielle/Dropbox/CMU/Research/Intrinsic_Alignments/' #This is the folder where everything is happening


# Set parameters which are used to create vectors in Delta and R
corrfunc_cutoff	=	500 	#Max Delta value
#deltapts	=	2000	# This gurantees convergence within about 0.2% (versus going to 3000 pts)
deltapts 	=	10000
Rpts		=	200 	#Number of points in Rvec
#R0		=	1.5 	#The "guess" R0 to which we will pick the closest value in Rvec to be R0
rpts		=	50000 	#points in 'small r' vector
Rmin 		= 	0.001
Rmax		=	22.0
zval		=	0.0 

outputfolder	= 	'/txtfiles/'
endfilename	=	'z='+str(zval)+'_r'+str(rpts)+'pts_Dabs'+str(corrfunc_cutoff)+'_D'+str(deltapts)+'pts_R'+str(Rpts)+'pts_Rmin='+str(Rmin)+'_Rmax='+str(Rmax)+'_nl'
#corrfunc1dfile	= 	'/corr_1d_z=0.32.txt'
corrfunc2dfile	=	'/corr_z=0.32.txt'
Rintfile	=	'/Rint_z=0.32.txt'
#cambinputfile	= 	'halofit_IA_morek_matterpower_z=0.dat'
#cambpath 	= 	'/home/danielle/Documents/CMU/camb/'

##############################################################
############## Set up the Delta and R vectors ################
##############################################################

# These are not in a function because I want them as global variables

deltavec	=	np.linspace(-corrfunc_cutoff, corrfunc_cutoff, deltapts) 
Rvec		=	np.logspace(np.log10(Rmin), np.log10(Rmax), Rpts)
deltafile	=	'corr_delta_z=0.32.txt'
Rfile		=	'corr_rp_z=0.32.txt'

#Define the vector of "small r" points (r=sqrt(R^2+Delta^2))
#rvec		=	np.linspace(0.0001, np.sqrt(max(Rvec)**2+max(abs(deltavec))**2), rpts)

###############################################################
#################### MAIN FUNCTION CALLS ######################
###############################################################

#Save delta and R vectors so I know what they are and can use them later:
savedelta_R()

print "Get Correlation Function"
# Get 3D correlation function xi(R,\Delta) (save instead of return, because this takes a while)
getcorrfunc()

print "Do R Integral"
# Perform the integral over R for the first term (again save instead of return)
getintoverR()

print '\nTime for completion:', '%.1f' % (time.time() - a), 'seconds'




