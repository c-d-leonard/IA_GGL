# This is a script which constrains the (amplitude? parameters of a model?) of intrinsic alignments, using multiple shape measurement methods.

import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import subprocess
import shutil

########## FUNCTIONS ##########

####### SET UP & BASICS #######

def setup_rp_bins(rmin, rmax, nbins):
	""" Sets up the edges of the bins of projected radius """
	
	bins = scipy.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
	
	return bins

def rp_bins_mid(rp_edges):
        """ Gets the middle of each projected radius bin."""

        logedges=np.log10(rp_edges)
        bin_centers=np.zeros(len(rp_edges)-1)
        for ri in range(0,len(rp_edges)-1):
                bin_centers[ri]    =       10**((logedges[ri+1] - logedges[ri])/2. +logedges[ri])

        return bin_centers

def com(z_):
	""" Gets the comoving distance in units of Mpc/h at a redshift or a set of redshifts. """

	OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN

	def chi_int(z):
		return 1. / (pa.H0 * ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )**(0.5))

	if hasattr(z_, "__len__"):
		chi=np.zeros((len(z_)))
		for zi in range(0,len(z_)):
			chi[zi] = scipy.integrate.quad(chi_int,0,z_[zi])[0]
	else:
		chi = scipy.integrate.quad(chi_int, 0, z_)[0]

	return chi

def z_interpof_com():
	""" Returns an interpolating function which can give z as a function of comoving distance. """

	z_vec = scipy.linspace(0., 10., 1000) # This hardcodes that we don't care about anything over z=10.

	com_vec = com(z_vec)

	z_of_com = scipy.interpolate.interp1d(com_vec, z_vec)

	return	z_of_com

def get_z_close(z_l, cut_MPc_h):
	""" Gets the z above z_l which is the highest z at which we expect IA to be present for that lens. cut_Mpc_h is that separation in Mpc/h."""

	com_l = com(z_l) # Comoving distance to z_l, in Mpc/h

	tot_com_high = com_l + cut_MPc_h
	tot_com_low = com_l - cut_MPc_h

	# Convert tot_com back to a redshift.

	z_cl_high = z_of_com(tot_com_high)
	z_cl_low = z_of_com(tot_com_low)

	return (z_cl_high, z_cl_low)

def get_areas(bins, z_eff):
        """ Gets the area of each projected radial bin, in square arcminutes. z_eff = effective lens redshift. """

        # Areas in units (Mpc/h)^2
        areas_mpch = np.zeros(len(bins)-1)
        for i in range(0, len(bins)-1):
                areas_mpch[i] =  np.pi * (bins[i+1]**2 - bins[i]**2)

        #Comoving distance out to effective lens redshift in Mpc/h
        chi_eff = com(z_eff)

        # Areas in square arcminutes (466560000 / pi = sqAM in a sphere)
        areas_sqAM = areas_mpch * (466560000. / np.pi) / (3 * np.pi * chi_eff**2)

        return areas_sqAM
	
def get_NofZ_unnormed(a, zs, z_min, z_max, zpts):
	""" Returns the dNdz of the sources as a function of photometric redshift, as well as the z points at which it is evaluated."""
	z = scipy.linspace(z_min, z_max, zpts)
	
	nofz_ = (z / zs)**(a-1) * np.exp(-0.5 * ( z / zs)**2)

	return (z, nofz_)

def get_z_frac(rp_cents_):
        """ Gets the fraction of sources of the full survey which are within our photo-z cut (excess and smooth background)"""
        
        # The normalization factor should be the norm over the whole range of z for which the number density of the survey is given, i.e. z min to zmax.       
        (z, dNdz) = N_of_zph_weighted(pa.zmin, pa.zmax, pa.zmin, pa.zmax, z_close_low, z_close_high, pa.zmin_ph, pa.zmax_ph, pa.e_rms_mean)
        
        frac_rand = scipy.integrate.simps(dNdz, z)
        
        boost_samp = get_boost(rp_cents_, pa.boost_samp)
        boost_tot = get_boost(rp_cents_, pa.boost_tot)
        
        frac_total = boost_samp / boost_tot * frac_rand

        return frac_total

def get_perbin_N_ls(rp_bins_, zeff_, ns_, nl_, A, rp_cents_):
	""" Gets the number of lens/source pairs relevant to each bin of projected radius """
	""" zeff_ is the effective redshift of the lenses. frac_ is the fraction of sources in the sample. ns_ is the number density of sources per square arcminute. nl_ is the number density of lenses per square degree. A is the survey area in square degrees."""
        
	frac_		=	get_z_frac(rp_cents_)

	# Get the area of each projected bin in square arcminutes
	bin_areas       =       get_areas(rp_bins_, zeff_)

	N_ls_pbin = nl_ * A * ns_ * bin_areas * frac_

	return N_ls_pbin


###### LINEAR  / NONLINEAR ALIGNMENT MODEL + 1 HALO TERMS FOR FIDUCIAL GAMMA_IA ######

def window_LA(tag):
	""" Get window function for projected correlations in linear alignment model. Tag = '+g' or 'gg', determines dndz's."""
	
	sigz = 0.001
	z = scipy.linspace(pa.zeff-5.*sigz, pa.zeff+5.*sigz, 50)
	print "z=", z
	dNdz_1 = 1. / np.sqrt(2. * np.pi) / sigz * np.exp(-(z-pa.zeff)**2 / (2. *sigz**2))
	
	chi = com(z)
	OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
	dzdchi = pa.H0 * ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )**(0.5)
	
	if (tag =='gg'):
		dNdz_2 = dNdz_1
	elif ((tag=='+g') or (tag=='g+')):
		(z, dNdz_2) = get_NofZ_unnormed(pa.alpha, pa.zs, pa.zeff-5.*sigz, pa.zeff+5.*sigz, 50)
		
	norm = scipy.integrate.simps(dNdz_1*dNdz_2 / chi**2 * dzdchi, z)
	
	window = dNdz_1*dNdz_2 / chi**2 * dzdchi / norm
	
	return (z, window )
	
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
	
	#plot_quant_vs_quant(z_, D, './D(z).png')
	
	return D
	
def get_Pk(z_, tag):
	""" Calls camb and returns the linear power spectrum for the current cosmological parameters and at the redshifts of interest. Returns a 2D array in k and z."""
	
	# The redshifts have to be in a certain order to make camb happy:
	z_list = list(z_)
	z_list.reverse()
	z_rev = np.asarray(z_list)
	
	cambfolderpath = '/home/danielle/Documents/CMU/camb'
	if (tag=='gg'):
		param_string='output_root=IA_shapes_gg\nombh2='+str(pa.OmB * (pa.HH0/100.)**2)+'\nomch2='+str(pa.OmC* (pa.HH0/100.)**2)+'\nhubble='+str(pa.HH0)+'\ntransfer_num_redshifts='+str(len(z_))
	elif ((tag=='gp') or (tag=='pg')):
		param_string='output_root=IA_shapes_gp\nombh2='+str(pa.OmB * (pa.HH0/100.)**2)+'\nomch2='+str(pa.OmC* (pa.HH0/100.)**2)+'\nhubble='+str(pa.HH0)+'\ntransfer_num_redshifts='+str(len(z_))
			
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
	if (tag=='gg'):
		(k, P) = np.loadtxt(cambfolderpath+'/IA_shapes_gg_matterpower_z='+str(z_[0])+'.dat',unpack=True)
	elif ((tag=='gp') or (tag=='pg')):
		(k, P) = np.loadtxt(cambfolderpath+'/IA_shapes_gp_matterpower_z='+str(z_[0])+'.dat',unpack=True)
	
	Pofkz = np.zeros((len(z_),len(k))) 
	for zi in range(0,len(z_)):
		if (tag=='gg'):
			k, Pofkz[zi, :] = np.loadtxt(cambfolderpath+'/IA_shapes_gg_matterpower_z='+str(z_[zi])+'.dat', unpack=True)
		elif ((tag=='gp') or (tag=='pg')):
			k, Pofkz[zi, :] = np.loadtxt(cambfolderpath+'/IA_shapes_gp_matterpower_z='+str(z_[zi])+'.dat', unpack=True)
		
	#plot_quant_vs_quant(k, Pofkz[0,:],'./Pktest1.png')
	#plot_quant_vs_quant(k, Pofkz[20,:],'./Pktest20.png')
	#plot_quant_vs_quant(k, Pofkz[40,:],'./Pktest40.png')
	
	return (k, Pofkz)

def get_pi(q1, q2, q3, z_):
	""" Returns the pi functions requires for the 1 halo term in wg+, at z_ """
	
	pi = q1 * np.exp(q2 * z_**q3)
	
	return pi
	
def get_P1haloIA(z, k):
	""" Returns the power spectrum required for the wg+ 1 halo term, at z and k_perpendicular ( = k) """
	
	#z=scipy.linspace(0., 3.0, 1000)
	
	#z = 0.11
	
	p1 = get_pi(pa.q11, pa.q12, pa.q13, z)
	p2 = get_pi(pa.q21, pa.q22, pa.q23, z)
	p3 = get_pi(pa.q31, pa.q32, pa.q33, z)
	
	#print "p1=", p1
	#print "p2=", p2
	#print "p3=", p3
	
	#exit()
	
	#plot_quant_vs_quant(z, p1, './p1.png')
	#plot_quant_vs_quant(z, p2, './p2.png')
	#plot_quant_vs_quant(z, p3, './p3.png')
	
	#plt.figure()
	#plt.semilogy(z, p1, 'b+')
	#plt.hold(True)
	#plt.semilogy(z, p2, 'r+')
	#plt.hold(True)
	#plt.semilogy(z, p3, 'k+')
	#plt.ylim(10**(-2), 10**3)
	#plt.savefig('./p1p2p3.png')
	#plt.close()
	
	#exit()
	
	
	P1halo = np.zeros((len(k), len(z)))
	for ki in range(0,len(k)):
		for zi in range(0,len(z)):
			P1halo[ki, zi]  = pa.ah * ( k[ki] / p1[zi] )**2 / ( 1. + ( k[ki] /p2[zi] )** p3[zi])
	
	#P1halo = pa.ah * ( k/ p1 )**2 / ( 1. + ( k /p2 )** p3)
			
	#plt.figure()
	#plt.loglog(k, P1halo, 'b+')
	#plt.ylim(10**(-10), 10**10)
	#plt.savefig('./P1halo.png')
	#plt.close()
	
	
	#exit()
	plot_quant_vs_quant(k, P1halo[:, 10], './P1halo10.png')
	plot_quant_vs_quant(k, P1halo[:, 25], './P1halo25.png')
	plot_quant_vs_quant(k, P1halo[:, 40], './P1halo40.png')
	
	#exit()
	
	return P1halo
	
def wgp_1halo(rp_c_):
	""" Returns the 1 halo term of wg+(rp) """
	
	(z, w) = window_LA('g+') # Same window function as for the NLA term.
	
	# Set up a k vector to integrate over:
	k = np.logspace(-5., 4., 10000)
	
	# Get the `power spectrum' term
	P1h = get_P1haloIA(z, k)
	
	# First do the integral over z:
	zint = np.zeros(len(k))
	for ki in range(0,len(k)):
		zint[ki] = scipy.integrate.simps(P1h[ki, :] * w , z)
		
	plot_quant_vs_quant(k, zint, './zint.png')
		
	# Now do the integral in k
	ans = np.zeros(len(rp_c_))
	for rpi in range(0,len(rp_c_)):
		integrand = k * zint * scipy.special.j0(rp_c_[rpi] * k)
		#for ki in range(0,len(k)):
			#if (integrand[ki]<0.1):
			#	integrand[ki] = 0.
		ans[rpi] = scipy.integrate.simps(k * zint * scipy.special.j0(rp_c_[rpi] * k), k)
		
	plot_quant_vs_quant(k, integrand, './integrate.png')
		
	wgp1h = ans / (2. * np.pi)
	
	plt.figure()
	plt.loglog(rp_c_, wgp1h, 'bo')
	plt.ylim(10**(-2), 20)
	plt.xlim(10**(-1), 200)
	plt.savefig('./wgp1h.png')
	plt.close()
		
	return wgp1h
	
def wgg_wgp_LA(rp_cents_):
	""" Returns wgg and wg+ in the linear alignment model"""
	
	z_gg, win_gg = window_LA('gg')
	z_gp, win_gp = window_LA('g+')
	
	print "check norm, gg=", scipy.integrate.simps(win_gg, z_gg)
	print "check norm, g+=", scipy.integrate.simps(win_gg, z_gg)
	
	(k_gg, P_gg) = get_Pk(z_gg, 'gg')
	(k_gp, P_gp) = get_Pk(z_gp, 'gp')
	
	plot_quant_vs_quant(k_gg, P_gg[0], './Pgg.png')
	
	D_gp = growth(z_gp)
	
	# FIRST do the integrals over z for wgg and wg+. Don't yet interpolate in k.
	zint_gg = np.zeros(len(k_gg))
	for ki in range(0,len(k_gg)):
		zint_gg[ki] = scipy.integrate.simps(win_gg * P_gg[:, ki], z_gg)
		
	plot_quant_vs_quant(k_gg, zint_gg, './zint_gg.png')


	zint_gp = np.zeros(len(k_gp))
	for ki in range(0,len(k_gp)):
		zint_gp[ki] = scipy.integrate.simps(win_gp * P_gp[:, ki] / D_gp, z_gp)
		
	# Define vectors of kp (kperpendicual) and kz
	kp_gg = np.logspace(np.log10(k_gg[0]), np.log10(k_gg[-1]/ np.sqrt(2.01)), 11000)
	kp_gp = np.logspace(np.log10(k_gp[0]), np.log10(k_gp[-1]/ np.sqrt(2.01)), 11000)
	kz_gg = np.logspace(np.log10(k_gg[0]), np.log10(k_gg[-1]/ np.sqrt(2.01)), 11000)
	kz_gp = np.logspace(np.log10(k_gp[0]), np.log10(k_gp[-1]/ np.sqrt(2.01)), 11000)
	
	# Interpolate the answers to the z integral in k so I can get it in terms of kperp and kz
	kinterp_gg = scipy.interpolate.interp1d(k_gg, zint_gg)
	kinterp_gp = scipy.interpolate.interp1d(k_gp, zint_gp)
	
	# Get the result of the z integral in terms of kperp and kz for both gg ...
	kpkz_gg = np.zeros((len(kp_gg), len(kz_gg)))
	for kpi in range(0,len(kp_gg)):
		for kzi in range(0, len(kz_gg)):
			kpkz_gg[kpi, kzi] = kinterp_gg(np.sqrt(kp_gg[kpi]**2 + kz_gg[kzi]**2))
			
	# And g+
	kpkz_gp = np.zeros((len(kp_gp), len(kz_gp)))
	for kpi in range(0,len(kp_gp)):
		#print "kpi=", kpi
		for kzi in range(0, len(kz_gp)):
			kpkz_gp[kpi, kzi] = kinterp_gp(np.sqrt(kp_gp[kpi]**2 + kz_gp[kzi]**2))
			
	# Do the integrals in kz
	kz_int_gg = np.zeros(len(kp_gg))
	for kpi in range(0,len(kp_gg)):
		kz_int_gg[kpi] = scipy.integrate.simps(kpkz_gg[kpi,:] * kp_gg[kpi] / kz_gg * np.sin(kz_gg*pa.ProjMax), kz_gg)
		
	kz_int_gp = np.zeros(len(kp_gp))
	for kpi in range(0,len(kp_gp)):
		kz_int_gp[kpi] = scipy.integrate.simps(kpkz_gp[kpi,:] * kp_gp[kpi]**3 / ( (kp_gp[kpi]**2 + kz_gp**2)*kz_gp) * np.sin(kz_gp*pa.ProjMax), kz_gp)
	
	# Finally, do the integrals in kperpendicular
	kp_int_gg = np.zeros(len(rp_cents_))
	kp_int_gp = np.zeros(len(rp_cents_))
	for rpi in range(0,len(rp_cents_)):
		kp_int_gg[rpi] = scipy.integrate.simps(scipy.special.j0(rp_cents_[rpi]* kp_gg) * kz_int_gg, kp_gg)
		kp_int_gp[rpi] = scipy.integrate.simps(scipy.special.jv(2, rp_cents_[rpi]* kp_gp) * kz_int_gp, kp_gp)
		
	wgg = kp_int_gg * pa.bs * pa. bd / np.pi**2 
	wgp = kp_int_gp * pa.Ai * pa.bd * pa.C1rho * (pa.OmC + pa.OmB) / np.pi**2
	
	wgg_stack = np.column_stack((rp_cents_, wgg))
	wgp_stack = np.column_stack((rp_cents_, wgp))
	np.savetxt('./txtfiles/wgg_11000pts.txt', wgg_stack)
	np.savetxt('./txtfiles/wgp_11000pts.txt', wgp_stack)
	
	
	#plt.figure()
	#plt.loglog(rp_cents_, wgp, 'b+')
	#plt.hold(True)
	#plt.loglog(rp_cents_, wgp1h, 'r+')
	#plt.savefig('./wgp_bothterm.png')
	#plt.close()

	#gammaIA = wgp / wgg
	
	return (wgg, wgp)
	
def gamma_fid(rp):
	""" Returns the fiducial gamma_IA from a combination of terms from different models which are valid at different scales """
	
	(wgg_NLA, wgp_NLA) = wgg_wgp_LA(rp)
	
	#(rp_hold, wgg_NLA) = np.loadtxt('./txtfiles/wgg_3000pts.txt', unpack=True)
	#(rp_hold, wgp_NLA) = np.loadtxt('./txtfiles/wgp_3000pts.txt', unpack=True)
	
	wgp_1h = wgp_1halo(rp)
	
	gammaIA = (wgp_NLA + 2. * pa.close_cut) / (wgg_NLA + 2. * pa.close_cut)
	
	plt.figure()
	plt.loglog(rp, wgp_NLA, 'bo')
	plt.hold(True)
	plt.loglog(rp, wgp_1h, 'go')
	plt.ylim(10**(-2), 20)
	plt.xlim(10**(-1), 200)
	plt.savefig('./wg+_both_11000.png')
	plt.close()
	
	plt.figure()
	plt.semilogx(rp, rp* wgg_NLA, 'go')
	plt.ylim(0, 300)
	plt.xlim(10**(-1), 200)
	plt.savefig('./wgg_k11000.png')
	plt.close()
	
	plt.figure()
	plt.loglog(rp, gammaIA, 'bo')
	plt.savefig('./gammaIA_k11000.png')
	plt.close()
	
	return
	

####### GETTING ERRORS #########

def get_boost(rp_cents_, propfact):
	""" Returns the boost factor in radial bins. propfact is a tunable parameter giving the proportionality constant by which boost goes like projected correlation function (= value at 1 Mpc/h). """

	Boost = (propfact-1.) *(rp_cents_)**(-0.8) + np.ones((len(rp_cents_))) # Empirical power law fit to the boost, derived from the fact that the boost goes like projected correlation function.

	return Boost

def N_of_zph_unweighted(z_a_def, z_b_def, z_a_norm, z_b_norm, z_a_norm_ph, z_b_norm_ph):
	""" Returns dNdz_ph, the number density in terms of photometric redshift, defined and normalized over the photo-z range (z_a_norm_ph, z_b_norm_ph), normalized over the spec-z range (z_a_norm, z_b_norm), but defined on the spec-z range (z_a_def, z_b_def)"""
	
	(z, dNdZ) = get_NofZ_unnormed(pa.alpha, pa.zs, z_a_def, z_b_def, pa.zpts)
	(z_norm, dNdZ_norm) = get_NofZ_unnormed(pa.alpha, pa.zs, z_a_norm, z_b_norm, pa.zpts)
	
	z_ph_vec = scipy.linspace(z_a_norm_ph, z_b_norm_ph, 5000)
	
	int_dzs = np.zeros(len(z_ph_vec))
	int_dzs_norm = np.zeros(len(z_ph_vec))
	for i in range(0,len(z_ph_vec)):
		int_dzs[i] = scipy.integrate.simps(dNdZ*p_z(z_ph_vec[i], z, pa.sigz), z)
		int_dzs_norm[i] = scipy.integrate.simps(dNdZ_norm*p_z(z_ph_vec[i], z_norm, pa.sigz), z_norm)
		
	norm = scipy.integrate.simps(int_dzs_norm, z_ph_vec)
	
	return (z_ph_vec, int_dzs / norm)
	
def N_of_zph_weighted(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph, erms):
	""" Returns dNdz_ph, the number density in terms of photometric redshift, defined over photo-z range (z_a_def_ph, z_b_def_ph), normalized over the photo-z range (z_a_norm_ph, z_b_norm_ph), normalized over the spec-z range (z_a_norm, z_b_norm), and defined on the spec-z range (z_a_def, z_b_def)"""
	
	(z, dNdZ) = get_NofZ_unnormed(pa.alpha, pa.zs, z_a_def_s, z_b_def_s, pa.zpts)
	(z_norm, dNdZ_norm) = get_NofZ_unnormed(pa.alpha, pa.zs, z_a_norm_s, z_b_norm_s, pa.zpts)
	
	z_ph_vec = scipy.linspace(z_a_def_ph, z_b_def_ph, 5000)
	z_ph_vec_norm = scipy.linspace(z_a_norm_ph, z_b_norm_ph, 5000)
	
	weights_ = weights(erms, z_ph_vec)
	weights_norm = weights(erms, z_ph_vec_norm)
	
	int_dzs = np.zeros(len(z_ph_vec))
	for i in range(0,len(z_ph_vec)):
		int_dzs[i] = scipy.integrate.simps(dNdZ*p_z(z_ph_vec[i], z, pa.sigz), z)
	
	int_dzs_norm = np.zeros(len(z_ph_vec_norm))
	for i in range(0,len(z_ph_vec_norm)):
		int_dzs_norm[i] = scipy.integrate.simps(dNdZ_norm*p_z(z_ph_vec_norm[i], z_norm, pa.sigz), z_norm)
		
	norm = scipy.integrate.simps(weights_norm*int_dzs_norm, z_ph_vec_norm)
	
	return (z_ph_vec, weights_*int_dzs / norm)

def N_in_samp(z_a, z_b, e_rms_weights):
	""" Number of galaxies in the photometric redshift range of the sample (assumed z_eff of lenses to z_close) from the SPECTROSCOPIC redshift range z_a to z_b """
	
	(z_ph, N_of_zp) = N_of_zph_weighted(z_a, z_b, pa.zmin, pa.zmax, z_close_low, z_close_high, z_close_low, z_close_high, pa.e_rms_mean)

	answer = scipy.integrate.simps(N_of_zp, z_ph)
	
	return (answer)

def N_corr(rp_cent):
	""" Computes the correction factor which accounts for the fact that some of the galaxies in the photo-z defined source sample are actually higher-z and therefore not expected to be affected by IA"""
	
	N_tot = N_in_samp(pa.zmin, pa.zmax, pa.e_rms_mean)
	N_close = N_in_samp(z_close_low, z_close_high, pa.e_rms_mean)
	boost = get_boost(rp_cent, pa.boost_samp)
	
	Corr_fac = 1. - (1. / boost) * ( 1. - (N_close / N_tot)) # fraction of the galaxies in the source sample which have spec-z in the photo-z range of interest.
	
	return Corr_fac

def N_corr_stat_err(rp_cents_):
	""" Gets the error on N_corr from statistical error on the boost. """
	
	N_tot = N_in_samp(pa.zmin, pa.zmax, pa.e_rms_mean)
	N_close = N_in_samp(z_close_low, z_close_high, pa.e_rms_mean)
	boost = get_boost(rp_cents_, pa.boost_samp)
	
	sig_Ncorr = (pa.sigB / boost**2) * np.sqrt(1. - N_close / N_tot)
	
	return sig_Ncorr

def sigma_e(z_s_, s_to_n):
	""" Returns a value for the model for the per-galaxy noise as a function of source redshift"""

	sig_e = 2. / s_to_n * np.ones(len(z_s_))

	return sig_e

def weights(e_rms, z_):
	""" Returns the inverse variance weights as a function of redshift. """
	
	weights = (1./(sigma_e(z_, pa.S_to_N)**2 + e_rms**2)) * np.ones(len(z_))
	
	return weights
	
def p_z(z_ph, z_sp, sigz):
	""" Returns the probability of finding a photometric redshift z_ph given that the true redshift is z_sp. """
	
	# I'm going to use a Gaussian probability distribution here for now
	p_z_ = np.exp(-(z_ph - z_sp)**2 / (2.*(sigz*(1.+z_sp))**2)) / (np.sqrt(2.*np.pi)*(sigz * (1. + z_sp)))
	
	return p_z_

def get_fid_gIA(rp_bins_c):
	""" This function computes the fiducial value of gamma_IA in each projected radial bin."""
	
	# This is a dummy thing for now.
	#fidvals = np.zeros(len(rp_bins_c))
	#fidvals = pa.A_fid * np.asarray(rp_bins_c)**pa.beta_fid
	
	fidvals = gamma_LA(rp_bins_c)

	return fidvals

def setup_shapenoise_cov(e_rms, N_ls_pbin):
	""" Returns a diagonal covariance matrix in bins of projected radius for a measurement dominated by shape noise. Elements are e_{rms}^2 / (N_ls) where N_ls is the number of l/s pairs relevant to each projected radius bin.""" 

	cov = np.diag(e_rms**2 / N_ls_pbin)
	
	return cov

def get_cov_btw_methods(cov_a_, cov_b_):
	""" Get the covariance between the methods given their correlation """
	
	cov_btw_methods = pa.cov_perc * np.sqrt(np.diag(cov_a_)) * np.sqrt(np.diag(cov_b_))
	
	return cov_btw_methods

def subtract_var(var_1, var_2, covar):
	""" Takes the variance of two non-independent  Gaussian random variables, and their covariance, and returns the variance of their difference."""
	
	var_diff = var_1 + var_2 - 2 * covar

	return var_diff

def get_gammaIA_stat_cov(Cov_1, Cov_2, rp_cents_, gIA_fid):
	""" Takes the covariance matrices of the constituent elements of gamma_{IA} and combines them to get the covariance matrix of gamma_{IA} in projected radial bins."""
	
	# Get the covariance between the shear of the two shape measurement methods in each bin:
	covar = get_cov_btw_methods(Cov_1, Cov_2)
	
	corr_fac = N_corr(rp_cents_)  # factor correcting for galaxies which have higher spec-z than the sample but which end up in the sample.

	corr_fac_err = N_corr_stat_err(rp_cents_) # statistical error on that from the boost

	stat_mat = np.diag(np.zeros(len(np.diag(Cov_1))))

	for i in range(0,len(np.diag(Cov_1))):	
		stat_mat[i, i] = (1.-pa.a_con)**2 * gIA_fid[i]**2 * ( corr_fac_err[i]**2 / corr_fac[i]**2 + subtract_var(Cov_1[i,i], Cov_2[i,i], covar[i]) / (corr_fac[i]**2 * (1.-pa.a_con)**2 * gIA_fid[i]**2))
		
	print "stat=", np.sqrt(np.diag(stat_mat))

	return stat_mat

def get_gammaIA_sys_cov(rp_cents_, sys_dN, sys_p, gIa_fid):
	""" Takes the centers of rp_bins and a systematic error sources from dNdz_s uncertainty (assumed to affect each r_p bin in the same way) and adds them to each other in quadrature."""
	
	corr_fac = N_corr(rp_cents_) # correction factor
	
	
	sys_mat = np.zeros((len(rp_cents_), len(rp_cents_)))
	
	for i in range(0,len(rp_cents_)):
		for j in range(0,len(rp_cents_)):
			sys_mat[i,j] = sys_dN[i]*sys_dN[j] * (1-pa.a_con)**2 * corr_fac[i] * corr_fac[j] * gIa_fid[i] * gIa_fid[j] + sys_p[i]*sys_p[j] * (1-pa.a_con)**2 * corr_fac[i] * corr_fac[j] * gIa_fid[i] * gIa_fid[j]
			
	print "sys=", np.sqrt(np.diag(sys_mat))

	return sys_mat
	
def get_gamma_tot_cov(sys_mat, stat_mat):
	""" Takes the covariance matrix from statistical error and systematic error, and adds them to get the total covariance matrix. Assumes stat and sys errors should be added in quadrature."""
	
	tot_cov = sys_mat+stat_mat
	
	print "tot=", np.sqrt(np.diag(tot_cov))
	
	return tot_cov

####### PLOTTING / OUTPUT #######

def plot_variance(cov_1, fidvalues_1, bin_centers, filename):
	""" Takes a covariance matrix, a vector of the fiducial values of the object in question, and the edges of the projected radial bins, and makes a plot showing the fiducial values and 1-sigma error bars from the diagonal of the covariance matrix. Outputs this plot to location 'filename'."""

	fig=plt.figure()
	plt.rc('font', family='serif', size=20)
	fig_sub=fig.add_subplot(111) #, aspect='equal')
	fig_sub.set_xscale("log")
	#fig_sub.set_yscale("log")
	fig_sub.errorbar(bin_centers,fidvalues_1*(1-pa.a_con), yerr = np.sqrt(np.diag(cov_1)), fmt='o')
	fig_sub.set_xlabel('$r_p$')
	fig_sub.set_ylabel('$\gamma_{IA}(1-a)$')
	fig_sub.set_ylim(-0.012, 0.012)
	fig_sub.tick_params(axis='both', which='major', labelsize=12)
	fig_sub.tick_params(axis='both', which='minor', labelsize=12)
	plt.tight_layout()
	plt.savefig(filename)

	return  

def plot_quant_vs_rp(quant, rp_cent, file):
	""" Plots any quantity vs the center of redshift bins"""

	plt.figure()
	plt.loglog(rp_cent, quant, 'ko')
	plt.xlabel('$r_p$')
	plt.tick_params(axis='both', which='major', labelsize=12)
	plt.tick_params(axis='both', which='minor', labelsize=12)
	plt.tight_layout()
	plt.savefig(file)

	return
	
def plot_quant_vs_quant(quant1, quant2, file):
	""" Plots any quantity vs any other."""

	plt.figure()
	plt.loglog(quant1, quant2, 'ko')
	plt.tick_params(axis='both', which='major', labelsize=12)
	plt.tick_params(axis='both', which='minor', labelsize=12)
	plt.tight_layout()
	plt.savefig(file)

	return

######### FISHER STUFF ##########

def par_derivs(params, rp_mid):
        """ Computes the derivatives of gamma_IA wrt the parameters of the IA model we care about constraining.
        Returns a matrix of dimensions (# r_p bins, # parameters)."""

	n_bins = len(rp_mid)

        derivs = np.zeros((n_bins, len(params)))

        # This is for a power law model gamma_IA = A * rp ** beta

        derivs[:, pa.A] = rp_mid**(pa.beta_fid)

        derivs[:, pa.beta] = pa.beta_fid * pa.A_fid * rp_mid**(pa.beta_fid-1) 

        return derivs

def get_Fisher(p_derivs, dat_cov):     
        """ Constructs the Fisher matrix, given a matrix of derivatives wrt parameters in each r_p bin and the data covariance matrix."""

	inv_dat_cov = np.linalg.inv(dat_cov)

        Fish = np.zeros((len(p_derivs[0,:]), len(p_derivs[0, :])))
        for a in range(0,len(p_derivs[0,:])):
                for b in range(0,len(p_derivs[0,:])):
                        Fish[a,b] = np.dot(p_derivs[:,a], np.dot(inv_dat_cov, p_derivs[:,b]))
        return Fish

def cut_Fisher(Fish, par_ignore ):
        """ Cuts the Fisher matrix to ignore any parameters we want to ignore. (par_ignore is a list of parameter names as defed in input file."""

        if (par_ignore!=None):
                Fish_cut = np.delete(np.delete(Fish, par_ignore, 0), par_ignore, 1)
        else:
                Fish_cut = Fish

        return Fish_cut

def get_par_Cov(Fish_, par_marg):
        """ Takes a Fisher matrix and returns a parameter covariance matrix, cutting out (after inversion) any parameters which will be marginalized over. Par_marg should either be None, if no parameters to be marginalised, or a list of parameters to be marginalised over by name from the input file."""

        par_Cov = np.linalg.inv(Fish_)

        if (par_marg != None):
                par_Cov_marg = np.delete(np.delete(par_Cov, par_marg, 0), par_marg, 1)
        else:
                par_Cov_marg = par_Cov

        return par_Cov_marg

def par_const_output(Fish_, par_Cov):
        """ Gunction to output  whatever information about parameters constaints we want given the parameter covariance matrix. The user should modify this function as desired."""

        # Put whatever you want to output here 

        print "1-sigma constraint on A=", np.sqrt(par_Cov[pa.A, pa.A])

        print "1-sigma constraint on beta=", np.sqrt(par_Cov[pa.beta, pa.beta])

        return


######## MAIN CALLS ##########

# Import the parameter file:
import IA_params_shapes as pa

# Set up projected radial bins
rp_bins 	= 	setup_rp_bins(pa.rp_min, pa.rp_max, pa.N_bins) # Edges
rp_cents	=	rp_bins_mid(rp_bins) # Centers

# Set up to get z as a function of comoving distance
z_of_com 	= 	z_interpof_com()

# Get the redshift corresponding to the maximum separation from the effective lens redshift at which we assume IA may be present (pa.close_cut is the separation in Mpc/h)
(z_close_high, z_close_low)	= 	get_z_close(pa.zeff, pa.close_cut)

gamma_fid(rp_cents)
exit()

# Get the number of lens source pairs for the source sample in projected radial bins
N_ls_pbin	=	get_perbin_N_ls(rp_bins, pa.zeff, pa.n_s, pa.n_l, pa.Area, rp_cents)

# Get the fiducial value of gamma_IA in each projected radial bin (this takes a while so only do it once
fid_gIA		=	get_fid_gIA(rp_cents)

# Get the covariance matrix in projected radial bins of gamma_t for both shape measurement methods
Cov_a		=	setup_shapenoise_cov(pa.e_rms_a, N_ls_pbin)
Cov_b		=	setup_shapenoise_cov(pa.e_rms_b, N_ls_pbin)

# Combine the constituent covariance matrices to get the covariance matrix for gamma_IA in projected radial bins
Cov_stat	=	get_gammaIA_stat_cov(Cov_a, Cov_b, rp_cents, fid_gIA) 
Cov_sys 	=	get_gammaIA_sys_cov(rp_cents, pa.sig_sys_dNdz, pa.sig_sys_dp, fid_gIA)

Cov_tot		=	get_gamma_tot_cov(Cov_sys, Cov_stat)

# Output a plot showing the 1-sigma error bars on gamma_IA in projected radial bins
plot_variance(Cov_tot, fid_gIA, rp_cents, pa.plotfile)

exit() # Below is Fisher stuff, don't worry about this yet

# Get the parameter derivatives required to construct the Fisher matrix
ders            =       par_derivs(pa.par, rp_cents)

# Get the Fisher matrix
fish            =       get_Fisher(ders, Cov_gIA)

# If desired, cut parameters which you want to fix from Fisher matrix:
fish_cut        =       cut_Fisher(fish, None)

# Get the covariance matrix from either fish or fish_cut, and marginalise over any desired parameters
parCov          =       get_par_Cov(fish_cut, None)

# Output whatever we want to know about the parameters:
par_const_output(fish_cut, parCov)
