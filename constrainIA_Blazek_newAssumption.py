# This is a script which predicts constraints on intrinsic alignments, using an updated version of the method of Blazek et al 2012 in which it is not assumed that only excess galaxies contribute to IA (rather it is assumed that source galaxies which are close to the lens along the line-of-sight can contribute.)

import numpy as np
import IA_params_Fisher as pa
import scipy
import scipy.integrate
import matplotlib.pyplot as plt
import scipy.interpolate


########## FUNCTIONS ##########

# SET UP PROJECTED RADIUS, DISTANCE, AND REDSHIFT #

def setup_rp_bins(rmin, rmax, nbins):
	""" Sets up the bins of projected radius (in units Mpc/h) """

	bins = scipy.logspace(np.log10(rmin), np.log10(rmax), nbins+1)

	return bins

def rp_bins_mid(rp_edges):
	""" Gets the middle of each projected radius bin."""

	logedges=np.log10(rp_edges)
	bin_centers=np.zeros(len(rp_edges)-1)
	for ri in range(0,len(rp_edges)-1):
		bin_centers[ri]    =       10**((logedges[ri+1] - logedges[ri])/2. +logedges[ri])

	return bin_centers

def get_z_close(z_l, cut_MPc_h):
	""" Gets the z above z_l which is the highest z at which we expect IA to be present for that lens. cut_Mpc_h is that separation in Mpc/h."""

	com_l = com(z_l) # Comoving distance to z_l, in Mpc/h
	tot_com = com_l + cut_MPc_h
	z_cl = z_of_com(tot_com) # Convert tot_com back to a redshift.
	
	return z_cl

def com(z_):
	""" Gets the comoving distance in units of Mpc/h at a given redshift, z_ (assuming the cosmology defined in the params file. """

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

	z_vec = scipy.linspace(0., 10., 10000) # This hardcodes that we don't care about anything over z=2100

	com_vec = com(z_vec)

	z_of_com = scipy.interpolate.interp1d(com_vec, z_vec)

	return	z_of_com

def average_in_bins(F_, R_, Rp_):
	""" This function takes a function F_ of projected radius evaluated at projected radial points R_ and outputs the averaged values in bins with edges Rp_""" 
	
	F_binned = np.zeros(len(Rp_)-1)
	for iR in range(0, len(Rp_)-1):
		indlow=next(j[0] for j in enumerate(R_) if j[1]>=(Rp_[iR]))
		indhigh=next(j[0] for j in enumerate(R_) if j[1]>=(Rp_[iR+1]))
		
		F_binned[iR] = scipy.integrate.simps(F_[indlow:indhigh], R_[indlow:indhigh]) / (R_[indhigh] - R_[indlow])
		
	return F_binned

	
############## THINGS TO DO WITH DNDZ ###############

def get_areas(bins, z_eff):
	""" Gets the area of each projected radial bin, in square arcminutes. z_eff = effective lens redshift. """	

	# Areas in units (Mpc/h)^2
	areas_mpch = np.zeros(len(bins)-1)
	for i in range(0, len(bins)-1):
		areas_mpch[i] = np.pi * (bins[i+1]**2 - bins[i]**2) 

	#Comoving distance out to effective lens redshift in Mpc/h
	chi_eff = com(z_eff)

	# Areas in square arcminutes (466560000 / pi = sqAM in a sphere)
	areas_sqAM = areas_mpch * (466560000. / np.pi) / (4 * np.pi * chi_eff**2)

	return areas_sqAM

def get_NofZ_unnormed(a, zs, z_min, z_max, zpts):
	""" Returns the dNdz of the sources as a function of photometric redshift, as well as the z points at which it is evaluated."""

	z = scipy.linspace(z_min+0.0001, z_max, zpts)
	nofz_ = (z / zs)**(a-1) * np.exp( -0.5 * (z / zs)**2)	

	return (z, nofz_)
	

def N_of_zs(z_a_def, z_b_def, z_a_norm, z_b_norm):
	""" Returns dNdz_s, the number density in terms of spectroscopic redshift, normalized over the spec-z range (z_a_norm, z_b_norm), and defined on the spec-z range (z_a_def, z_b_def)"""
	
	(z, dNdZ) = get_NofZ_unnormed(pa.alpha, pa.zs, z_a_def, z_b_def, pa.zpts)
	(z_norm, dNdZ_norm) = get_NofZ_unnormed(pa.alpha, pa.zs, z_a_norm, z_b_norm, pa.zpts)
		
	norm = scipy.integrate.simps(dNdZ_norm, z_norm)
	
	return (z, dNdZ / norm)
	
def N_of_zph(z_a_def_s, z_b_def_s, z_a_norm_s, z_b_norm_s, z_a_def_ph, z_b_def_ph, z_a_norm_ph, z_b_norm_ph):
	""" Returns dNdz_ph, the number density in terms of photometric redshift, defined over photo-z range (z_a_def_ph, z_b_def_ph), normalized over the photo-z range (z_a_norm_ph, z_b_norm_ph), normalized over the spec-z range (z_a_norm, z_b_norm), and defined on the spec-z range (z_a_def, z_b_def)"""
	
	(z, dNdZ) = get_NofZ_unnormed(pa.alpha, pa.zs, z_a_def_s, z_b_def_s, pa.zpts)
	(z_norm, dNdZ_norm) = get_NofZ_unnormed(pa.alpha, pa.zs, z_a_norm_s, z_b_norm_s, pa.zpts)
	
	z_ph_vec = scipy.linspace(z_a_def_ph, z_b_def_ph, 5000)
	z_ph_vec_norm = scipy.linspace(z_a_norm_ph, z_b_norm_ph, 5000)
	
	int_dzs = np.zeros(len(z_ph_vec))
	for i in range(0,len(z_ph_vec)):
		int_dzs[i] = scipy.integrate.simps(dNdZ*p_z(z_ph_vec[i], z, pa.sigz), z)
	
	int_dzs_norm = np.zeros(len(z_ph_vec_norm))
	for i in range(0,len(z_ph_vec_norm)):
		int_dzs_norm[i] = scipy.integrate.simps(dNdZ_norm*p_z(z_ph_vec_norm[i], z_norm, pa.sigz), z_norm)
		
	norm = scipy.integrate.simps(int_dzs_norm, z_ph_vec_norm)
	
	return (z_ph_vec, int_dzs / norm)

def get_z_frac(z_1, z_2):
	""" Gets the fraction of sources in a sample between z_1 and z_2, for dNdz given by a normalized nofz_1 computed at redshifts z_v"""

	# The normalization factor should be the norm over the whole range of z for which the number density of the survey is given, i.e. z min to zmax.       
	(z_ph_vec, N_of_p) = N_of_zph(pa.zmin, pa.zmax, pa.zmin, pa.zmax, z_1, z_2, pa.zphmin, pa.zphmax)
	
	frac = scipy.integrate.simps(N_of_p, z_ph_vec)

	return frac

def get_perbin_N_ls(rp_bins_, zeff_, ns_, nl_, zbinmin, zbinmax, A):
	""" Gets the number of lens/source pairs relevant to each bin of projected radius """
	""" zeff_ is the effective redshift of the lenses. frac_ is the fraction of sources in the sample. ns_ is the number density of sources per square arcminute. nl_ is the number density of lenses per square degree. A is the survey area in square degrees."""
	
	frac_ = get_z_frac(zbinmin, zbinmax)

	# Get the area of each projected bin in square arcminutes
	bin_areas       =       get_areas(rp_bins_, zeff_)

	N_ls_pbin = nl_ * A * ns_ * bin_areas * frac_

	return N_ls_pbin

def p_z(z_ph, z_sp, sigz):
	""" Returns the probability of finding a photometric redshift z_ph given that the true redshift is z_sp. """
	
	# I'm going to use a Gaussian probability distribution here, but you could change that.
	p_z_ = np.exp(-(z_ph - z_sp)**2 / (2.*(sigz*(1.+z_sp))**2)) / (np.sqrt(2.*np.pi)*(sigz*(1.+z_sp)))
	
	return p_z_

################### THEORETICAL VALUES FOR FRACTIONAL ERROR CALCULATION ########################333

def sigma_e(z_s_, s_to_n):
	""" Returns a value for the model for the per-galaxy noise as a function of source redshift"""

	# This is a dummy things for now
	sig_e = 2. / s_to_n * np.ones(len(z_s_))

	return sig_e

def weights(e_rms, z_, z_l_):
	""" Returns the inverse variance weights as a function of redshift. """
	
	SigC_t = get_SigmaC_theory(z_, z_l_)
	
	weights = 1./(SigC_t**2*(sigma_e(z_, pa.S_to_N)**2 + e_rms**2 * np.ones(len(z_))))
	
	return weights

def sum_weights(z_l, z_min_s, z_max_s, erms, rp_bins_, rp_bin_c_):
	""" Returns the sum over rand-source pairs of the estimated weights, in each projected radial bin. Pass different z_min_s and z_max_s to get rand-close, rand-far, and all-rand cases."""
	
	(z_ph, dNdz_ph) = N_of_zph(pa.zmin, pa.zmax, pa.zmin, pa.zmax, z_min_s, z_max_s, pa.zphmin, pa.zphmax)

	sum_ans = [0]*(len(rp_bins_)-1)
	for i in range(0,len(rp_bins_)-1):
		Integrand = dNdz_ph* weights(erms, z_ph, z_l)
		sum_ans[i] = scipy.integrate.simps(Integrand, z_ph)

	return sum_ans

def get_bSigma(z_s_, z_l_):
	
	""" Returns the photo-z bias to the estimated critical surface density. In principle this is a model fit from the spectroscopic subsample of data. """

	# This is a dummy return value for now
	b_Sig = 1. * np.ones(len(z_s_))

	return b_Sig

def get_SigmaC_theory(z_s_, z_l_):
	""" Returns the theoretical value of Sigma_c, the critcial surface mass density """

	com_s = com(z_s_) 
	com_l = com(z_l_) 

	a_l = 1. / (z_l_ + 1.)

	a_s = 1. / (z_s_ + 1.)

	# This is missing a factor of c^2 / 4piG - I'm hoping this cancels everywhere? Check.
	Sigma_c = (a_l * a_s * com_s) / ((a_s*com_s - a_l * com_l) * com_l)

	return Sigma_c

def get_boost(rp_cents_, propfact):
	""" Returns the boost factor in radial bins. propfact is a tunable parameter giving the proportionality constant by which boost goes like projected correlation function (= value at 1 Mpc/h). """

	Boost = propfact *(rp_cents_)**(-0.8) + np.ones((len(rp_cents_))) # Empirical power law fit to the boost, derived from the fact that the boost goes like projected correlation function.

	plot_quant_vs_rp(Boost, rp_cents_, './boost.png')

	return Boost

def get_F(z_l, z_close_max, z_max_samp, erms, rp_bins_, rp_bin_c):
	""" Returns F (the weighted fraction of lens-source pairs from the smooth dNdz which are contributing to IA) """
	
	print "z_close_max=", z_close_max
	print "z_max_sampl=", z_max_samp

	# Sum over `rand-close'
	numerator = sum_weights(z_l, z_l, z_close_max, erms, rp_bins_, rp_bin_c)

	#Sum over all `rand'
	denominator = sum_weights(z_l, z_l, z_max_samp, erms, rp_bins_, rp_bin_c)

	F = np.asarray(numerator) / np.asarray(denominator)
	
	print "F=", F

	#plot_quant_vs_rp(F, rp_bin_c, './F.png')

	return F

def get_cz(z_l, z_max_samp, erms, rp_bins_, rp_bins_c):
	""" Returns the value of the photo-z bias parameter c_z"""

	# The denominator is just a sum over weights of all random-source pairs
	denominator = sum_weights(z_l, z_l, z_max_samp, erms, rp_bins_, rp_bins_c)

	# The numerator is very similar but with different factors of the photometric bias to Sigma_C:
	(z_s, dNdz_s) = get_NofZ(pa.alpha, pa.zs, z_l, z_max_samp, pa.zpts)

        SigC_t = get_SigmaC_theory(z_s, z_l)

        b_Sig = get_bSigma(z_s, z_l)

        sig_e = sigma_e(z_s, pa.S_to_N)

        sum_ans = [0]*(len(rp_bins_)-1)
        for i in range(0,len(rp_bins_)-1):
                Integrand = dNdz_s  / (SigC_t**2 * b_Sig * (erms**2*np.ones(len(z_s)) + sig_e**2))
                sum_ans[i] = scipy.integrate.simps(Integrand, z_s)

	numerator = sum_ans

	cz = np.asarray(numerator) / np.asarray(denominator)

	plot_quant_vs_rp(cz, rp_bins_c, './cz.png')

	return cz

def get_Sig_IA(z_l, z_min_s, z_cut_IA, z_max_samp, erms, rp_bins_, rp_bin_c_, boost):
	""" Returns the value of <\Sigma_c>_{IA} in radial bins """
	
	# There are four terms here. The two in the denominators are sums over randoms (or sums over lenses that can be written as randoms * boost), and these are already set up to calculate.
	denom_rand_far = sum_weights(z_l, z_cut_IA, z_max_samp, erms, rp_bins_, rp_bin_c_)
	denom_rand = sum_weights(z_l, z_l, z_max_samp, erms, rp_bins_, rp_bin_c_)
	
	# The two in the numerator require summing over weights and Sigma_C. 
	
	#For the sum over rand-far, this follows directly from the same type of expression as when summing weights:
	(z_s, dNdz_s) = get_NofZ(pa.alpha, pa.zs, z_cut_IA, z_max_samp, pa.zpts)
	SigC_t = get_SigmaC_theory(z_s,	z_l)
	b_Sig = get_bSigma(z_s, z_l)
	sig_e = sigma_e(z_s, pa.S_to_N)
	
	rand_far_num = [0]*(len(rp_bins_)-1)
	for i in range(0,len(rp_bins_)-1):
		Integrand = dNdz_s  / (SigC_t * b_Sig * (erms**2*np.ones(len(z_s)) + sig_e**2))
		#print "Integrand, far=", Integrand
		rand_far_num[i] = scipy.integrate.simps(Integrand, z_s)
			
	# The other numerator sum is itself the sum of (a sum over all randoms) and (a term which represents the a sum over excess). 
	
	# The rand part:
	(z_s_all, dNdz_s_all) = get_NofZ(pa.alpha, pa.zs, z_l, z_max_samp, pa.zpts)
	SigC_t_all = get_SigmaC_theory(z_s_all,	z_l)
	b_Sig_all = get_bSigma(z_s_all, z_l)
	sig_e_all = sigma_e(z_s_all, pa.S_to_N)
	
	rand_all_num = [0]*(len(rp_bins_)-1)
	for i in range(0,len(rp_bins_)-1):
		Integrand = dNdz_s_all  / (SigC_t_all * b_Sig_all * (erms**2*np.ones(len(z_s_all)) + sig_e_all**2))
		#print "Integrand, all=", Integrand
		rand_all_num[i] = scipy.integrate.simps(Integrand, z_s_all)
	
	#The excess part:
	excess_num = [0]*(len(rp_bins_)-1)	
	for i in range(0, len(rp_bins_)-1):
			p_z_arr = p_z(z_s_all, z_l, pa.sigz)
			Integrand = p_z_arr / (SigC_t_all * (sig_e_all**2 + erms**2 * np.ones(len(z_s_all))))
			excess_num[i] = scipy.integrate.simps(Integrand, z_s_all)
		
	Sig_IA = ((np.asarray(boost)-1.)*np.asarray(excess_num) + np.asarray(rand_all_num) - np.asarray(rand_far_num)) / (np.asarray(boost)*np.asarray(denom_rand) - np.asarray(denom_rand_far))
	
	plot_quant_vs_rp(Sig_IA, rp_bin_c_, './Sig_IA')

	return Sig_IA

def get_est_DeltaSig(z_l, z_max_samp, erms, rp_bins, rp_bins_c, boost):
	""" Returns the value of tilde Delta Sigma in bins"""
	
	# This has numerous parts to it. 
	
	# The first term is (1 + b_z) \Delta \Sigma ^{theory}
	
	cz = get_cz(z_l, z_max_samp, erms, rp_bins, rp_bins_c)
	
	print "cz=", cz
	
	DS_the = get_DeltaSig_theory(z_l, rp_bins, rp_bins_c)
	
	# The second term is a fraction of sums. The denominator is a sum of weights over rand-source pairs; we know how to get this:
	denom_rand = sum_weights(z_l, z_l, z_max_samp, erms, rp_bins, rp_bins_c)
	
	# The first term is a sum over lens-source pairs, over weights, Sigma C, and gamma_IA. It is computed as a sum over randoms plus an excess term.
	
	# Get the fiducial value of gamma_IA in bins:
	g_IA_fid = get_fid_gIA(rp_bins_c)
	
	# First the randoms term. Note that we assume gamma_IA is independent of source and lens redshift.
	(z_s_all, dNdz_s_all) = get_NofZ(pa.alpha, pa.zs, z_l, z_max_samp, pa.zpts)
	SigC_t_all = get_SigmaC_theory(z_s_all,	z_l)
	b_Sig_all = get_bSigma(z_s_all, z_l)
	sig_e_all = sigma_e(z_s_all, pa.S_to_N)
	
	rand_all_num = [0]*(len(rp_bins)-1)
	for i in range(0,len(rp_bins)-1):
		Integrand = dNdz_s_all  / (SigC_t_all * b_Sig_all * (erms**2*np.ones(len(z_s_all)) + sig_e_all**2))
		rand_all_num[i] = scipy.integrate.simps(Integrand, z_s_all) * g_IA_fid[i]
		
	# Now get the excess term
	excess_num = [0]*(len(rp_bins)-1)
	for i in range(0,len(rp_bins)-1):
		Integrand = p_z(z_s_all, z_l, pa.sigz)/ (SigC_t_all * (sig_e_all**2 + erms**2 * np.ones(len(z_s_all))))
		excess_num[i] = scipy.integrate.simps(Integrand, z_s_all) *g_IA_fid[i]
		
	print "DS_the=", np.asarray(DS_the)
	print "cz=", np.asarray(cz)
	print "boost=", np.asarray(boost)
	print "excess num=", np.asarray(excess_num)
	print "rand all num=", np.asarray(rand_all_num)
	print "denom rand=", np.asarray(denom_rand)
		
	EstDeltaSig = np.asarray(DS_the) / np.asarray(cz) + (np.asarray(boost) * np.asarray(excess_num) + np.asarray(rand_all_num) ) / (np.asarray(denom_rand))
	
	first_term = np.asarray(DS_the) / np.asarray(cz)
	second_term = (np.asarray(boost) * np.asarray(excess_num) + np.asarray(rand_all_num) ) / (np.asarray(denom_rand))
	
	plot_quant_vs_rp(EstDeltaSig, rp_bins_c, './EstDeltaSig.png')
	plot_quant_vs_rp(first_term, rp_bins_c, './Firstterm_estDeltaSigma.png')
	plot_quant_vs_rp(second_term, rp_bins_c, './Secondterm_estDeltaSigma.png')

	return EstDeltaSig

def Rhalo(M_insol, z):
	""" Get the radius of a halo in Mpc/h given its mass IN SOLAR MASSES. Uses the Rvir / Mvir relationship from Giocolo 2010, which employs Delta_virial, from Eke et al 1996."""
	
	OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
	E_ofz = ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )**(0.5)#the dimensionless part of the hubble parameter 
	rho_crit = 4.126 * 10** 11 * E_ofz**2 # This is rho_crit in units of Msol h^3 / Mpc^3
	OmM = (pa.OmC+pa.OmB)*(1+z)**3 / ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )
	#Dv = Delta_virial(z)
	
	#Rvir = ( 3. * M_insol * OmM / (4. * np.pi * rho_crit * Dv))**(1./3.)
	Rvir = ( 3. * M_insol / (4. * np.pi * rho_crit * OmM * 180.))**(1./3.)
	
	print "Rvir= ", Rvir
	
	return Rvir

def cvir(M_insol, z):
	""" Returns the concentration parameter of the NFW profile, c_{vir}. Uses the Neto 2007 definition, referenced in Giocoli 2010 """
	
	#cvi = pa.c14 / (1. + z) * (M_insol / 10**14)**(-0.11)
	
	cvi = 7.
	
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
	plt.savefig('./plots/nfw_0605_6e13h_inBlazek.png')
	plt.close()
	
	return rho_nfw

def get_DeltaSig_theory_update(z_l, rp_bins, rp_bins_c):
	""" Returns the theoretical value of Delta Sigma in bin using projection over the NFW profile and over the 2-pt correlation function at larger scales, rather than using the lensing-related definition."""
	
	###### First get the term from halofit (valid at larger scales) ######
	# Import the correlation function as a function of R and Pi, obtained via getting P(k) from CAMB and then using FFT_log, Anze Slozar version. 
	corr = np.loadtxt('./txtfiles/corr_2d_z='+str(z_l)+'_kmax=4000_HF.txt')
	rpvec = np.loadtxt('./txtfiles/corr_rp_z='+str(z_l)+'_kmax=4000.txt')
	Pivec = np.loadtxt('./txtfiles/corr_delta_z='+str(z_l)+'_kmax=4000.txt') 
	
	# Get Sigma(R)
	OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
	E_ofz = ( (pa.OmC+pa.OmB)*(1+z_l)**3 + OmL + (pa.OmR+pa.OmN) * (1+z_l)**4 )**(0.5)#the dimensionless part of the hubble parameter 
	rho_crit = 4.126 * 10** 11 * E_ofz**2 # This is rho_crit in units of Msol h^3 / Mpc^3
	OmM = (pa.OmC + pa.OmB)* (1. + z_l)**(3) / ( (pa.OmC+pa.OmB)*(1+z_l)**3 + OmL + (pa.OmR+pa.OmN) * (1+z_l)**4 )
	rho_m = OmM * rho_crit # units of Msol h^3 / Mpc^3
	
	Sigma_HF = np.zeros(len(rpvec))
	for ri in range(0,len(rpvec)):
		Sigma_HF[ri] = rho_m * scipy.integrate.simps(corr[:, ri], Pivec)
	
	plt.figure()
	plt.loglog(rpvec, Sigma_HF * (pa.HH0 / 100.) / (10**12), 'r+')
	plt.xlim(0.0003,8)
	plt.ylim(0.3,20000)
	plt.savefig('./plots/test_SigmaHF_new.png')
	plt.close()
	
	# Now average over R to get the first averaged term in Delta Sigma
	barSigma_HF = np.zeros(len(rpvec))
	for ri in range(0,len(rpvec)):
		barSigma_HF[ri] = 2. / rpvec[ri]**2 * scipy.integrate.simps(rpvec[0:ri+1]**2*Sigma_HF[0:ri+1], np.log(rpvec[0:ri+1]))

	plt.figure()
	plt.loglog(rpvec, barSigma_HF * (pa.HH0 / 100.) / (10**12), 'r+')
	plt.xlim(0.0003,8)
	plt.ylim(0.3,20000)
	plt.savefig('./plots/test_barSigmaHF_new.png')
	plt.close()
	
	DeltaSigma_HF = barSigma_HF - Sigma_HF
	
	plt.figure()
	plt.loglog(rpvec, DeltaSigma_HF * (pa.HH0 / 100.) / (10**12), 'r+')
	plt.xlim(0.0003,8)
	plt.ylim(0.3,20000)
	plt.savefig('./plots/test_DeltaSigmaHF_new.png')
	plt.close()
	
	####### Now get the 1 halo term (valid at smaller scales) #######
	
	# Get the r (unprojected) vector at which to get the NFW profile
	(rvec, notneeded) = np.loadtxt('./txtfiles/corr_bothterms_z='+str(z_l)+'_HF.txt', unpack=True)
	
	rho = rho_NFW(rvec, pa.Mvir, z_l)
	
	rho_interp = scipy.interpolate.interp1d(rvec, rho)

	rho_2D = np.zeros((len(rpvec), len(Pivec)))
	for ri in range(0, len(rpvec)):
		for pi in range(0, len(Pivec)):
			rho_2D[ri, pi] = rho_interp(np.sqrt(rpvec[ri]**2 + Pivec[pi]**2))
	
	Sigma_1h = np.zeros(len(rpvec))
	for ri in range(0,len(rpvec)):
		Sigma_1h[ri] = scipy.integrate.simps(rho_2D[ri, :], Pivec)
		
	plt.figure()
	plt.loglog(rpvec, Sigma_1h * (pa.HH0 / 100.) / (10**12), 'r+')
	plt.xlim(0.0003,8)
	plt.ylim(0.3,20000)
	plt.savefig('./plots/test_Sigma1h_new.png')
	plt.close()
	
	# Now average over R to get the first averaged term in Delta Sigma
	barSigma_1h = np.zeros(len(rpvec))
	for ri in range(0,len(rpvec)):
		barSigma_1h[ri] = 2. / rpvec[ri]**2 * scipy.integrate.simps(rpvec[0:ri+1]**2*Sigma_1h[0:ri+1], np.log(rpvec[0:ri+1]))
	
	plt.figure()
	plt.loglog(rpvec, barSigma_1h * (pa.HH0 / 100.) / (10**12), 'r+')
	plt.xlim(0.0003,8)
	plt.ylim(0.3,20000)
	plt.savefig('./plots/test_barSigma1h_new.png')
	plt.close()
	
	DeltaSigma_1h = barSigma_1h - Sigma_1h
	
	plt.figure()
	plt.loglog(rpvec, DeltaSigma_1h * (pa.HH0 / 100.) / (10**12), 'r+')
	plt.xlim(0.0003,8)
	plt.ylim(0.3,20000)
	plt.savefig('./plots/test_DeltaSigma1h_new.png')
	plt.close()
	
	plt.figure()
	plt.loglog(rpvec, DeltaSigma_1h * (pa.HH0 / 100.) / (10**12), 'g+', label='1-halo')
	plt.hold(True)
	plt.loglog(rpvec, DeltaSigma_HF * (pa.HH0 / 100.) / (10**12), 'm+', label='halofit')
	plt.hold(True)
	plt.loglog(rpvec, (DeltaSigma_HF + DeltaSigma_1h) * (pa.HH0 / 100.) / (10**12), 'k+', label='total')
	plt.xlim(0.05,20)
	plt.ylim(0.3,200)
	plt.xlabel('$r_p$, Mpc/h')
	plt.ylabel('$\Delta \Sigma$, $h M_\odot / pc^2$')
	plt.legend()
	plt.savefig('./plots/test_DeltaSigmatot_new.png')
	plt.close()
	
	return

def get_DeltaSig_theory(z_l, rp_bins, rp_bins_c):
	""" Returns the theoretical value of Delta Sigma in bins. This Delta Sigma has dimensions of 1 / length, to correspond with leaving the factor of c^2 / 4piG off Sigma_c. """
	
	# The theoretical value for Delta Sigma requires the correlation function at the lens redshift, which we calculate elsewhere and import.
	corr = np.loadtxt('./txtfiles/corr_2d_z='+str(z_l)+'_kmax=4000_M6e13h_fixRhoC.txt')
	Rint = np.loadtxt('./txtfiles/Rint_z='+str(z_l)+'_kmax=4000_M6e13h_fixRhoC.txt')
	corrterm = Rint - corr
	# Load also the vectors in projected radius and line-of-sight distance (Delta) along which the correlation funtion is calculated
	rpvec = np.loadtxt('./txtfiles/corr_rp_z='+str(z_l)+'_kmax=4000.txt')
	deltavec = np.loadtxt('./txtfiles/corr_delta_z='+str(z_l)+'_kmax=4000.txt') 
	
	#print "Rint=", Rint
	
	#print "len(deltavec)=", len(deltavec)
	#print "len(Rint)=", len(Rint)
	#print "Rint=", Rint
	
	
	# Test the behaviour of these things we have imported 
	r_2d_test = np.zeros(len(rpvec)*len(deltavec))
	corr_1d_fr_2d = np.zeros(len(rpvec)*len(deltavec))
	i=0
	for di in range(0, len(deltavec)):
		for ri in range(0,len(rpvec)):
			r_2d_test[i] = (deltavec[di]**2+rpvec[ri]**2)**(0.5)
			corr_1d_fr_2d[i] = corr[di, ri]
			i=i+1
			
	plt.figure()
	plt.loglog(r_2d_test, corr_1d_fr_2d , '+')
	plt.ylim(0.01, 3000)
	plt.xlim(0.05, 100)
	plt.savefig('./plots/test_corrfunc_insideDeltaSigma_6e13h_fixRhoC.png')
	plt.close()
	
	Rint_proj = np.zeros(len(rpvec))
	proj_corr = np.zeros(len(rpvec))
	for ri in range(0,len(rpvec)):
			Rint_proj[ri] = scipy.integrate.simps(Rint[:,ri], deltavec)
			proj_corr[ri] = scipy.integrate.simps(corr[:,ri], deltavec)
			#print "ri=", ri, "Rint proj=", Rint_proj[ri], "proj_corr=", proj_corr[ri]
			
	OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
	E_ofz = ( (pa.OmC+pa.OmB)*(1+z_l)**3 + OmL + (pa.OmR+pa.OmN) * (1+z_l)**4 )**(0.5)#the dimensionless part of the hubble parameter 
	rho_crit = 4.126 * 10** 11 * E_ofz**2 # This is rho_crit in units of Msol h^3 / Mpc^3
	#print "rho_crit=", rho_crit
	#print "E_ofz=", E_ofz
	OmM = (pa.OmC + pa.OmB)* (1. + z_l)**(3) / ( (pa.OmC+pa.OmB)*(1+z_l)**3 + OmL + (pa.OmR+pa.OmN) * (1+z_l)**4 )
	rho_m = OmM * rho_crit # units of Msol h^3 / Mpc^3
	#print "rho_m=", rho_m
	print "OmM=", OmM
			
	plt.figure()
	#plt.semilogx(rpvec, Rint_proj, 'b+')
	#plt.hold(True)
	plt.loglog(rpvec, rho_m * proj_corr * (pa.HH0 / 100.) / (10**12), 'r+')
	plt.xlim(0.0003,8)
	plt.ylim(0.3,20000)
	plt.savefig('./plots/test_Sigma?_6e13h_fixRhoC.png')
	plt.close()
	
	
	# Get the dNdz for the full z range:
	(z, Nofz)=  N_of_zs(pa.zmin, pa.zmax, pa.zmin, pa.zmax)
	#get_NofZ(pa.alpha, pa.zs, pa.zeff, pa.zmax, pa.zpts)
	
	#plot_nofz(Nofz, z, './Nofz_test.png')
	
	# Get the comoving distances corresponding to these source z's
	chiSvec = com(z)
	# and to the lens z:
	chiLmean = com(z_l)

	# the radial window function used should be in terms of comoving distance rather than redshift, which means there is a factor of H / c:
	OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
	Wind_fac = pa.H0 * np.sqrt( (pa.OmC+pa.OmB) * (1.+z)**3 + OmL + (pa.OmR+pa.OmN) * (1.+ z)**4)
	
	#First, do the integral in comovoing distance to the source, at each Delta 
	
	# Find the delta index which corresponds to the comoving distance above which W(chiS) is 0, because the chiS integral is zero above this point no matter what:
	index_maxd=next(j[0] for j in enumerate(deltavec) if j[1]>=(chiSvec[-1]-chiLmean))
	
	holdXisint=np.zeros(len(deltavec))
	for di in range(0,len(deltavec[0:index_maxd])):	
		Integrand= Nofz * Wind_fac * (chiSvec - chiLmean - deltavec[di]) / ((1+z_of_com(chiSvec)) * (chiSvec / (1+z_of_com(chiSvec)) - chiLmean / (1+ z_of_com(chiLmean)))) 
		holdXisint[di] = scipy.integrate.simps(Integrand, chiSvec)
	
	#Now we do the integral in Delta at each R.  Note that we are leaving out the factor of rho_c  
	indexL=next(j[0] for j in enumerate(chiSvec) if j[1]>=chiLmean)
	z_delt = z_of_com(chiLmean+deltavec)
	a_delt = 1. / (1.+z_delt)

	holdDeltint=np.zeros(len(rpvec))
	for ri in range(0,len(rpvec)):
		#Integrand = (chiLmean + deltavec) / (a_delt * chiLmean *(1.+z[indexL])) * holdXisint * corrterm[:,ri]
		Integrand = (chiLmean + deltavec) / (a_delt * chiLmean *(1.+z[indexL])) * holdXisint * corr[:,ri]
		holdDeltint[ri] = scipy.integrate.simps(Integrand, deltavec)

	#Get Delta Sigma, in units of h Msun / pc^2
	#Delta_Sigma = 0.279 * holdDeltint * (pa.OmC + pa.OmB)
	Sigma = 0.279 * holdDeltint * (pa.OmC + pa.OmB)
	
	print "h=", pa.HH0/100.
	# Units Msun / pc^2
	Sigma_noh = 0.279 * holdDeltint * (pa.OmC + pa.OmB) * (pa.HH0) / 100.
	
	#ds_stack = np.column_stack((rpvec, Delta_Sigma))
	#np.savetxt('./txtfiles/DeltaSigma_M1e14_newexp.txt', ds_stack)
	
	s_stack = np.column_stack((rpvec, Sigma))
	np.savetxt('./txtfiles/Sigma_HF.txt', s_stack)
	
	plt.figure()
	plt.loglog(rpvec, Sigma, '+')
	plt.xlim(0.0003,8)
	plt.ylim(0.3,20000)
	plt.savefig('./plots/Sigma_noaverage_6e13h_fixRhoC.png')
	plt.close()
	
	plt.figure()
	plt.loglog(rpvec/ (pa.HH0/100.) * 1000., Sigma_noh, '+')
	plt.xlim(0.1,3000)
	plt.ylim(0.01,10**6)
	plt.xlabel('kpc')
	plt.savefig('./plots/Sigma_noh_6e13h_fixRhoC.png')
	plt.close()
	
	exit()
	
	# Average in bins
	#Delt_Sig_ave = average_in_bins(Delta_Sigma, rpvec, rp_bins)
	
	#plot_quant_vs_rp(Delt_Sig_ave, rp_bins_c, './DeltaSig_theory_M1e18.png')

	return Delt_Sig_ave

# ERRORS FOR FRACTIONAL ERROR CALCULATION #

def setup_shapenoise_cov(e_rms, N_ls_pbin):
	""" Returns a diagonal covariance matrix in bins of projected radius for a measurement dominated by shape noise. Elements are e_{rms}^2 / (N_ls) where N_ls is the number of l/s pairs relevant to each projected radius bin.""" 

	cov = np.diag(e_rms**2 / N_ls_pbin)
	
	return cov

def subtract_var(var_1, var_2, covar):
	""" Takes the variance of two non-independent  Gaussian random variables, and their covariance, and returns the variance of their difference."""
	
	var_diff = var_1 + var_2 - 2 * covar

	return var_diff

def get_gammaIA_cov(Cov_1, Cov_2, sys_level, covar, rp_bins, rp_bins_c):
	""" Takes information about the uncertainty on constituent elements of gamma_{IA} and combines them to get the covariance matrix of gamma_{IA} in projected radial bins."""
	""" We are only interested right now in the diagonal elements of the covariance matrix, so we assume it is diagonal. """ 

	# Set up the diagonal, statistical-error part of the matrix
	gammaIA_diag = np.diag(np.zeros(len(np.diag(Cov_1))))

	cz_a = get_cz(pa.zeff, pa.z_close, pa.e_rms_a, rp_bins, rp_bins_c)
	cz_2 = get_cz(pa.z_close, pa.zmax, pa.e_rms_a, rp_bins, rp_bins_c)
	DeltaCov_1 = setup_shapenoise_cov()
	DeltaCov_2 = setup_shapenoise_cov()
	DeltaSig_1 = get_est_DeltaSig()
	DeltaSig_2 = get_est_DeltaSig()
	BCov_1 = get_Btermcov()
	BCov_2 = get_Btermcov()
	Sig_IA_1 = get_Sig_IA()
	Sig_IA_2 = get_Sig_IA()
	BFfac_1 = get_B() - 1. + get_F()
	BFfac_2 = get_B() - 1. + get_F()

	# Calculate same
	for i in range(0,len(np.diag(Cov_1))):	 
		gammaIA_diag[i,i] = gamIA_fid[i]**2 * (subtract_var(cz_1*DeltaCov_1[i,i], cz_2*DeltaCov_2[i,i], covar_Deta[i]) / (cz_1 * DeltaSig_1 - cz_2 * DeltaSig_2)**2 + subtract_var(cz_1 * Sig_IA_1 * BCov_1[i,i], cz_2 * Sig_IA_2 * BCov_2[i,i]) / (cz_1 * Sig_IA_1 * BFfac_1 - cz_2 * Sig_IA_2 * BFfac_2)**2)

	# Set up the systematics contribution
	#sys_level is a single value for each systematic error contribution
	sys_mat = np.zeros((len(np.diag(Cov_1)), len(np.diag(Cov_1))))
	# Get a single systematics matrix by adding systematic errors in quadrature to each other and then to statistical errors.
	for i in range(0,len(sys_level)):
		sys_mat = sys_level[i]**2 * np.ones((len(np.diag(Cov_1)), len(np.diag(Cov_1)))) + sys_mat
	
	gammaIA_cov = sys_mat + gammaIA_diag

	return gammaIA_cov

def get_fid_gIA(rp_bins_c):
	""" This function computes the fiducial value of gamma_IA in each projected radial bin."""
	
	fidvals = pa.A_fid * np.asarray(rp_bins_c)**pa.beta_fid

	return fidvals

# FISHER FORECAST COMPUTATIONS #

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

# FOR EASE OF OUTPUT #
def plot_nofz(nofz_, z_, file):
	""" Plots the redshift distribution as a function of z (mostly for sanity check)"""

	plt.figure()
        plt.plot(z_, nofz_)
        plt.xlabel('$z$')
        plt.ylabel('$\\frac{dN}{dz}$')
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tick_params(axis='both', which='minor', labelsize=12)
        plt.tight_layout()
        plt.savefig(file)

	return

def plot_quant_vs_rp(quant, rp_cent, file):
	""" Plots any quantity vs the center of redshift bins"""

	plt.figure()
	plt.loglog(rp_cent, quant, 'ko')
	plt.xlabel('$r_p$')
	plt.xlim(0.05,20)
	plt.ylim(0.5,200)
	plt.tick_params(axis='both', which='major', labelsize=12)
	plt.tick_params(axis='both', which='minor', labelsize=12)
	plt.tight_layout()
	plt.savefig(file)

	return
	
def plot_variance(cov_1, fidvalues_1, bin_centers, filename):
	""" Takes a covariance matrix, a vector of the fiducial values of the object in question, and the edges of the projected radial bins, and makes a plot showing the fiducial values and 1-sigma error bars from the diagonal of the covariance matrix. Outputs this plot to location 'filename'."""


	fig_sub=plt.subplot(111)
	plt.rc('font', family='serif', size=20)
	#fig_sub=fig.add_subplot(111) #, aspect='equal')
	fig_sub.errorbar(bin_centers,fidvalues_1, yerr = np.sqrt(np.diag(cov_1)), fmt='o')
	fig_sub.set_xscale("log")
	fig_sub.set_yscale("log") #, nonposy='clip')
	fig_sub.set_xlabel('$r_p$')
	fig_sub.set_ylabel('$\gamma_{IA}$')
	fig_sub.tick_params(axis='both', which='major', labelsize=12)
	fig_sub.tick_params(axis='both', which='minor', labelsize=12)
	plt.tight_layout()
	plt.savefig(filename)

	return  


######## MAIN CALLS ##########

# Set up projected bins
rp_bins 	= 	setup_rp_bins(pa.rp_min, pa.rp_max, pa.N_bins)
rp_cent		=	rp_bins_mid(rp_bins)

print "getting z of com"
# Set up a function to get z as a function of comoving distance
z_of_com = z_interpof_com()
print "got z of com"

DSig_the = get_DeltaSig_theory_update(pa.zeff, rp_bins, rp_cent)

exit()

# Get the redshift corresponding to the maximum separation from the effective lens redshift at which we assume IA may be present
# pa.close_cut is the separation in Mpc/h.
z_close = get_z_close(pa.zeff, pa.close_cut)

# Set up the redshift distribution of sources
#(z, dNdz)	=	get_NofZ(pa.alpha, pa.zs, pa.zmin, pa.zmax, pa.zpts)

# Testing for getting fiducial value of various quantities in order to include fractional statistical error on the boost-like factor
#Boost = get_boost(rp_cent, pa.Boost_prop)
F = get_F(pa.zeff, z_close, pa.zeff+pa.delta_z, pa.e_rms_a, rp_bins, rp_cent)
F = get_F(pa.zeff, z_close, pa.zeff+pa.delta_z, pa.e_rms_b, rp_bins, rp_cent)
exit()
#CZ = get_cz(pa.zeff, pa.zmax, pa.e_rms_a, rp_bins, rp_cent)
#Sig_IA = get_Sig_IA(pa.zeff, pa.zmin, z_close, pa.zmax, pa.e_rms_a, rp_bins, rp_cent, Boost)
#DSig_the = get_DeltaSig_theory(pa.zeff, rp_bins, rp_cent)
#get_est_DeltaSig(pa.zeff, pa.zmax, pa.e_rms_a, rp_bins, rp_cent, Boost)

# Get the number of lens source pairs for each source sample in projected radial bins
N_ls_pbin_a	=	get_perbin_N_ls(rp_bins, pa.zeff, pa.n_s, pa.n_l, pa.zeff, pa.zeff+pa.delta_z,pa.Area)
N_ls_pbin_b	=	get_perbin_N_ls(rp_bins, pa.zeff, pa.n_s, pa.n_l, pa.zeff+pa.delta_z, pa.zphmax, pa.Area)

# Get the covariance matrix in projected radial bins of Delta Sigma for samples a and b
Cov_a		=	setup_shapenoise_cov(pa.e_rms_a, N_ls_pbin_a)
Cov_b		=	setup_shapenoise_cov(pa.e_rms_b, N_ls_pbin_b)

# Combine the constituent covariance matrices to get the covariance matrix for gamma_IA in projected radial bins
Cov_gIA		= 	get_gammaIA_cov(Cov_a, Cov_b, pa.sys_sigc, np.zeros(len(np.diag(Cov_b))))

# Get the fiducial value of gamma_IA in each projected radial bin
fid_gIA		=	get_fid_gIA(rp_bins)

# Output a plot showing the 1-sigma error bars on gamma_IA in projected radial biplot_variance
plot_variance(Cov_gIA, fid_gIA, rp_cent, pa.plotfile)

# Get the parameter derivatives required to construct the Fisher matrix
ders		=	par_derivs(pa.par, rp_cent)

# Get the Fisher matrix
fish 		=	get_Fisher(ders, Cov_gIA)

# If desired, cut parameters which you want to fix from Fisher matrix:
fish_cut 	=	cut_Fisher(fish, None)

# Get the covariance matrix from either fish or fish_cut, and marginalise over any desired parameters
parCov		=	get_par_Cov(fish_cut, None)

# Output whatever we want to know about the parameters:
par_const_output(fish_cut, parCov)
