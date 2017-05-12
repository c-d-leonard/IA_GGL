# This file contains functions for getting w_{l+} which are shared between the Blazek et al. 2012 method and the multiple shape measurements method.

# Functions to get the 1halo term of w_{l+}

def get_pi(q1, q2, q3, z_):
	""" Returns the pi functions requires for the 1 halo term in wg+, at z_ """
	
	pi = q1 * np.exp(q2 * z_**q3)
	
	return pi

def get_P1haloIA(z, k, ah, q11, q12, q13, q21, q22, q23, q31, q32, q33):
	""" Returns the power spectrum required for the wg+ 1 halo term, at z and k_perpendicular ( = k) """
	
	p1 = get_pi(q11, q12, q13, z)
	p2 = get_pi(q21, q22, q23, z)
	p3 = get_pi(q31, q32, q33, z)
	
	P1halo = ah * ( k / p1 )**2 / ( 1. + ( k /p2 )** p3)
	
	#plt.figure()
	#plt.loglog(k, P1halo, 'm+')
	#plt.ylim(10**(-18), 10**3)
	#plt.xlim(10**(-3), 10**3)
	#plt.savefig('./plots/P1halo_g+.png')
	#plt.close()
	
	P1halo = np.zeros((len(k), len(z)))
	for ki in range(0,len(k)):
		for zi in range(0,len(z)):
			P1halo[ki, zi]  = pa.ah * ( k[ki] / p1[zi] )**2 / ( 1. + ( k[ki] /p2[zi] )** p3[zi])
	
	return P1halo

def window():
	""" Get window function for w_{l+} and w_{ls} 2-halo terms."""
	
	sigz = pa.sigz_gwin  # This is a very small value to basically set the lenses to a delta function.
	z = scipy.linspace(pa.zeff-5.*sigz, pa.zeff+5.*sigz, 50)
	dNdz_1 = 1. / np.sqrt(2. * np.pi) / sigz * np.exp(-(z-pa.zeff)**2 / (2. *sigz**2)) # Lens distribution - a very narrow Gaussian about the effective redshift.
	
	chi = com(z)
	OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
	dzdchi = pa.H0 * ( (pa.OmC+pa.OmB)*(1+z)**3 + OmL + (pa.OmR+pa.OmN) * (1+z)**4 )**(0.5)
	
	(z, dNdz_2) = get_NofZ_unnormed(pa.alpha_fid, pa.zs_fid, pa.zeff-5.*sigz, pa.zeff+5.*sigz, 50)
		
	norm = scipy.integrate.simps(dNdz_1*dNdz_2 / chi**2 * dzdchi, z)
	
	win = dNdz_1*dNdz_2 / chi**2 * dzdchi / norm
	
	return (z, win )

def get_Pk(z_):
	""" Calls camb and returns the nonlinear halofit power spectrum for the current cosmological parameters and at the redshifts of interest. Returns a 2D array in k and z."""
	
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
	
	return (k, Pofkz)

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
	
	return D

def wgp_1halo(rp_c_, z, k, bd, Ai, ah, q11, q12, q13, q21, q22, q23, q31, q32, q33):
	""" Returns the 1 halo term of wg+(rp) """
	
	(z, w) = window() 
	
	# Set up a k vector to integrate over:
	k = np.logspace(-5., 7., 100000)
	
	# Get the `power spectrum' term
	P1h = get_P1haloIA(z, k, ah, q11, q12, q13, q21, q22, q23, q31, q32, q33)
	
	# First do the integral over z:
	zint = np.zeros(len(k))
	for ki in range(0,len(k)):
		zint[ki] = scipy.integrate.simps(P1h[ki, :] * w , z)
		
	#plot_quant_vs_quant(k, zint, './zint.png')
		
	# Now do the integral in k
	ans = np.zeros(len(rp_c_))
	for rpi in range(0,len(rp_c_)):
		integrand = k * zint * scipy.special.j0(rp_c_[rpi] * k)
		ans[rpi] = scipy.integrate.simps(integrand, k)
		#ans[rpi] = scipy.integrate.simps(k * P1h * scipy.special.j0(rp_c_[rpi] * k), k)
		
	# Set this to zero above about 2 * virial radius (I've picked this value somewhat aposteriori, should do better). This is to not include 1halo contributions well outside the halo.
	Rvir = Rhalo(pa.Mvir)

	for ri in range(0,len(rp_c_)):
		if (rp_c_[ri]> 2.*Rvir):
			ans[ri] = 0.
	
	wgp1h = ans / (2. * np.pi)
	
	#plt.figure()
	#plt.loglog(rp_c_, wgp1h, 'bo')
	#plt.xlim(0.1, 200)
	#plt.ylim(0.01, 30)
	#plt.xlabel('$r_p$, Mpc/h com')
	#plt.ylabel('$w_{g+}$, Mpc/ h com')
	#plt.savefig('./plots/wg+_1h_ah=1.pdf')
	#plt.close()
	
	wgp_save = np.column_stack((rp_c_, wgp1h))
	np.savetxt('./txtfiles/wgp_1halo_LRG.txt', wgp_save)
		
	return wgp1h

def wgp_2halo(rp_cents_, bd, Ai):
	""" Returns wgp from the nonlinear alignment model (2-halo term only). """
	
	# Get the redshift window function
	z_gp, win_gp = window()
	
	# Get the required matter power spectrum from camb 
	(k_gp, P_gp) = get_Pk(z_gp, 'gp')
	
	# Get the growth factor
	D_gp = growth(z_gp)
	
	# First do the integral over z. Don't yet interpolate in k.
	zint_gp = np.zeros(len(k_gp))
	for ki in range(0,len(k_gp)):
		zint_gp[ki] = scipy.integrate.simps(win_gp * P_gp[:, ki] / D_gp, z_gp)
		
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
			
	# g+: integral in kz	
	kz_int_gp = np.zeros(len(kp_gp))
	for kpi in range(0,len(kp_gp)):
		kz_int_gp[kpi] = scipy.integrate.simps(kpkz_gp[kpi,:] * kp_gp[kpi]**3 / ( (kp_gp[kpi]**2 + kz_gp**2)*kz_gp) * np.sin(kz_gp*pa.close_cut), kz_gp)
			
	# Finally, do the integrals in kperpendicular
	kp_int_gp = np.zeros(len(rp_cents_))
	for rpi in range(0,len(rp_cents_)):
		kp_int_gp[rpi] = scipy.integrate.simps(scipy.special.jv(2, rp_cents_[rpi]* kp_gp) * kz_int_gp, kp_gp)
		
	wgp_NLA = kp_int_gp * Ai * bd * pa.C1rho * (pa.OmC + pa.OmB) / np.pi**2
	
	wgp_stack = np.column_stack((rp_cents_, wgp_NLA))
	np.savetxt('./txtfiles/wgp_2halo_'+str(pa.kpts_wgp)+'pts_LRG.txt', wgp_stack)
	
	return wgp_NLA
