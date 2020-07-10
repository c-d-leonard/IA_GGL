# Make all the paper plots except for the F vs no F one

import numpy as np
import matplotlib.pyplot as plt

survey='SDSS'
endfile = 'fixDls'

# Import the parameter file:
if (survey=='SDSS'):
	import params as pa
elif (survey=='LSST_DESI'):
	import params_LSST_DESI as pa
else:
	print "We don't have support for that survey yet; exiting."
	exit()


# StoN 1D plot
(rp, StoN_Blazek) = np.loadtxt('./txtfiles/StoN/StoN_sysb_stat_Blazek_'+survey+'_deltaz='+str(pa.delta_z)+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt', unpack=True)
#(rp, StoN_shapes_1) = np.loadtxt('./txtfiles/StoN/StoN_sysB_stat_shapes_'+survey+'_rlim='+str(pa.mlim)+'_a0.2_rho0.8_'+endfile+'.txt', unpack=True)
#(rp, StoN_shapes_2) = np.loadtxt('./txtfiles/StoN/StoN_sysB_stat_shapes_'+survey+'_rlim='+str(pa.mlim)+'_a0.8_rho0.2_'+endfile+'.txt', unpack=True)

"""
plt.rc('font', family='serif', size=14)
f, axarr = plt.subplots(2, sharex=True, figsize=[5,8])
axarr[0].scatter(rp, StoN_shapes_1, color='b', marker='^', s=100, label='$a=0.2, \, \\rho=0.8$')
axarr[0].hold(True)
axarr[0].scatter(rp ,StoN_shapes_2,color='#FFA500', marker='o', s=100, label='$a=0.8, \, \\rho=0.2$')
axarr[1].scatter(rp , StoN_shapes_1 / StoN_Blazek, color='b', marker='^', s=100, label='$a=0.2, \, \\rho=0.8$')
axarr[1].hold(True)
axarr[1].scatter(rp ,StoN_shapes_2 / StoN_Blazek,color='#FFA500', marker='o', s=100, label='$a=0.8, \, \\rho=0.2$')
axarr[0].set_xscale("log")
axarr[1].set_ylim(0, 2.5)
axarr[0].set_ylim(0, 1.5)
axarr[0].set_xlim(0.05,20.)
axarr[1].set_xlabel('$r_p, \, {\\rm Mpc / h}$', fontsize=20)
axarr[0].set_ylabel('$S/N$', fontsize=20)
axarr[1].set_ylabel('$\\frac{S/N}{S/N_{\\rm B2012}}$', fontsize=30)
axarr[0].legend(loc='upper left', fontsize=14)
plt.tight_layout()
plt.savefig('./plots/StoN1d_SDSS_2vs8_'+endfile+'.png')"""

"""# StoN 1D plots (will not use all these, just want to look at this for scale-dependence
a_vec = np.loadtxt('./txtfiles/a_survey='+survey+'.txt')
rho_vec = np.loadtxt('./txtfiles/rho_survey='+survey+'.txt')
(rp, StoN_Blazek) = np.loadtxt('./txtfiles/StoN/StoN_sysb_stat_Blazek_'+survey+'_deltaz='+str(pa.delta_z)+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt', unpack=True)

for ai in range(0,len(a_vec)):
	for ri in range(0,len(rho_vec)):
		(rp, StoN) = np.loadtxt('./txtfiles/StoN/StoN_sysB_stat_shapes_'+survey+'_rlim='+str(pa.mlim)+'_a'+str(a_vec[ai])+'_rho'+str(rho_vec[ri])+'_'+endfile+'.txt', unpack=True)
		
		plt.rc('font', family='serif', size=14)
		f, axarr = plt.subplots(2, sharex=True, figsize=[5,8])
		axarr[0].scatter(rp, StoN, color='b', marker='^', s=100, label='a='+str(a_vec[ai])+' rho='+str(rho_vec[ri]))
		axarr[1].scatter(rp , StoN / StoN_Blazek, color='b', marker='^', s=100, label='a='+str(a_vec[ai])+' rho='+str(rho_vec[ri]))
		axarr[0].set_xscale("log")
		#axarr[1].set_ylim(0, 2.5)
		#axarr[0].set_ylim(0, 1.5)
		axarr[0].set_xlim(0.05,20.)
		axarr[1].set_xlabel('$r_p, \, {\\rm Mpc / h}$', fontsize=20)
		axarr[0].set_ylabel('$S/N$', fontsize=20)
		axarr[1].set_ylabel('$\\frac{S/N}{S/N_{\\rm B2012}}$', fontsize=30)
		axarr[0].legend(loc='upper left', fontsize=14)
		plt.tight_layout()
		plt.savefig('./plots/StoN_1d_shapes_stat_sysB/StoN1d_SDSS_a='+str(a_vec[ai])+'_rho='+str(rho_vec[ri])+'_'+endfile+'.png')
		plt.close()"""
		
		
# Make the 1d plot we want for the paper specifically.

(rp, StoN_shapes_1) = np.loadtxt('./txtfiles/StoN/StoN_sysB_stat_shapes_'+survey+'_rlim='+str(pa.mlim)+'_a0.2_rho0.8_'+endfile+'.txt', unpack=True)
(rp, StoN_shapes_2) = np.loadtxt('./txtfiles/StoN/StoN_sysB_stat_shapes_'+survey+'_rlim='+str(pa.mlim)+'_a0.8_rho0.2_'+endfile+'.txt', unpack=True)

if survey=='SDSS':
	title_words= 'SDSS'
	max_ston = 1.2
elif survey =='LSST_DESI':
	title_words= "LSST+DESI"
	max_ston = 11.
else:
	print "We don't have support for that survey"
	exit()

plt.rc('font', family='serif', size=14)
f, axarr = plt.subplots(2, sharex=True, figsize=[5,8])
axarr[0].scatter(rp, StoN_shapes_1, color='b', marker='^', s=100, label='$a=0.2, \, \\rho=0.8$')
axarr[0].hold(True)
axarr[0].scatter(rp ,StoN_shapes_2,color='#FFA500', marker='o', s=100, label='$a=0.8, \, \\rho=0.2$')
axarr[1].scatter(rp , StoN_shapes_1 / StoN_Blazek, color='b', marker='^', s=100, label='$a=0.2, \, \\rho=0.8$')
axarr[1].hold(True)
axarr[1].scatter(rp ,StoN_shapes_2 / StoN_Blazek,color='#FFA500', marker='o', s=100, label='$a=0.8, \, \\rho=0.2$')
axarr[0].set_xscale("log")
axarr[1].set_ylim(0, 2.5)
axarr[0].set_ylim(0, max_ston)
axarr[0].set_xlim(0.05,20.)
axarr[1].set_xlabel('$r_p, \, {\\rm Mpc / h}$', fontsize=20)
axarr[0].set_ylabel('$S/N$', fontsize=20)
axarr[1].set_ylabel('Ratio with Blazek et al. 2012', fontsize=15)
axarr[0].legend(loc='upper left', fontsize=14)
plt.suptitle(title_words, fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('./plots/StoN1d_'+survey+'_2vs8_'+endfile+'.png')
plt.close()

# StoN 2D plot
StoNsq_Blazek = np.loadtxt('./txtfiles/StoN/StoN_stat_sysB_Blazek_'+survey+'_deltaz='+str(pa.delta_z)+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt')
StoN_Blazek = np.sqrt(StoNsq_Blazek)
StoNsq_Blazek_gamt = np.loadtxt('./txtfiles/StoN/StoN_stat_sysB_gamt_Blazek_'+survey+'_deltaz='+str(pa.delta_z)+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt')
StoN_Blazek_gamt = np.sqrt(StoNsq_Blazek_gamt)
a = np.loadtxt('./txtfiles/a_survey='+survey+'.txt')
covperc = np.loadtxt('./txtfiles/rho_survey='+survey+'.txt')
StoNsq_shapes = np.loadtxt('./txtfiles/StoN/StoNsq_stat_sysB_shapes_survey='+survey+'_rlim='+str(pa.mlim)+'_fixB_'+endfile+'.txt')
StoN_shapes = np.sqrt(StoNsq_shapes)
StoNratio = StoN_shapes / StoN_Blazek
StoNratio_gamtBl = StoN_shapes / StoN_Blazek_gamt

print np.amax(StoNratio)

plt.figure(figsize=(10, 10))
plt.rcParams["font.family"] = "serif"
#plt.scatter([0.7], [0.75], color='white', marker='o', s=150)
plt.imshow(StoNratio, extent=[covperc[0], covperc[-1], a[-1], a[0]], aspect=1, interpolation='spline36')
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize='30')
plt.clim(0,2.52)
plt.contour(covperc, a, StoNratio, [1.0], colors='k', linewidths=4)
plt.xlabel('$\\rho$', fontsize=40)
plt.ylabel('$a$', fontsize=40)
plt.tick_params(axis='both', labelsize='30')
plt.subplots_adjust(top=0.88)
plt.suptitle(title_words, fontsize='35')
plt.tight_layout()
plt.savefig('./plots/StoN_2d_stat_sysB_'+survey+'_'+endfile+'.png')
plt.close()

plt.figure(figsize=(10, 10))
plt.rcParams["font.family"] = "serif"
#plt.scatter([0.7], [0.75], color='white', marker='o', s=150)
plt.imshow(StoNratio_gamtBl, extent=[covperc[0], covperc[-1], a[-1], a[0]], aspect=1, interpolation='spline36')
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize='30')
plt.clim(0,1.7)
plt.contour(covperc, a, StoNratio, [1.0], colors='k', linewidths=4)
plt.xlabel('$\\rho$', fontsize=40)
plt.ylabel('$a$', fontsize=40)
plt.tick_params(axis='both', labelsize='30')
plt.subplots_adjust(top=0.88)
plt.suptitle(title_words, fontsize='35')
plt.tight_layout()
plt.savefig('./plots/StoN_2d_stat_sysB_gamtBl_'+survey+'_'+endfile+'.png')
plt.close()
