# Get the slope of the power law for different levels of systematic error due to redshift

import numpy as np

SURVEY = 'LSST_DESI'
endfile = 'fixDls'

# Import the parameter file:
if (SURVEY=='SDSS'):
	import params as pa
elif (SURVEY=='LSST_DESI'):
	import params_LSST_DESI as pa
else:
	print "We don't have support for that survey yet; exiting."
	exit()

# Load information for A_ph from our method
# Both are 2D matrices. The sysz one is a function of a and fraclevel - pick the a value we want to use. The stat one is a function of a and rho - pick the value of a and rho we want to use.
StoNsq_sysz_Ncorr_mat = np.loadtxt('./txtfiles/StoN/StoNsq_sysz_shapes_survey='+SURVEY+'_rlim='+str(pa.mlim)+'_fixB_'+endfile+'.txt')
StoNsq_stat_Ncorr_mat = np.loadtxt('./txtfiles/StoN/StoNsq_stat_shapes_survey='+SURVEY+'_rlim='+str(pa.mlim)+'_fixB_'+endfile+'.txt')
avec = np.loadtxt('./txtfiles/a_survey='+SURVEY+'.txt')
rhovec = np.loadtxt('./txtfiles/rho_survey='+SURVEY+'.txt')

# Here we pick our a and rho values and find their indices:
a = 0.2; rho = 0.8
ind_a = next(j[0] for j in enumerate(avec) if j[1]>=a)
ind_rho = next(j[0] for j in enumerate(rhovec) if j[1]>=rho)

StoN_sq_sysz_Ncorr = StoNsq_sysz_Ncorr_mat[ind_a, :]
StoN_sq_stat_Ncorr = StoNsq_stat_Ncorr_mat[ind_a, ind_rho]
print "a=0.2, rho=0.8. S/N integrated=", np.sqrt(StoN_sq_stat_Ncorr)

#Load the stuff from the Blazek 2012 method.
levels, StoN_cza, StoN_czb, StoN_Fa, StoN_Fb, StoN_Siga, StoN_Sigb = np.loadtxt('./txtfiles/StoN/StoN_SysToStat_Blazek_'+SURVEY+'_deltaz='+str(pa.delta_z)+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt', unpack=True)
StoNstat = np.loadtxt('./txtfiles/StoN/StoNstat_Blazek_'+SURVEY+'_deltaz='+str(pa.delta_z)+'_rlim='+str(pa.mlim)+'_'+endfile+'.txt')

print "Blazek, StoNstat=", np.sqrt(StoNstat)

# Save the slopes to file
cza_slope = (np.sqrt(StoNstat) / np.sqrt(StoN_cza[0])) / pa.fudge_frac_level[0]
czb_slope = (np.sqrt(StoNstat) / np.sqrt(StoN_czb[0])) / pa.fudge_frac_level[0]
Fa_slope = (np.sqrt(StoNstat) / np.sqrt(StoN_Fa[0])) / pa.fudge_frac_level[0]
Fb_slope = (np.sqrt(StoNstat) / np.sqrt(StoN_Fb[0])) / pa.fudge_frac_level[0]
Siga_slope = (np.sqrt(StoNstat) / np.sqrt(StoN_Siga[0])) / pa.fudge_frac_level[0]
Sigb_slope = (np.sqrt(StoNstat) / np.sqrt(StoN_Sigb[0])) / pa.fudge_frac_level[0]
F_slope =  (np.sqrt(StoN_sq_stat_Ncorr) / np.sqrt(StoN_sq_sysz_Ncorr[0]))/ pa.fudge_frac_level[0]

save_slopes =  np.asarray([cza_slope, czb_slope, Fa_slope, Fb_slope,  Siga_slope, Sigb_slope,F_slope])

# 1 column, from top to bottom slopes for cz_a, cz_b, F_a, F_b, SigIA_a, SigIA_b, A_ph
np.savetxt('./txtfiles/sysz_slopes_'+SURVEY+'_'+endfile+'.txt', save_slopes)

# Get the values of the max sig_sysz / sig_stat for each one alone:

max_frac_err = np.loadtxt('./txtfiles/sysfracmax_survey='+pa.survey+'_deltaz='+str(pa.delta_z)+'_'+endfile+'.txt')
	
ratio_cza = max_frac_err * cza_slope
ratio_czb = max_frac_err * czb_slope
ratio_Fa = max_frac_err * Fa_slope
ratio_Fb = max_frac_err * Fb_slope
ratio_Siga = max_frac_err * Siga_slope
ratio_Sigb = max_frac_err * Sigb_slope
ratio_F = max_frac_err * F_slope

save_ratios =  np.asarray([ratio_cza, ratio_czb, ratio_Fa, ratio_Fb, ratio_Siga, ratio_Sigb,ratio_F])

# 1 column, from top to bottom slopes for cz_a, cz_b, F_a, F_b, SigIA_a, SigIA_b, A_ph
np.savetxt('./txtfiles/sysz_ratios_'+SURVEY+'_'+endfile+'.txt', save_ratios)
