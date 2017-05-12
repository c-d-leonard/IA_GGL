import numpy as np
import matplotlib.pyplot as plt

# Make plots which display which systematic errors are the most important in the Blazek et al. method via including fudge factors.

# Import the case where there is no systematic error from an inadequate spectroscopic subsample (but includes systematic error onthe boost)
(rp, err_nosys) = np.loadtxt('../txtfiles/frac_totError_Blazek_SDSS_NsatThresh.txt', unpack=True)

# These are the levels of fudge factor systematic error we consider.
frac_levels = [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

# Make a plot visualizing the effect of uncertainty in the different factors, and import the sys-only files for below.

# 1. cz_a

err_cza_sys = [0]*len(frac_levels)
err_cza_sys_only = [0]*len(frac_levels)
for i in range(0,len(frac_levels)):
	rp, err_cza_sys[i] = np.loadtxt('../txtfiles/frac_totError_Blazek_SDSS_NsatThresh_fudgeczA='+str(frac_levels[i])+'.txt', unpack=True)
	rp, err_cza_sys_only[i] = np.loadtxt('../txtfiles/frac_sysZError_Blazek_SDSS_NsatThresh_fudgeczA='+str(frac_levels[i])+'.txt', unpack=True)
	
"""plt.figure()
plt.loglog(rp, err_nosys, 'ko', label='no z sys')
plt.hold(True)
plt.loglog(rp, err_cza_sys[0], 'mo', label=str(frac_levels[0]))
plt.hold(True)
plt.loglog(rp, err_cza_sys[1], 'bo', label=str(frac_levels[1]))
plt.hold(True)
plt.loglog(rp, err_cza_sys[2], 'ro', label=str(frac_levels[2]))
plt.hold(True)
plt.loglog(rp, err_cza_sys[3], 'go', label=str(frac_levels[3]))
plt.legend()
plt.xlabel('$r_p$, Mpc/h')
plt.ylabel('Fractional error')
plt.xlim(0.04, 35)
#plt.ylim(0.02, 3)
plt.legend()
plt.title('Frac sys error, czA')
plt.savefig('./plots/test_sys_impact_cza_NsatThresh.pdf')
plt.close()"""


# 2. cz_b

err_czb_sys = [0]*len(frac_levels)
err_czb_sys_only = [0]*len(frac_levels)
for i in range(0,len(frac_levels)):
	rp, err_czb_sys[i] = np.loadtxt('../txtfiles/frac_totError_Blazek_SDSS_NsatThresh_fudgeczB='+str(frac_levels[i])+'.txt', unpack=True)
	rp, err_czb_sys_only[i] = np.loadtxt('../txtfiles/frac_sysZError_Blazek_SDSS_NsatThresh_fudgeczB='+str(frac_levels[i])+'.txt', unpack=True)
	
"""plt.figure()
plt.loglog(rp, err_nosys, 'ko', label='no z sys')
plt.hold(True)
plt.loglog(rp, err_czb_sys[0], 'mo', label=str(frac_levels[0]))
plt.hold(True)
plt.loglog(rp, err_czb_sys[1], 'bo', label=str(frac_levels[1]))
plt.hold(True)
plt.loglog(rp, err_czb_sys[2], 'ro', label=str(frac_levels[2]))
plt.hold(True)
plt.loglog(rp, err_czb_sys[3], 'go', label=str(frac_levels[3]))
plt.legend()
plt.xlabel('$r_p$, Mpc/h')
plt.ylabel('Fractional error')
plt.xlim(0.04, 35)
#plt.ylim(0.02, 3)
plt.legend()
plt.title('Frac sys error, czB')
plt.savefig('./plots/test_sys_impact_czB_NsatThresh.pdf')
plt.close()"""
	
# 3. F_a

err_Fa_sys = [0]*len(frac_levels)
err_Fa_sys_only = [0]*len(frac_levels)
for i in range(0,len(frac_levels)):
	rp, err_Fa_sys[i] = np.loadtxt('../txtfiles/frac_totError_Blazek_SDSS_NsatThresh_fudgeFA='+str(frac_levels[i])+'.txt', unpack=True)
	rp, err_Fa_sys_only[i] = np.loadtxt('../txtfiles/frac_sysZError_Blazek_SDSS_NsatThresh_fudgeFA='+str(frac_levels[i])+'.txt', unpack=True)
	
"""plt.figure()
plt.loglog(rp, err_nosys, 'ko', label='no z sys')
plt.hold(True)
plt.loglog(rp, err_Fa_sys[0], 'mo', label=str(frac_levels[0]))
plt.hold(True)
plt.loglog(rp, err_Fa_sys[1], 'bo', label=str(frac_levels[1]))
plt.hold(True)
plt.loglog(rp, err_Fa_sys[2], 'ro', label=str(frac_levels[2]))
plt.hold(True)
plt.loglog(rp, err_Fa_sys[3], 'go', label=str(frac_levels[3]))
plt.legend()
plt.xlabel('$r_p$, Mpc/h')
plt.ylabel('Fractional error')
plt.xlim(0.04, 35)
#plt.ylim(0.02, 3)
plt.legend()
plt.title('Frac sys error, Fa')
plt.savefig('./plots/test_sys_impact_Fa_NsatThresh.pdf')
plt.close()"""

# 4. F_b

err_Fb_sys = [0]* len(frac_levels)
err_Fb_sys_only = [0]* len(frac_levels)
for i in range(0,len(frac_levels)):
	rp, err_Fb_sys[i] = np.loadtxt('../txtfiles/frac_totError_Blazek_SDSS_NsatThresh_fudgeFB='+str(frac_levels[i])+'.txt', unpack=True)
	rp, err_Fb_sys_only[i] = np.loadtxt('../txtfiles/frac_sysZError_Blazek_SDSS_NsatThresh_fudgeFB='+str(frac_levels[i])+'.txt', unpack=True)
	
"""plt.figure()
plt.loglog(rp, err_nosys, 'ko', label='no z sys')
plt.hold(True)
plt.loglog(rp, err_Fb_sys[0], 'mo', label=str(frac_levels[0]))
plt.hold(True)
plt.loglog(rp, err_Fb_sys[1], 'bo', label=str(frac_levels[1]))
plt.hold(True)
plt.loglog(rp, err_Fb_sys[2], 'ro', label=str(frac_levels[2]))
plt.hold(True)
plt.loglog(rp, err_Fb_sys[3], 'go', label=str(frac_levels[3]))
plt.legend()
plt.xlabel('$r_p$, Mpc/h')
plt.ylabel('Fractional error')
plt.xlim(0.04, 35)
#plt.ylim(0.02, 3)
plt.legend()
plt.title('Frac sys error, Fb')
plt.savefig('./plots/test_sys_impact_Fb_NsatThresh.pdf')
plt.close()"""

# 5. SigIA_a

err_sigA_sys =[0]*len(frac_levels)
err_sigA_sys_only =[0]*len(frac_levels)
for i in range(0,len(frac_levels)):
	rp, err_sigA_sys[i] = np.loadtxt('../txtfiles/frac_totError_Blazek_SDSS_NsatThresh_fudgesigA='+str(frac_levels[i])+'.txt', unpack=True)
	rp, err_sigA_sys_only[i] = np.loadtxt('../txtfiles/frac_sysZError_Blazek_SDSS_NsatThresh_fudgesigA='+str(frac_levels[i])+'.txt', unpack=True)
	
"""plt.figure()
plt.loglog(rp, err_nosys, 'ko', label='no z sys')
plt.hold(True)
plt.loglog(rp, err_sigA_sys[0], 'mo', label=str(frac_levels[0]))
plt.hold(True)
plt.loglog(rp, err_sigA_sys[1], 'bo', label=str(frac_levels[1]))
plt.hold(True)
plt.loglog(rp, err_sigA_sys[2], 'ro', label=str(frac_levels[2]))
plt.hold(True)
plt.loglog(rp, err_sigA_sys[3], 'go', label=str(frac_levels[3]))
plt.legend()
plt.xlabel('$r_p$, Mpc/h')
plt.ylabel('Fractional error')
plt.xlim(0.04, 35)
#plt.ylim(0.02, 3)
plt.legend()
plt.title('Frac sys error, sigA')
plt.savefig('./plots/test_sys_impact_sigA_NsatThresh.pdf')
plt.close()"""

# 6. SigIA_b

err_sigB_sys = [0]*len(frac_levels)
err_sigB_sys_only = [0]*len(frac_levels)
for i in range(0,len(frac_levels)):
	rp, err_sigB_sys[i] = np.loadtxt('../txtfiles/frac_totError_Blazek_SDSS_NsatThresh_fudgesigB='+str(frac_levels[i])+'.txt', unpack=True)
	rp, err_sigB_sys_only[i] = np.loadtxt('../txtfiles/frac_sysZError_Blazek_SDSS_NsatThresh_fudgesigB='+str(frac_levels[i])+'.txt', unpack=True)
	
"""plt.figure()
plt.loglog(rp, err_nosys, 'ko', label='no z sys')
plt.hold(True)
plt.loglog(rp, err_sigB_sys[0], 'mo', label=str(frac_levels[0]))
plt.hold(True)
plt.loglog(rp, err_sigB_sys[1], 'bo', label=str(frac_levels[1]))
plt.hold(True)
plt.loglog(rp, err_sigB_sys[2], 'ro', label=str(frac_levels[2]))
plt.hold(True)
plt.loglog(rp, err_sigB_sys[3], 'go', label=str(frac_levels[3]))
plt.legend()
plt.xlabel('$r_p$, Mpc/h')
plt.ylabel('Fractional error')
plt.xlim(0.04, 35)
#plt.ylim(0.02, 3)
plt.legend()
plt.title('Frac sys error, sigB')
plt.savefig('./plots/test_sys_impact_sigB_NsatThresh.pdf')
plt.close()"""


# Okay, now, for each quantity and each level of fractional error, get the maximum ratio of sys_z_only / (stat + sys_boost), and make a plot of this max value vs fractional error for each quantity

max_cza = np.zeros(len(frac_levels))
max_czb = np.zeros(len(frac_levels))
max_Fa = np.zeros(len(frac_levels))
max_Fb = np.zeros(len(frac_levels))
max_SigA = np.zeros(len(frac_levels))
max_SigB = np.zeros(len(frac_levels))

for i in range(0, len(frac_levels)):
	max_cza[i] = max(err_cza_sys_only[i] / err_nosys)
	max_czb[i] = max(err_czb_sys_only[i] / err_nosys)
	max_Fa[i] = max(err_Fa_sys_only[i] / err_nosys)
	max_Fb[i] = max(err_Fb_sys_only[i] / err_nosys)
	max_SigA[i] = max(err_sigA_sys_only[i] / err_nosys)
	max_SigB[i] = max(err_sigB_sys_only[i] / err_nosys)
	

plt.figure()
plt.loglog(frac_levels, max_cza, 'ko', label='$c_z^a$')
plt.hold(True)
plt.loglog(frac_levels, max_czb, 'mo', label='$c_z^b$')
plt.hold(True)
plt.loglog(frac_levels, max_Fa, 'bo', label='$F_a$')
plt.hold(True)
plt.loglog(frac_levels, max_Fb, 'ro', label='$F_b$')
plt.hold(True)
plt.loglog(frac_levels, max_SigA, 'go', label='$<\\Sigma_{IA}^a>$')
plt.hold(True)
plt.loglog(frac_levels, max_SigB, 'yo', label='$<\\Sigma_{IA}^b>$')
plt.legend()
plt.xlabel('Fractional error level')
plt.ylabel('max(sysZ / (stat + sysB))')
plt.xlim(0.005, 10)
plt.ylim(0.001, 50)
plt.legend()
plt.title('Ratio, sysZ to (stat + sysB)')
plt.savefig('../plots/ratio_sysZ_stat+sysB.pdf')
plt.close()	

	
