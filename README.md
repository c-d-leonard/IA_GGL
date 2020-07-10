Disclaimer: This is very ugly code.

The workflow which I actually used to run everything for the 2018 paper can be seen in run_all_local.sh, but this may not be that helpful.

There are several 'driver' scripts to compute different quantities in different scenarios. Let's consider the one called 'constrain_IA_Blazekmethod_gamt_update.py', in the home directory. This script computes from theory gamma_IA and the covariance matrix (assuming we use gamma_t rather than DeltaSigma - that's why it's not called in the run_all_local script above, because we used DeltaSigma for the paper). It uses a more 'direct' approach to do the gamma_IA theory prediction (gamma_IA ~ wgp / wgg), but can also calculate the theoretical values of the separate observational quantities like boost, F etc because they are used in computing the covariance matrix.

To run it, you need to open it and first, at the top, change the SURVEY tag to be the survey you are considering. I have only implemented two options here, the ones we considered in the 2018 paper: SDSS and LSST_DESI. We will need to add DES options. You should also set a string for the variable endfilename - this will be appended to any output files and should be something which identifies the calculation you are doing (beyond the survey setup) so you can keep track if you need to rerun this several times.

The script is structured such that all the functions are defined first, and then at the bottom is the main driving part of the script. If you scroll to the bottom, you will see that the first thing we do is look for a parameter file to match the SURVEY you have specified. This is going to need to be constructed for DES. You can open params_LSST_DESI.py as an example for the case of LSST sources and DESI lenses. params.py is for SDSS sources and lenses. This reads all the parameter numbers describing the observational setup. Finding values for all these will be a nontrivial task.

We then set up bins in projected radius $r_p$. This code works in terms of $r_p$ rather than $\theta$, but converting back and forth using the mean redshift of the lens sample should be very easy to incorporate. rp_bins are the bin edges and rp_cent are the central values of the bins. Both in Mpc/h.

We then set up a function to get z as a function of comoving distance. This is called from an external file called shared_functions_setup which is imported as setup.

We then call gamma_fid, which is a function defined in this same file above to get the fiducial theory value of gamma_IA.

If we look at the function gamma_fid, we see that first we are calling two functions wgg_full and wgp_full, with filename arguments which depend on the SURVEY. You will recognize that gamma_IA = wgp / wgg, so that's why we are getting these. Those functions are in an external file, imported under the name ws. The full name, if we look above, is shared_functions_wlp_wls.py. To follow what's happening we need to go look in that file.

Looking first at wgg_full, we see that the function will first try to import the calculated value from the given filename if it exists. This is because these calculations take an annoyingly long time (or they do the way I had set them up because I'm not using FFTlog here, I'm doing some kind of horrible brute force fourier transform). The 1halo term and the 2halo term are either loaded if the respective files exist or they are computed if not, using functions wgg_1halo_Four() and wgg_2halo(). wgg_2halo uses ccl to get the nonlinear matter power spectrum - this syntax may need updating to be compatible with new ccl versions. It also calls the function window to get the window function of the survey - this will probably need modifying for DES. The 1halo term is much more complicated and most of the rest of the functions in this file are for calculating that for either wgg and wgp. 

Note particularly - I have coded up particular HODs which are appropriate for the surveys I used in this project (SDSS and LSST+DESI). You can see this in e.g. the function get_Pkgg_1halo_multiz, but basically all the 1halo functions. We will probably need to code up a specific DES HOD to model the 1halo stuff in this sample. This will be nontrivial.

Assuming we can get these wgg and wgp quantities, heading back to the main driver script (constrain_IA_Blazekmethod_gamt_update.py) within function gamma_fid, the next thing that happens is that we get the red fraction. This is important because red and blue galaxies contribute to IA differently. If the file where we expect to find the red fraction already exists we load that, otherwise we call the function get_fred, which is in this same file. The guts of this though is again done using a function in shared_functions_setup.py (imported as setup) and depends on the luminosity function, which is also constructed using a function in that file with parameters which are defined in the params file for that survey setup.

At this point we should have a theory calculation of gamma_IA. We may also still want theory calculations of things like boost factor, F, Sigma_c^IA ^-1 etc. In this script, these get called as part of the next bit which computes theoretical covariance matrices, but realistically you may want to take the relevant functions and move them to a new script so that they could be called individually. 

F and Sigma_C quantities are both computed directly in this file, using the functions get_F and get_SigmaC_avg (although we want to check to make sure that does the average the way we want). If you look at get_Boost, however, you will notice that all this function does is load a file. This file is, for some reason, computed in a seperate script, get_boosts_full.py. It looks like I commented out a bunch of the Boosts for different samples at some point but this is probably just to speed because I only wanted to do one of them at that moment; in principle these should be uncommented.

You will notice that this file relies on importing a 2halo and 1halo 3d matter correlation function. These were in this case precomputed using a different file: run_local_1.py (you need to set the SURVEY here too). Here, I got the 2halo one from class plus fftlog but I would recommend using CCL for this now (at the time we didn't have this capability). For the 1halo term, I'm calling functions from shared_functions_wlp_wls.py, as you can see, to get the 1halo power spectrum and then using FFTlog to Fourier transform these power spectra into 1halo correlation functions terms. There is probably a better way to do this. It's pretty gross. Possibly some of the new CCL halomodel functionality might be useful here. 









