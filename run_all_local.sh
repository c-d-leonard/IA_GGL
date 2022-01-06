#/bin/bash

# Overarching script to run everything for IA method paper
# other than things that need to be pre-run on coma. 

# Get 1halo and 2halo power spectrum terms that will need FFTing
#python run_local_1.py

#exit()

# FFT and save the results.
cp ./txtfiles/1halo_terms/P*Planck18pars.txt ~/Software/FFTLog/test/
cp ./txtfiles/halofit_Pk/P*Planck18pars.txt ~/Software/FFTLog/test/
cd ~/Software/FFTLog/

#make test_gg_1h.out
make test_gg_1h_multifile.out
make test_gm_1h.out
make test_gm_2h.out
make test_gg_2h_multifile.out
make clean

cd ~/Research/IA_measurement_GGL/IA_GGL/
mv ~/Software/FFTLog/xi*1h*Planck18pars.txt ./txtfiles/xi_1h_terms/
mv ~/Software/FFTLog/xi*2h*Planck18pars.txt ./txtfiles/halofit_xi/
rm ~/Software/FFTLog/test/P*_Planck18pars.txt

exit 1

# Get the boost
python get_boosts_full.py
exit 1

python test_shearratio_pz.py

exit 1

# Run the main files to get covariances and signal to noises in both methods
python constrain_IA_BlazekMethod.py
python constrain_IA_ShapesMethod_fixB.py

# Process the output to get the values for the sysz tables
python get_slope_sysz.py
# And the StoN plots we don't alredy have
python make_plots.py
