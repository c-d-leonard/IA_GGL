#/bin/bash

# Overarching script to run everything for IA method paper
# other than things that need to be pre-run on coma. 

# Get 1halo and 2halo power spectrum terms that will need FFTing
python run_local_1.py

# FFT and save the results.
cp ./txtfiles/1halo_terms/P*fixDls.txt ~/Documents/CMU/Software/FFTLog-master-slosar/test/
cp ./txtfiles/halofit_Pk/P*fixDls.txt ~/Documents/CMU/Software/FFTLog-master-slosar/test/
cd ~/Documents/CMU/Software/FFTLog-master-slosar/

make test_gg_1h.out
make test_gg_1h_multifile.out
make test_gm_1h.out
make test_gm_2h.out
make test_gg_2h_multifile.out
make clean

cd ~/Dropbox/CMU/Research/Intrinsic_Alignments/
mv ~/Documents/CMU/Software/FFTLog-master-slosar/xi*1h*fixDls.txt ./txtfiles/xi_1h_terms/
mv ~/Documents/CMU/Software/FFTLog-master-slosar/xi*2h*fixDls.txt ./txtfiles/halofit_xi/
rm ~/Documents/CMU/Software/FFTLog-master-slosar/test/P*_fixDls.txt

# Get the boost
python get_boosts_full.py

# Run the main files to get covariances and signal to noises in both methods
python constrain_IA_BlazekMethod.py
python constrain_IA_ShapesMethod_fixB.py

# Process the output to get the values for the sysz tables
python get_slope_sysz.py
# And the StoN plots we don't alredy have
python make_plots.py
