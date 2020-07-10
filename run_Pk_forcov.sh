#/bin/bash

# Overarching script to power spectra for the cosmic variance covariance terms and transfer them to coma.

# BEFORE RUNNING THIS CHANGE THE endfile PARAMETER IN ALL THESE FILES. 

# Get the power spectra for gamma_t (our method)
#python gamma_t_Variance_SDSS_Bl.py
#python gamma_t_Variance_LSST_Bl.py

# Get the power spectra for Delta_Sigma (Blazek method)

#python DeltaSigma_Variance_SDSSA.py
#python DeltaSigma_Variance_SDSSB.py

python DeltaSigma_Variance_LSSTA.py
python DeltaSigma_Variance_LSSTB.py

# This command cannot actually be run in this file because it will require a password - just paste it in the terminal.

#scp ./txtfiles/Pgg/*HOD_fully_updated* ./txtfiles/Pgk/*HOD_fully_updated* ./txtfiles/Pkk/*HOD_fully_updated* ./txtfiles/PggPkk/*HOD_fully_updated* ./txtfiles/const/*HOD_fully_updated* ./txtfiles/SigCsq/*HOD_fully_updated* danielll@coma.hpc1.cs.cmu.edu:/home/danielll/IntrinsicAlignments/txtfiles/

#scp ./txtfiles/Pgg/*HOD_fully_updated* ./txtfiles/Pgk/*HOD_fully_updated* ./txtfiles/Pkk/*HOD_fully_updated* ./txtfiles/PggPkk/*HOD_fully_updated* ./txtfiles/const/*HOD_fully_updated* ./txtfiles/SigCsq/*HOD_fully_updated* danielll@coma.hpc1.cs.cmu.edu:/home/danielll/IntrinsicAlignments/txtfiles/

