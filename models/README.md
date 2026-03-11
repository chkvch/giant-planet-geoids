# Description
This directory contains model output in plaintext format. 

For example, the primary Uranus shape model plotted in Figure 2a,b of the paper are given in [uranus_model_symmetric_wind.txt](uranus_model_symmetric_wind.txt). The rigidly rotating reference geoids corresponding to this model are given in [uranus_model_symmetric_wind_reference_geoid_17.25h.txt](uranus_model_symmetric_wind_reference_geoid_17.25h.txt) and similarly named files with different suffixes labeling different presumed rotation periods. 

Similar output is here for Neptune, Saturn, and Jupiter.

[statistical_samples/](statistical_samples/) contains output of the uncertainty quantification study described in Section 4 of the paper, including the output used to create Figures 5 and 6 from the paper. 
- Files with the suffix `_chain.txt` give the parameters (at least the polar radius, in some cases more), the equatorial radius, and the log-likelihood for each sample in the corresponding [`emcee.EnsembleSampler`](https://emcee.readthedocs.io/en/stable/user/sampler/) chain.
- `uranus_vary_wind_chain.txt.gz` contains many columns and was compressed to reduce its footprint; it can be expanded with `gunzip uranus_vary_wind_chain.txt.gz` before reading.
- The subdirectories contain a small number of full shape models randomly selected from their corresponding statistical samples.
- [uncertainties_summary.txt](statistical_samples/uncertainties_summary.txt) tabulates the standard deviations of the polar and equatorial radii obtained from each sample, as plotted in Figure 6 of the paper.