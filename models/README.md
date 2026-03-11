# Description
This directory contains model output in plaintext format. 

Taking Uranus as an example, the primary shape models are given in [uranus_model_symmetric_wind.txt](uranus_model_symmetric_wind.txt) and [uranus_model_composite_wind.txt](uranus_model_composite_wind.txt). 

Reference geoid shapes corresponding to the symmetric wind model are given in [uranus_model_symmetric_wind_reference_geoid_17.25h.txt](uranus_model_symmetric_wind_reference_geoid_17.25h.txt) and similarly named files for a handful of reference rotation periods.

[statistical_samples/](statistical_samples/) contains output of the uncertainty quantification study described in Section 4 of the paper. 
    - Files with the suffix `_chain.txt` give the parameters (at least the polar radius, in some cases more), the equatorial radius, and the log-likelihood for each sample in the corresponding [`emcee.EnsembleSampler`](https://emcee.readthedocs.io/en/stable/user/sampler/) chain.
    - `uranus_vary_wind_chain.txt.gz` contains many columns and was compressed to reduce its footprint; it can be expanded with `gunzip uranus_vary_wind_chain.txt.gz` before reading.
    - The subdirectories contain a small number of full shape models randomly selected from their corresponding statistical samples.