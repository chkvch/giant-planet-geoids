# fluid-planet-geoids
A model for calculating shapes of constant-pressure surfaces in differentially rotating giant planet atmospheres.

<img src="/figures/uranus_optimize_rpol.png" width="500" />

The method is based on integrating a tangent to the local vertical, which is modified by latitude-dependent rotation and the planet's multipolar gravity field. This type of calculation has long been applied to analyses of radio occultation experiments by spacecraft, e.g.,
- Lindal, Sweetnam, and Eshleman (1985, AJ); [The atmosphere of Saturn - an analysis of the Voyager radio occultation measurements](https://scixplorer.org/abs/1985AJ.....90.1136L/abstract)
- Lindal (1992, AJ); [The Atmosphere of Neptune: an Analysis of Radio Occultation Data Acquired with Voyager 2](https://scixplorer.org/abs/1992AJ....103..967L/abstract)
- Buccino, Helled, Parisi, Hubbard, and Folkner (2020, JGR: Planets); [Updated Equipotential Shapes of Jupiter and Saturn Using Juno and Cassini Grand Finale Gravity Science Measurements](https://scixplorer.org/abs/2020JGRE..12506354B/abstract)
- Galanti, Kaspi, and Guillot (2023, GRL); [The Shape of Jupiter and Saturn Based on Atmospheric Dynamics, Radio Occultations and Gravity Measurements](https://scixplorer.org/abs/2023GeoRL..5002321G/abstract)

This version was implemented to study the shapes of Uranus and Neptune with up-to-date knowledge of their atmospheric rotation profiles and gravity fields. 

## Quickstart
If you are interested in the outputs of models presented in Mankovich et al. (2026, PSJ submitted; doi tbd), these are available in the in the `models/` subdirectory. 

To run the code, clone this repository and make sure that dependencies are installed, e.g., with 
```conda env create -f environment.yaml && conda activate geoid```. (If you don't have conda or python set up yet, try [miniforge](https://github.com/conda-forge/miniforge).) 

Then run the basic unit tests: `python test.py -v`. If all goes well, you can open the Jupyter notebook [`uranus_shape.ipynb`](uranus_shape.ipynb) to reproduce results from our paper. If you are not set up with Jupyter, the main program in `geoid.py` also contains a minimal run of a Uranus shape model that may be adapted to your needs.

## Wind data
The wind profiles adopted for Uranus, Neptune, Jupiter, and Saturn are given in [`data/`](data/) and associated routines in [`wind_profiles.py`](wind_profiles.py). The original profiles are sourced from the following publications:
- Sromovsky, de Pater, Fry, Hammel, and Marcus (2015, Icarus 258); [High S/N Keck and Gemini AO imaging of Uranus during 2012-2014: New cloud patterns, increasing activity, and improved wind measurements](https://scixplorer.org/abs/2015Icar..258..192S/abstract)
- Karkoschka (2015, Icarus 250); [Uranus' southern circulation revealed by Voyager 2: Unique characteristics](https://scixplorer.org/abs/2015Icar..250..294K/abstract)
- Tollefson, de Pater, Marcus, Luszcz-Cook, Sromovsky, Fry, Fletcher, and Wong (2018, Icarus 311); [Vertical wind shear in Neptune's upper atmosphere explained with a modified thermal wind equation](https://scixplorer.org/abs/2018Icar..311..317T/abstract)
- Tollefson, Wong, de Pater, Simon, Orton, Rogers, Atreya, Cosentino, Januszewski, Morales-Juberías, and Marcus (2017, Icarus 296); [Changes in Jupiter's Zonal Wind Profile preceding and during the Juno mission](https://scixplorer.org/abs/2017Icar..296..163T/abstract)
- García-Melendo, Pérez-Hoyos, Sánchez-Lavega, and Hueso (2011, Icarus 215); [Saturn's zonal wind profile in 2004-2009 from Cassini ISS images and its long-term variability](https://scixplorer.org/abs/2011Icar..215...62G/abstract)

## Attribution
If this code was useful to you, please cite Mankovich et al. (2026, PSJ submitted; doi tbd). 

## License
This code is licensed under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).