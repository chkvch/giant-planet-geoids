# latitudes are planetocentric

# Uranus occultations
uranus = {'voyager 2 ingress': (-2.9, 25555.3), 'voyager 2 egress': (-6.6, 25550.8)} # Lindal et al. 1987
# uranus = {'voyager 2 ingress': (-2.9, 25556.7), 'voyager 2 egress': (-6.6, 25550.7)} # Caruso et al.'s 2025 AGU poster

# Neptune occultations, Lindal 1992
neptune = {'voyager 2 egress': (-40.07, 24599.58)}

if False:
    # Jupiter occultations, Lindal et al. 1981, 100-mbar surface
    # fair warning! these occultation radii were pulled from the pixellated plot in Lindal et al. 1981's Figure 7.
    # I did my best but expect these are only accurate to within 5 km, similar to Lindal's estimated standard deviation of 4 km.
    # identifying which are ingress versus egress from Lindal is a squinty endeavor but they are given in Buccino et al. (2020 JGR:Planets) Figure 1
    jupiter = {
        'pioneer 10 ingress':(60.94052, 67.92648e3),
        'pioneer 10 egress':(28.19087, 70.39084e3),
        'pioneer 11 egress':(19.70111, 70.93895e3),
        'voyager 1 egress':(0.16012, 71.53487e3),
        'voyager 1 ingress':(-10.17727, 71.37230e3),
        'voyager 2 ingress':(-70.62033, 67.36508e3),
    }
else:
    # instead take values from Helled et al. 2009's Table 2. they also scraped from Lindal et al. 1981, with slightly different results.
    # these are the same points that Galanti et al. 2023 (GRL) compared to.
    jupiter = {
        'pioneer 10 ingress':(60.3, 67933.93),
        'pioneer 10 egress':(28., 70415.08),
        'pioneer 11 egress':(19.8, 70943.95),
        'voyager 1 egress':(0.07, 71538.61),
        'voyager 1 ingress':(-10.1, 71378.73),
        'voyager 2 ingress':(-71.8, 67293.64),
    }

# 100-mbar radii from Juno occultations
import numpy as np
juno_occs = np.genfromtxt('data/juno_occultations.txt', skip_header=2, names=('occ', 'lat_pc', 'r_km', 'lon'), dtype=None)
[jupiter.update({str(occ):(lat_pc, r_km)}) for occ, lat_pc, r_km in zip(juno_occs['occ'], juno_occs['lat_pc'], juno_occs['r_km'])]

# Saturn 100-mbar radii from radio occultations, Lindal et al. 1985
saturn = {
    'pioneer 11 ingress':(-9.7, 60138.0),
    'voyager 1 egress':(-2.4, 60353.5),
    'voyager 1 ingress':(-71.2, 54948.4),
    'voyager 2 ingress':(30.5, 58545.4),
    'voyager 2 egress':(-26.6, 58913.4),
}