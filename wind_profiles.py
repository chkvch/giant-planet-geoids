import numpy as np
from scipy.interpolate import interp1d

_phi_grid = np.linspace(-np.pi/2, np.pi/2, 1001) # just a reasonably dense grid for setting up interpolating functions

def uranus_omega_tot_interpolant(option='symmetric', randomize=False):
    # specify total angular frequency of rotation as a function of latitude.
    # for uranus, the rotation profile is already given by S15 as a polynomial function of latitude, so it's not really
    # necessary to specify on a grid and then recreate interpolant, but this is a convenient way of setting it up just once.
    omega_tot = uranus_omega_tot(_phi_grid, option=option, randomize=randomize)
    return interp1d(_phi_grid, omega_tot, bounds_error=None, fill_value='extrapolate')

def neptune_omega_tot_interpolant(option='voyager', pole_attenuation='cosine', randomize=False):
    omega_tot = neptune_omega_tot(_phi_grid, option=option, pole_attenuation=pole_attenuation, randomize=randomize)
    return interp1d(_phi_grid, omega_tot, bounds_error=None, fill_value='extrapolate')

def jupiter_omega_tot_interpolant(option='2016_PJ03_201803001433999163'):
    omega_tot = jupiter_omega_tot(_phi_grid, option=option)
    return interp1d(_phi_grid, omega_tot, bounds_error=None, fill_value='extrapolate')

def saturn_omega_tot_interpolant(option='garciamelendo2011'):
    omega_tot = saturn_omega_tot(_phi_grid, option=option)
    return interp1d(_phi_grid, omega_tot, bounds_error=None, fill_value='extrapolate')

def saturn_omega_tot_interpolant_by_cylindrical_radius(option='garciamelendo2011'): # experimenting with a version that preserves omega on axial columns
    omega_tot = saturn_omega_tot(_phi_grid, option=option)

    # ascribe garcia-melendo velocities to the reference ellipsoid with IAU equatorial/polar radii
    re = 60268e5
    rp = 54364e5
    r_ellipsoid = np.sqrt(np.cos(_phi_grid) ** 2 / re ** 2 + np.sin(_phi_grid) ** 2 / rp ** 2) ** -1.
    r_cylindrical = r_ellipsoid * np.cos(_phi_grid)
    r_cylindrical[0] = 0.
    r_cylindrical[-1] = 0.

    # return interp1d(r_cylindrical, omega_tot, bounds_error=False, fill_value=0.)
    n = len(r_cylindrical) // 2 # use northern hemisphere only
    return interp1d(r_cylindrical[n:], omega_tot[n:], bounds_error=False, fill_value='extrapolate')
    # return interp1d(r_cylindrical, omega_tot)

def uranus_omega_tot(phi, option='symmetric', randomize=False):
    ''' 
    if option == 'symmetric', use even polynomial fit from Sromovsky et al. 2015, https://doi.org/10.1016/j.icarus.2015.05.029, Equations 2-4. 
    if option == 'composite',  use asymmetric composite fit described in Section 4.5 and Table 6.
    '''

    # reference ellipsoid defined in S15's Equation 4
    rp = 24973e5
    re = 25559e5
    beta = rp / re
    theta = np.arctan( beta ** -2. * np.tan(phi) ) # user-specified planetocentric grid phi to planetographic latitude theta, rad
    # r_cylindrical = re / np.sqrt(1. + (rp / re) ** 2 * np.tan(theta) ** 2) # distance from rotation axis, cm

    if option == 'composite': # get drift rates directly from Table 6
        if randomize:
            raise ValueError('cannot draw random profile with composite option: no errors available for this profile.')
        theta_table = np.linspace(90, -90, 181) # planetographic latitude from table
        dphidt_table = - np.genfromtxt('data/sromovsky2015_table_6.txt') # minus sign gives eastward drift to be consistent with dphidt from Eq. 2 used for the other wind profiles
        isort = np.argsort(theta_table)
        dphidt = interp1d(theta_table[isort], dphidt_table[isort], bounds_error=None, fill_value='extrapolate')(theta * 180 / np.pi)
    elif option =='symmetric': # Equation 2 and Table 7
        from scipy.special import legendre as Pn
        n = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
        c = np.array([-1.245225, -3.582487, -0.118185, 0.848593, 0.315199, -0.188857, -0.263077, -0.026728, 0.104192, 0.059944])
        dc = np.array([0.0079,0.0162,0.0194,0.0221,0.0235,0.0283,0.0359,0.0371,0.0432,0.0362])

        dphidt = np.zeros_like(theta)
        if randomize: # assume gaussian-distributed coefficients
            from scipy.stats import norm
            cc = norm(loc=c, scale=dc).rvs()
            for nval, cval in zip(n, cc):
                dphidt += cval * Pn(nval)(np.sin(theta))
        else:
            for nval, cval in zip(n, c):
                dphidt += cval * Pn(nval)(np.sin(theta))
    else:
        raise ValueError(f'wind option {option} not recognized')

    # per S15 Section 4.1, dphi/dt is the *eastward* longitudinal drift rate; for Uranus's *westward* rotation in the IAU convention, dphi/dt and bulk spin (SIII) have opposite signs.
    Omega_tot = dphidt * np.pi / 180. / 3600 - np.pi * 2 / (17.24 * 3600) # add SIII rotation to arrive at total angular velocity
    return Omega_tot

def neptune_wind_profile_tollefson2018(phi, option='voyager', pole_attenuation='cosine', randomize=False):
    # Tollefson et al. 2018 Table 1. assuming quoted uncertainties are 1-sigma because that's what is shown in their plots
    # assume phi given in rad, convert to deg as in Tollefson's polynomials
    phi_deg = phi * 180. / np.pi
    if randomize:
        # assume gaussian-distributed coefficients.
        from scipy.stats import norm
        match option:
            case 'voyager':
                a = norm(loc=-398.,     scale=1.        ).rvs()
                b = norm(loc= 1.88e-1,  scale=1.40e-2   ).rvs()
                c = norm(loc=-1.20e-5,  scale=3.00e-6   ).rvs()
            case 'h-band 2013':
                a = norm(loc=-325.,     scale=16.       ).rvs()
                b = norm(loc= 1.58e-1,  scale=2.20e-2   ).rvs()
                c = norm(loc=-1.21e-5,  scale=4.67e-6   ).rvs()
            case 'k-band 2013':
                a = norm(loc=-415.,     scale=42.       ).rvs()
                b = norm(loc= 2.35e-1,  scale=5.34e-2   ).rvs()
                c = norm(loc=-2.23e-5,  scale=1.14e-5   ).rvs()
            case 'h-band 2014':
                a = norm(loc=-292.,     scale=29.       ).rvs()
                b = norm(loc= 1.45e-1,  scale=4.91e-2   ).rvs()
                c = norm(loc=-1.18e-5,  scale=1.11e-5   ).rvs()
            case 'k-band 2014':
                a = norm(loc=-433.,     scale=56.       ).rvs()
                b = norm(loc= 2.40e-1,  scale=7.88e-2   ).rvs()
                c = norm(loc=-2.73e-5,  scale=1.90e-5   ).rvs()
    else: # get profile from the centroid fits
        match option:
            case 'voyager':
                a = -398.
                b =  1.88e-1
                c = -1.20e-5
            case 'h-band 2013':
                a = -325.
                b =  1.58e-1
                c = -1.21e-5
            case 'k-band 2013':
                a = -415.
                b =  2.35e-1
                c = -2.23e-5
            case 'h-band 2014':
                a = -292.
                b =  1.45e-1
                c = -1.18e-5
            case 'k-band 2014':
                a = -433.
                b =  2.40e-1
                c = -2.73e-5

    u_wind = a + b * phi_deg ** 2 + c * phi_deg ** 4
    u_wind *= 1e2 # cm s^-1

    # attenuate near poles as in French et al. 1998
    match pole_attenuation:
        case 'exp':
            f = np.ones_like(phi_deg)
            f *= (1. - np.exp(10 * np.pi / 180 * ( phi_deg - 90.)))
            f *= (1. - np.exp(10 * np.pi / 180 * (-phi_deg - 90.)))
            u_wind *= f
        case 'cosine':
            cutoff = 75.
            isort = np.argsort(phi_deg)
            u_wind[phi_deg >  cutoff] = np.interp( cutoff, phi_deg[isort], u_wind[isort])
            u_wind[phi_deg < -cutoff] = np.interp(-cutoff, phi_deg[isort], u_wind[isort])
            u_wind[phi_deg >  cutoff] *=  np.cos(np.pi / 2 * (phi_deg[phi_deg >  cutoff] - cutoff) / (90. - cutoff))
            u_wind[phi_deg < -cutoff] *= -np.cos(np.pi / 2 * (phi_deg[phi_deg < -cutoff] - cutoff) / (90. - cutoff))
        case 'linear':
            cutoff = 75.
            isort = np.argsort(phi_deg) # sorting makes sure the interp (snap to values at cutoff) works regardless of supplied phi increasing or decreasing
            u_wind[phi_deg >  cutoff] = np.interp( cutoff, phi_deg[isort], u_wind[isort])
            u_wind[phi_deg < -cutoff] = np.interp(-cutoff, phi_deg[isort], u_wind[isort])
            u_wind[phi_deg >  cutoff] *= 1. - ( phi_deg[phi_deg >  cutoff] - cutoff) / (90. - cutoff)
            u_wind[phi_deg < -cutoff] *= 1. - (-phi_deg[phi_deg < -cutoff] - cutoff) / (90. - cutoff)
        case None:
            pass

    return u_wind

def neptune_omega_tot(phi, option='voyager', pole_attenuation='cosine', randomize=False):
    # convert Tollefson et al.'s wind speed profile to a profile of total rotation frequency using their assumed shape model and System III rotation
    rp = 24342e5
    re = 24766e5
    r_ellipsoid = np.sqrt(np.cos(phi) ** 2 / re ** 2 + np.sin(phi) ** 2 / rp ** 2) ** -1.
    r_cylindrical = r_ellipsoid * np.cos(phi)
    
    omega_rigid = np.pi * 2 / (16.11 * 3600) # planetary rotation rate assumed in measuring the wind velocities
    omega_wind = neptune_wind_profile_tollefson2018(phi, option=option, pole_attenuation=pole_attenuation, randomize=randomize) / r_cylindrical
    # because of division by small number, omega may be wonky at the two polar zones, simply assume same rotation period as neighbor zone
    omega_wind[0] = omega_wind[1]
    omega_wind[-1] = omega_wind[-2]

    return omega_rigid + omega_wind

def jupiter_wind_profile_tollefson2017(phi, option='2016_PJ03_201803001433999163'):

    skip_header = 20 if '2016_PJ03' in option else 19
    data = np.genfromtxt(f'data/tollefson2017_supplementary/ZWP_j{option}.txt', skip_header=skip_header, names=('lat', 'u', 'sigma_u'))

    # table supplies planetographic latitude.
    # here convert to planetocentric latitude using an ellipsoid with equatorial and polar radii from Lindal et al. 1981.
    re = 71541e5
    rp = 66896e5
    beta = rp / re
    phi_planetographic = np.arctan( beta ** -2. * np.tan(phi) )
    isort = np.argsort(data['lat'])
    u_wind = interp1d(data['lat'][isort] * np.pi / 180, data['u'][isort], bounds_error=False, fill_value=0.)(phi_planetographic) * 1e2 # cm s^-1
    return u_wind

def jupiter_omega_tot(phi, option='2016_PJ03_201803001433999163'):
    re = 71541e5
    rp = 66896e5
    r_ellipsoid = np.sqrt(np.cos(phi) ** 2 / re ** 2 + np.sin(phi) ** 2 / rp ** 2) ** -1.
    r_cylindrical = r_ellipsoid * np.cos(phi)
    
    p_rot_h_truth = 9. + 55. / 60 + 29.71 / 3600 # e.g., Dessler 1983
    omega_rigid = np.pi * 2 / (p_rot_h_truth * 3600) # planetary rotation rate assumed in measuring the wind velocities
    omega_wind = jupiter_wind_profile_tollefson2017(phi, option=option) / r_cylindrical

    return omega_rigid + omega_wind

def saturn_wind_profile_garciamelendo2011(phi):
    data = np.genfromtxt('data/saturn_wind1.txt', skip_header=8, names=('lat_pc', 'lat_pg', 'u', 'sigma_u', 'npts'))
    u_wind = interp1d(data['lat_pc'] * np.pi / 180, data['u'], bounds_error=False, fill_value=0.)(phi) * 1e2 # cm s^-1

    # south pole velocities remain large, zero out velocities close to the poles. this may influence results for Saturn.
    p = phi * 180 / np.pi
    u_wind[p < -80] = 0.
    u_wind[p > 80] = 0.

    return u_wind

def saturn_omega_tot(phi, option='garciamelendo2011'):
    re = 60268e5
    rp = 54364e5
    r_ellipsoid = np.sqrt(np.cos(phi) ** 2 / re ** 2 + np.sin(phi) ** 2 / rp ** 2) ** -1.
    r_cylindrical = r_ellipsoid * np.cos(phi)

    if option == 'garciamelendo2011':
        omega_rigid = 810.7939024 * np.pi / 180 / 24 / 3600 # Garcia-Melendo et al. 2011 used System III rotation (Desch and Kaiser 1981) as reference
        omega_wind = saturn_wind_profile_garciamelendo2011(phi) / r_cylindrical
    else:
        raise ValueError

    return omega_rigid + omega_wind

# the following data-related routines are only here to plot comparisons to the fits actually used by the models.
def get_sromovsky2015_data():
    # planetographic latitude center, avg, stddev [deg], drift rate [deg E / h], stddev [deg/h], wind speed, stddev [m/s], bin count
    names = 'planetographic_latitude_center', 'planetographic_latitude_avg', 'planetographic_latitude_stddev', 'eastward_drift_rate', 'eastward_drift_rate_error', 'eastward_drift_rate_stddev', 'wind_speed', 'wind_speed_stddev', 'bin_count'
    data = np.genfromtxt(f'data/sromovsky2015_tables_3_4.txt', skip_header=1, names=names, delimiter=',')
    # print(data['planetographic_latitude_avg'])
    planetographic_latitude_avg_rad = data['planetographic_latitude_avg'] * np.pi / 180
    # calculate planetocentric latitude for S15's assumed shape model
    re = 25559.
    rp = 24973.
    beta = rp / re
    srom_planetocentric_latitude_avg = np.arctan( beta ** 2 * np.tan(planetographic_latitude_avg_rad))
    srom_planetocentric_latitude_stddev = data['planetographic_latitude_stddev'] * np.pi / 180 * beta ** 2 * np.cos(srom_planetocentric_latitude_avg) ** 2 / np.cos(planetographic_latitude_avg_rad) ** 2

    # do not calculate velocities; leave it to the user
    return {'phi':srom_planetocentric_latitude_avg, 'dtheta':srom_planetocentric_latitude_stddev, 'eastward_drift_rate':data['eastward_drift_rate'], 'eastward_drift_rate_stddev':data['eastward_drift_rate_stddev']}

def get_karkoschka2015_data():
    names = 'planetographic_latitude', 'tracking_period_h', 'correlation_period_h'
    data = np.genfromtxt(f'data/karkoschka2015_table_4.txt', skip_header=1, names=names)
    planetographic_latitude_rad = data['planetographic_latitude'] * np.pi / 180
    re = 25559.
    rp = 24973.
    beta = rp / re
    planetocentric_latitude = np.arctan( beta ** 2 * np.tan(planetographic_latitude_rad) )

    return {'phi':planetocentric_latitude, 'tracking_period_h':data['tracking_period_h'], 'correlation_period_h':data['correlation_period_h']}
