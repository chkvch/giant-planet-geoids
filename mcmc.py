''' implements likelihood functions for the mcmc experiments in section 4 of the paper '''

import geoid
import numpy as np

import occultation_data

def get_errors(r_pol, jn, r_ref, gm, omega, planet='uranus'):

    # calculate shape model
    g = geoid.geoid(r_pol, jn, r_ref, gm, omega=omega)

    # compare to data
    occs = getattr(occultation_data, planet) # dictionary of occultation latitudes and radii
    rocc = np.array([_r for occ, (_lat, _r) in occs.items()]) # array of just radii
    lat = np.array([_lat for occ, (_lat, _r) in occs.items()]) # array of just latitudes
    errors_km = np.array([ (_rocc - np.interp(_lat, g.lat[::-1], 1e-5 * g.r[::-1])) for _rocc, _lat in zip(rocc, lat)])

    return errors_km, g

### experiment 1: only vary r_pol and fit to uncertain occultations
def lnp_prior(theta):
    if theta[0] < 2e9:
        return -np.inf
    elif theta[0] > 3e9:
        return -np.inf
    else:
        return 0

def lnp(theta, jn, r_ref, gm, omega, sigma_rocc_km=5., debug=False, planet='uranus'): 
    # returns log likelihood such that product with lnp_prior is proportional to log posterior probability

    if np.isinf(lnp_prior(theta)): 
        return -np.inf, {}

    r_pol = theta[0]
    try:
        errors_km, g = get_errors(r_pol, jn, r_ref, gm, omega, planet)
    except Exception as e:
        if debug: raise
        print(*e.args)
        return -np.inf, {}
        
    return -np.sum(errors_km ** 2 / 2 / sigma_rocc_km ** 2), g



### experiment 2: vary r_pol, j2, j4

# Uranus gravity prior
# French et al. 2024, Icarus: ring occultations constrain Uranus's Jn
from scipy.stats import multivariate_normal
french_j2, french_dj2 = 3509.291e-6, 0.412e-6
french_j4, french_dj4 = -35.522e-6, 0.466e-6
correlation = 0.9861
gauss_jn_uranus = multivariate_normal(
    mean=np.array([french_j2, french_j4]),
    cov=np.array([
        [french_dj2 ** 2, correlation * french_dj2 * french_dj4], 
        [correlation * french_dj2 * french_dj4, french_dj4 ** 2]
        ])
)

# Neptune gravity prior
# Jacobson 2009. no j2-j4 correlation
j09_j2, j09_dj2 = 3408.43e-6, 4.5e-6
j09_j4, j09_dj4 = -33.4e-6, 2.9e-6
gauss_jn_neptune = multivariate_normal(
    mean=np.array([j09_j2, j09_j4]),
    cov=np.array([
        [j09_dj2 ** 2, 0],
        [0, j09_dj4 ** 2]
    ])
)

def lnp_prior_vary_jn(theta, planet='uranus'):
    if theta[0] < 2e9:
        return -np.inf
    elif theta[0] > 3e9:
        return -np.inf
    else: # multivariate Gaussian prior for J2-J4
        gauss_jn = {'uranus':gauss_jn_uranus, 'neptune':gauss_jn_neptune}[planet]
        return gauss_jn.logpdf(theta[1:]) - gauss_jn.logpdf(gauss_jn.mean) # second term changes normalization so that lnp=0 at mean

def lnp_vary_jn(theta, r_ref, gm, omega, sigma_rocc_km=5., debug=False, planet='uranus'): # here j2, j4 are passed inside theta, alongside r_pol
    # prior: uniform in r_pol and Gaussian in Jn
    _lnp = lnp_prior_vary_jn(theta)
    if np.isinf(_lnp):
        return -np.inf, {}

    r_pol = theta[0]
    jn = *theta[1:], 0. # take j6=0
    try:
        errors_km, g = get_errors(r_pol, jn, r_ref, gm, omega, planet)
    except Exception as e:
        if debug: raise
        print(*e.args)
        return -np.inf, {}
        
    # on top of Gaussian prior in Jn, likelihood function penalizes mismatch to occultations
    _lnp -= np.sum(errors_km ** 2 / 2 / sigma_rocc_km ** 2)

    return _lnp, g



### experiment 3: vary r_pol and coefficients in wind fits

# Uranus: Legendre coefficients in Sromovsky+ 2015 symmetric wind profile fit, Table 3
cn = np.array([-1.245225, -3.582487, -0.118185, 0.848593, 0.315199, -0.188857, -0.263077, -0.026728, 0.104192, 0.059944])
dcn = np.array([0.0079,0.0162,0.0194,0.0221,0.0235,0.0283,0.0359,0.0371,0.0432,0.0362])
# gauss_cn = norm(loc=cn, scale=dcn)
cov = np.zeros((len(cn), len(cn)))
np.fill_diagonal(cov, dcn ** 2)
gauss_cn = multivariate_normal(mean=cn, cov=cov)

from scipy.stats import Covariance
gauss_t18_voy = multivariate_normal(mean=np.array([-398., 1.88e-1, -1.20e-5]), cov=Covariance.from_diagonal(np.array([1., 1.40e-2, 3.00e-6]) ** 2))
gauss_t18_h13 = multivariate_normal(mean=np.array([-325., 1.58e-1, -1.21e-5]), cov=Covariance.from_diagonal(np.array([16., 2.20e-2, 4.67e-6]) ** 2))
gauss_t18_h14 = multivariate_normal(mean=np.array([-292., 1.45e-1, -1.18e-5]), cov=Covariance.from_diagonal(np.array([29., 4.91e-2, 1.11e-5]) ** 2))
gauss_t18_k14 = multivariate_normal(mean=np.array([-433., 2.40e-1, -2.73e-5]), cov=Covariance.from_diagonal(np.array([56., 7.88e-2, 1.90e-5]) ** 2))
gauss_norm = multivariate_normal(mean=np.ones(3), cov=Covariance.from_diagonal(np.ones(3)))

def lnp_prior_vary_wind(theta, planet='uranus', wind_option=None):
    if theta[0] < 2e9:
        return -np.inf
    elif theta[0] > 3e9:
        return -np.inf
    else:
        if planet == 'uranus':
            cn_theta = theta[1:]
            assert cn.shape == cn_theta.shape, 'got wrong shape for Sromovsky+2015 wind coefficients'
            return gauss_cn.logpdf(cn_theta) - gauss_cn.logpdf(cn) # second term changes normalization so that lnp=0 at mean
        elif planet == 'neptune':
            abc = theta[1:]
            assert len(abc) == 3, 'got wrong shape for Tollefson+2018 wind coefficients'
            # gauss = {'voyager':gauss_t18_voy, 'h-band 2014':gauss_t18_h14, 'k-band 2014':gauss_t18_k14}[wind_option]
            # return gauss.logpdf(abc) - gauss.logpdf(gauss.mean)
            return gauss_norm.logpdf(abc) - gauss_norm.logpdf(gauss_norm.mean)
        else:
            raise ValueError(f'planet {planet} not understood')

_phi_grid = np.linspace(-np.pi/2, np.pi/2, 1001) # just a reasonably dense grid for setting up interpolating functions
def lnp_vary_wind(theta, jn, r_ref, gm, sigma_rocc_km=5., debug=False, planet='uranus', wind_option=None): # here j2, j4 are passed inside theta, alongside r_pol
    from scipy.special import legendre as Pn
    from scipy.interpolate import interp1d
    
    # prior: uniform in r_pol and Gaussian in wind fit spectral/polynomial coefficients.
    if planet == 'uranus': # construct Uranus 1-bar rotation profile for the params theta passed by sampler
        # Uranus reference ellipsoid
        rp = 24973e5
        re = 25559e5
        beta = rp / re
        _th = np.arctan( beta ** -2. * np.tan(_phi_grid) ) # planetocentric grid phi to planetographic latitude theta, rad -- not to be confused with parameter vector theta

        n = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18]) # n values for coefficients in S15 Table 3
        dphidt = np.zeros_like(_th)
        cn_model = theta[1:]
        for nval, cval in zip(n, cn_model):
            dphidt += cval * Pn(nval)(np.sin(_th))

        _omega_tot = -dphidt * np.pi / 180. / 3600 + np.pi * 2 / (17.24 * 3600) # add SIII rotation to arrive at total angular velocity
        omega = interp1d(_phi_grid, _omega_tot, bounds_error=None, fill_value='extrapolate')
        
        _lnp = lnp_prior_vary_wind(theta, planet='uranus')

    elif planet == 'neptune':
        phi_deg = _phi_grid * 180. / np.pi

        # assume a, b, c being sampled are regularized: rescale and translate back to physical a, b, c
        gauss = {'voyager':gauss_t18_voy, 'h-band 2013':gauss_t18_h13, 'h-band 2014':gauss_t18_h14, 'k-band 2014':gauss_t18_k14}[wind_option]
        a, b, c = theta[1:] * np.sqrt(gauss.cov.diagonal()) + gauss.mean

        u_wind = a + b * phi_deg ** 2 + c * phi_deg ** 4
        u_wind *= 1e2 # cm s^-1
        cutoff = 75. # cosine cutoff
        isort = np.argsort(phi_deg)
        u_wind[phi_deg >  cutoff] = np.interp( cutoff, phi_deg[isort], u_wind[isort])
        u_wind[phi_deg < -cutoff] = np.interp(-cutoff, phi_deg[isort], u_wind[isort])
        u_wind[phi_deg >  cutoff] *=  np.cos(np.pi / 2 * (phi_deg[phi_deg >  cutoff] - cutoff) / (90. - cutoff))
        u_wind[phi_deg < -cutoff] *= -np.cos(np.pi / 2 * (phi_deg[phi_deg < -cutoff] - cutoff) / (90. - cutoff))
        # convert to profile of total rotation frequency using T18's assumed shape model and System III rotation
        rp = 24342e5
        re = 24766e5
        r_ellipsoid = np.sqrt(np.cos(_phi_grid) ** 2 / re ** 2 + np.sin(_phi_grid) ** 2 / rp ** 2) ** -1.
        r_cylindrical = r_ellipsoid * np.cos(_phi_grid)
        omega_rigid = np.pi * 2 / (16.11 * 3600) # planetary rotation rate assumed in measuring the wind velocities
        omega_wind = u_wind / r_cylindrical
        # because of division by small number, omega may be wonky at the two polar zones, simply assume same rotation period as neighbor zone
        omega_wind[0] = omega_wind[1]
        omega_wind[-1] = omega_wind[-2]
        omega = interp1d(_phi_grid, omega_rigid + omega_wind, bounds_error=None, fill_value='extrapolate')
        # omega = interp1d(_phi_grid, omega_rigid * np.ones_like( omega_wind), bounds_error=None, fill_value='extrapolate')

        _lnp = lnp_prior_vary_wind(theta, planet='neptune', wind_option=wind_option)

    else:
        raise ValueError(f'planet {planet} not understood')

    if np.isinf(_lnp):
        # if debug: raise ValueError('outside prior')
        return -np.inf, {}

    r_pol = theta[0]
    try:
        errors_km, g = get_errors(r_pol, jn, r_ref, gm, omega, planet)
    except Exception as e:
        if debug: raise
        print(*e.args)
        return -np.inf, {}
        
    _lnp -= np.sum(errors_km ** 2 / 2 / sigma_rocc_km ** 2)

    g._omega = omega

    return _lnp, g