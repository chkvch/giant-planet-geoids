import numpy as np
from scipy.integrate import trapezoid, cumulative_trapezoid, solve_ivp
from scipy.interpolate import interp1d

from numpy.polynomial import Polynomial, Legendre
from numpy.polynomial.polynomial import polyval
from numba import jit

# first we make a bunch of pure functions so we can use just-in-time compilation for speed.
# the handling of Legendre polynomials is crude but we know we only care about the first couple of degrees anyway:
c2 = Legendre.basis(2).convert(kind=Polynomial).coef
c4 = Legendre.basis(4).convert(kind=Polynomial).coef
c6 = Legendre.basis(6).convert(kind=Polynomial).coef
c2p = Legendre.basis(2).convert(kind=Polynomial).deriv().coef
c4p = Legendre.basis(2).convert(kind=Polynomial).deriv().coef
c6p = Legendre.basis(2).convert(kind=Polynomial).deriv().coef

@jit()
def p2(x):
    return polyval(x, c2)

@jit()
def p4(x):
    return polyval(x, c4)

@jit()
def p6(x):
    return polyval(x, c6)

@jit()
def dp2(x):
    return polyval(x, c2p)

@jit()
def dp4(x):
    return polyval(x, c4p)

@jit()
def dp6(x):
    return polyval(x, c6p)

@jit(nopython=True)
def u_(r, mu, j2, j4, j6, r_ref, gm_tot, omega): # pseudopotential; only used by split version
    u =  j2 * (r_ref / r) ** 2 * p2(mu)
    u += j4 * (r_ref / r) ** 4 * p4(mu)
    u += j6 * (r_ref / r) ** 6 * p6(mu)
    u -= 1.
    u *= gm_tot / r
    u -= 0.5 * omega ** 2 * r ** 2 * (1. - mu ** 2)
    return u

@jit(nopython=True)
def gr_(r, mu, j2, j4, j6, r_ref, gm_tot, omega): # radial component of gravity
    gr = -gm_tot / r ** 2
    gr += gm_tot / r ** 2 * (1. + 2) * j2 * (r_ref / r) ** 2 * p2(mu)
    gr += gm_tot / r ** 2 * (1. + 4) * j4 * (r_ref / r) ** 4 * p4(mu)
    gr += gm_tot / r ** 2 * (1. + 6) * j6 * (r_ref / r) ** 6 * p6(mu)
    gr += 2. / 3 * omega ** 2 * r * (1. - p2(mu))
    return gr

@jit(nopython=True)
def gphi_(r, mu, j2, j4, j6, r_ref, gm_tot, omega): # latitudinal component of gravity
    gth = -1. / 3 * omega ** 2 * r * dp2(mu) * np.sqrt(1. - mu ** 2)
    gth -= gm_tot / r ** 2 * j2 * (r_ref / r) ** 2 * dp2(mu) * np.sqrt(1. - mu ** 2)
    gth -= gm_tot / r ** 2 * j4 * (r_ref / r) ** 4 * dp4(mu) * np.sqrt(1. - mu ** 2)
    gth -= gm_tot / r ** 2 * j6 * (r_ref / r) ** 6 * dp6(mu) * np.sqrt(1. - mu ** 2)
    return -gth # theta (our planetocentric latitude) and phi (Lindal notation planetocentric latitude) unit vectors point in opposite directions

@jit(nopython=True)
def psi_(r, mu, j2, j4, j6, r_ref, gm_tot, omega): # planetographic latitude = phi + psi
    return np.arctan(gphi_(r, mu, j2, j4, j6, r_ref, gm_tot, omega) / gr_(r, mu, j2, j4, j6, r_ref, gm_tot, omega))

class geoid:
    def __init__(self, r_pol, jn, r_ref, gm, npts=1001, omega=None, verbose=False, solve_ivp_tol=1e-10):

        '''
        computes a geoid by following the tangent to the isobaric surface.
        
        the original version (included below) followed Lindal et al. (1985, AJ) in calculating a rigidly rotating reference geoid (isopotential)
        and then using an input wind profile to calculate the dynamical heights that describe the departure of the isobaric surface
        from that reference geoid.

        this version involves the same ideas but follows the formulation of Lindal 1992 (or Galanti et al. 2023) in retaining the full (nonuniform) rotation
        frequency in the expressions for the radial and polar components of the gravity vector.
        this leads to a simple ODE that can be integrated directly to get the isobaric shape directly.
        the dynamical heights can then be derived by subtracting from this result a shape calculated assuming rigid rotation.

        the results from the two methods agree quite closely, but this method makes it simpler to incorporate the full (nonuniform) rotation frequency
        without converting to and from wind velocity. feature tracking ultimately probes the total angular velocity, and expressing the results in terms
        of velocity always involves assumptions about the planetary shape, which we prefer to be agnostic about until we have a shape model.

        the shape is calculated directly from the assumed polar radius, jn, mass, and rotation profile, making no reference to an interior model.

        the particular reference pressure (100 mbar for Saturn in Lindal et al. 1985; 1 bar for Uranus in Lindal et al. 1987) enters only implicitly 
        through the imposed polar radius.

        given gravity data (jn) and a rotation profile derived from feature tracking, we would vary the input polar radius until a good fit to 
        radio occultation radii is found.

        r_pol           :   assumed polar radius, cm
        jn              :   iterable containing the degree 2, 4, and 6 zonal gravity coefficients
        r_ref           :   reference radius used to normalize the given jn, cm
        gm              :   planet gravitational mass GM, cm^3 s^-2
        npts            :   [optional] number of zones in latitude, mostly evenly distributed but denser close to poles, default 1001
        omega           :   either a callable function (e.g., an interpolant) that gives the input *total* angular velocity as a function of planetocentric 
                            latitude, or a scalar, in which case we assume uniform rotation. in either case units rad s^-1
        verbose         :   [optional] print output
        solve_ivp_tol   :   [optional] absolute and relative tolerance to pass to solve_ivp; beware default tolerances
        '''

        self.mu = np.linspace(-1, 1, npts - 20) # take out 20 points to be distributed close to the poles
        self.mu = np.insert(self.mu, 1, np.linspace(self.mu[0], self.mu[1], 12)[1:-1]) # insert 10 evenly spaced points between the first 2 points
        self.mu = np.insert(self.mu, -1, np.linspace(self.mu[-2], self.mu[-1], 12)[1:-1]) # same between the last 2 points
        self.phi = np.arccos(self.mu) - np.pi / 2 # planetocentric latitude (from equator), rad
        self.lat = np.arccos(self.mu) * 180. / np.pi - 90 # planetocentric latitude, deg

        j2, j4, j6 = jn

        # it would be nice to define dr_dphi as a pure function (it would be very close to psi_() above), to take advantage of numba/jit,
        # but this won't be possible: if omega callable (i.e., a scipy.interpolate.interp1d instance), passing an object and hence need the python interpreter.
        if not callable(omega): # rigid rotation
            def dr_dphi(phi, r):
                return r * gphi_(r, np.sin(phi), j2, j4, j6, r_ref, gm, omega) / gr_(r, np.sin(phi), j2, j4, j6, r_ref, gm, omega)
            self.omega = omega * np.ones_like(self.phi)
        else: # fun rotation
            def dr_dphi(phi, r):
                return r * gphi_(r, np.sin(phi), j2, j4, j6, r_ref, gm, omega(phi)) / gr_(r, np.sin(phi), j2, j4, j6, r_ref, gm, omega(phi))
            self.omega = omega(self.phi)

        sol = solve_ivp(dr_dphi, np.array([self.phi[0], self.phi[-1]]), np.array([r_pol]), t_eval=self.phi, rtol=solve_ivp_tol, atol=solve_ivp_tol) # beware default tolerances
        assert sol.success, 'failed in solve_ivp'
        self.r = sol.y[0] # shape solution

        self.psi = psi_(self.r, self.mu, j2, j4, j6, r_ref, gm, self.omega) # difference between planetographic and planetocentric latitude, rad
        self.planetographic_latitude = self.lat + self.psi * 180 / np.pi # planetographic latitude, deg

class geoid_barotropic:
    def __init__(self, r_pol, jn, r_ref, gm, npts=1001, omega=None, verbose=False, solve_ivp_tol=1e-10):

        '''
        experiment: same as geoid.geoid, but callable omega is a function not of latitude but distance from rotation axis.
        affects best fitting polar radius for saturn by less than 0.1 km. increases equatorial radius by 0.2 km. increases rms error by 0.2 km.
        '''

        self.mu = np.linspace(-1, 1, npts - 20) # take out 20 points to be distributed close to the poles
        self.mu = np.insert(self.mu, 1, np.linspace(self.mu[0], self.mu[1], 12)[1:-1]) # insert 10 evenly spaced points between the first 2 points
        self.mu = np.insert(self.mu, -1, np.linspace(self.mu[-2], self.mu[-1], 12)[1:-1]) # same between the last 2 points
        self.phi = np.arccos(self.mu) - np.pi / 2 # planetocentric latitude (from equator), rad
        self.lat = np.arccos(self.mu) * 180. / np.pi - 90 # planetocentric latitude, deg

        j2, j4, j6 = jn

        if not callable(omega): # rigid rotation
            def dr_dphi(phi, r):
                return r * gphi_(r, np.sin(phi), j2, j4, j6, r_ref, gm, omega) / gr_(r, np.sin(phi), j2, j4, j6, r_ref, gm, omega)
            self.omega = omega * np.ones_like(self.phi)
        else: # fun rotation
            def dr_dphi(phi, r):
                this_omega = omega( r * np.cos(phi) )
                return r * gphi_(r, np.sin(phi), j2, j4, j6, r_ref, gm, this_omega) / gr_(r, np.sin(phi), j2, j4, j6, r_ref, gm, this_omega)
            self.omega = omega(self.phi)

        sol = solve_ivp(dr_dphi, np.array([self.phi[0], self.phi[-1]]), np.array([r_pol]), t_eval=self.phi, rtol=solve_ivp_tol, atol=solve_ivp_tol) # beware default tolerances
        assert sol.success, 'failed in solve_ivp'
        self.r = sol.y[0]

        self.psi = psi_(self.r, self.mu, j2, j4, j6, r_ref, gm, self.omega) # difference between planetographic and planetocentric latitude, rad
        self.planetographic_latitude = self.lat + self.psi * 180 / np.pi # planetographic latitude, deg

class geoid_split:
    def __init__(self, r_pol, jn, r_ref, gm, omega_rigid, npts=1001, omega=None, verbose=False):

        '''
        similar to the above but first calculates a rigidly rotating reference geoid rotating at specified omega_rigid,
        then separately addresses the winds by calculating a dynamical height as in Lindal et al. (1985 AJ) Equation 17.
        
        the wind speed is derived from the specified angular frequency omega, by subtracting rigid rotation at omega_rigid.
        since this step requires knowledge of the shape (distance from the rotation axis), it is done iteratively while
        solving for the reference geoid.

        r_pol       :   assumed polar radius, cm
        jn          :   iterable containing the degree 2, 4, and 6 zonal gravity coefficients
        r_ref       :   reference radius used to normalize the given jn, cm
        gm          :   planet gravitational mass GM, cm^3 s^-2
        omega_rigid :   rotation frequency assumed for the rigidly rotating reference geoid, rad s^-1
        npts        :   [optional] number of zones in latitude, mostly evenly distributed but denser close to poles, default 1001
        omega       :   [optional] a callable function (e.g., an interpolant) that gives the input *total* angular velocity as a function of planetocentric 
                        latitude, rad s^-1. if not set, final shape will be that of the reference geoid rigidly rotating at omega_rigid.
        verbose     :   [optional] print output
        '''

        self.mu = np.linspace(-1, 1, npts - 20) # take out 20 points to be distributed close to the poles
        self.mu = np.insert(self.mu, 1, np.linspace(self.mu[0], self.mu[1], 12)[1:-1]) # insert 10 evenly spaced points between the first 2 points
        self.mu = np.insert(self.mu, -1, np.linspace(self.mu[-2], self.mu[-1], 12)[1:-1]) # same between the last 2 points
        self.phi = np.arccos(self.mu) - np.pi / 2 # planetocentric latitude (from equator), rad
        self.lat = np.arccos(self.mu) * 180. / np.pi - 90 # planetocentric latitude, deg

        j2, j4, j6 = jn

        u_ref = u_(r_pol, 1., j2, j4, j6, r_ref, gm, omega_rigid)
        self.r_geoid = r_pol * np.ones_like(self.mu) # initial guess for radii of the reference geoid
        for i in range(20): # iterate until reference geoid has correct potential everywhere
            u = u_(self.r_geoid, self.mu, j2, j4, j6, r_ref, gm, omega_rigid)
            error = u - u_ref
            self.r_geoid += error / gr_(self.r_geoid, self.mu, j2, j4, j6, r_ref, gm, omega_rigid)
            if verbose: 
                print(f'iteration {i}: r_pol={1e-5*self.r_geoid[0]:.2f} r_eq={1e-5*self.r_geoid[len(self.lat) // 2]:.2f} rmserr={np.mean(error ** 2) ** 0.5 / -u_ref:.2e}')
            if np.all(np.abs(error / u_ref) < 1e-10): break
        else:
            raise RuntimeError('reference geoid failed to converge')

        if isinstance(omega, type(None)): # rigid rotation
            self.r = self.r_geoid
            self.h_mean = 0
            self.omega = omega_rigid * np.ones_like(self.phi)
        else: # winds; solve for dynamical heights
            self.u_wind = (omega(self.phi) - omega_rigid) * self.r_geoid * np.cos(self.phi) # cm s^-1

            self.h_integrand = self.r_geoid * self.u_wind * (2. * omega_rigid + self.u_wind / self.r_geoid / np.cos(self.phi))
            self.h_integrand *= np.sin(self.phi + psi_(self.r_geoid, self.mu, j2, j4, j6, r_ref, gm, omega_rigid)) 
            self.h_integrand /= np.cos(psi_(self.r_geoid, self.mu, j2, j4, j6, r_ref, gm, omega_rigid)) 
            # first approximation to gavg: just evaluate on reference geoid
            gavg = -np.sqrt(gr_(self.r_geoid, self.mu, j2, j4, j6, r_ref, gm, omega_rigid) ** 2 + gphi_(self.r_geoid, self.mu, j2, j4, j6, r_ref, gm, omega_rigid) ** 2)
            integral = cumulative_trapezoid(self.h_integrand, x=self.phi, initial=0.)
            h_prev = -1. * np.ones_like(self.phi)
            for i in range(10): # ~1 iteration needed because h modifies the mean magnitude of gravity vector on field line connecting isobar to reference geoid
                self.h = integral / gavg
                if verbose: print(f'gavg iteration {i}: h_eq={1e-5 * self.h[len(self.lat) // 2]:.2f}')
                if np.all(np.abs(self.h - h_prev) * 1e-5 < 0.001): break
                h_prev = self.h
                # here a crude arithmetic average of the isobar and reference geoid gravities; switching to either end member changes things by < 1 km, so this should be good enough
                gavg =  -0.5 * np.sqrt(gr_(self.r_geoid, self.mu, j2, j4, j6, r_ref, gm, omega_rigid) ** 2 \
                    + gphi_(self.r_geoid, self.mu, j2, j4, j6, r_ref, gm, omega_rigid) ** 2)
                gavg += -0.5 * np.sqrt(gr_(self.r_geoid + self.h, self.mu, j2, j4, j6, r_ref, gm, omega_rigid) ** 2 \
                    + gphi_(self.r_geoid + self.h, self.mu, j2, j4, j6, r_ref, gm, omega_rigid) ** 2)
            else:
                raise RuntimeError('dynamical height failed to converge')

            self.r = self.r_geoid + self.h
            self.h_mean = -1. / 180 * trapezoid(self.h, x=self.lat)

            self.omega = omega(self.phi)

        self.psi = psi_(self.r, self.mu, j2, j4, j6, r_ref, gm, self.omega) # difference between planetographic and planetocentric latitude, rad
        self.planetographic_latitude = self.lat + self.psi * 180 / np.pi # planetographic latitude, deg

if __name__ == '__main__':

    print('running a minimal a uranus model.')
    import wind_profiles
    omega = wind_profiles.uranus_omega_tot_interpolant()

    # constants for Uranus; French et al. (2024 Icarus) Table 17, Fit 15 (Adopted solution)
    r_ref = 25559e5
    jn = 3509.291e-6, -35.522e-6, 0.
    gm = 5793950.3e15

    # fix the polar radius and solve for the shape
    r_pol = 25000e5 # this likely doesn't lead to a perfect fit to occultations

    print(f"       {'r_pol':>10} {'r_eq':>10}")
    # preferred (slower) method: retain full rotation
    g = geoid(r_pol, jn, r_ref, gm, omega=omega) # note, no presumed rigid rotation period enters the calculation
    g0 = geoid(r_pol, jn, r_ref, gm, omega=np.pi*2/(17.25*3600)) # only specify "bulk" rotation if want to construct a rigidly rotating reference geoid
    print(f' full: {1e-5*r_pol:10.1f} {1e-5*g.r[len(g.mu) // 2]:10.1f}')

    # alternate method: shape gets contributions from rigid and differential rotation parts; calculate reference geoid and dynamical height separately
    g = geoid_split(r_pol, jn, r_ref, gm, omega_rigid=np.pi*2/(17.25*3600), omega=omega)
    print(f'split: {1e-5*r_pol:10.1f} {1e-5*g.r[len(g.mu) // 2]:10.1f}')
