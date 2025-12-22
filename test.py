import unittest
import geoid
import wind_profiles
import numpy as np

r = 25e8
mu = 0.3
j2, j4, j6 = 0.1, 0, 0
r_ref = 25e8
gm_tot = 5793950.3e15
omega = 1e-4
relative_tolerance = 1e-6 # relative tolerance to allow for intermediate results; amounts to a bit smaller than 0.1 km on a 25,000 km radius
radius_tolerance_km = 0.1

class TestGeoidMethods(unittest.TestCase):

    def test_u(self):
        # self.assertEqual(geoid.u_(r, mu, j2, j4, j6, r_ref, gm_tot, omega), -2430609294380.0)
        u = geoid.u_(r, mu, j2, j4, j6, r_ref, gm_tot, omega)
        u_truth = -2430609294380.0
        self.assertTrue(abs( (u - u_truth) / u) < relative_tolerance)

    def test_gr(self):
        gr = geoid.gr_(r, mu, j2, j4, j6, r_ref, gm_tot, omega)
        gr_truth = -1005.7920572560001
        self.assertTrue(abs( (gr - gr_truth) / gr) < relative_tolerance)

    def test_gphi(self):
        # self.assertEqual(geoid.gphi_(r, mu, j2, j4, j6, r_ref, gm_tot, omega), 86.7444430507603)
        gphi = geoid.gphi_(r, mu, j2, j4, j6, r_ref, gm_tot, omega)
        gphi_truth = 86.7444430507603
        self.assertTrue(abs( (gphi - gphi_truth) / gphi) < relative_tolerance)

    def test_psi(self):
        psi = geoid.psi_(r, mu, j2, j4, j6, r_ref, gm_tot, omega)
        psi_truth = -0.08603202172739485
        self.assertTrue(abs( (psi - psi_truth) / psi) < relative_tolerance)

    def test_geoid(self):
        omega = wind_profiles.uranus_omega_tot_interpolant()
        r_pol = 25e8
        g = geoid.geoid(r_pol, [j2, j4, j6], r_ref, gm_tot, omega=omega)
        # self.assertEqual(1e-5 * g.r[len(g.mu) // 2], 29534.997717741164)
        self.assertTrue(np.abs(1e-5 * g.r[len(g.mu) // 2] - 29534.99) < radius_tolerance_km)

    def test_geoid_split(self):
        omega = wind_profiles.uranus_omega_tot_interpolant()
        omega_rigid = np.pi * 2 / 16.58 / 3600
        r_pol = 25e8
        g = geoid.geoid_split(r_pol, [j2, j4, j6], r_ref, gm_tot, omega_rigid, omega=omega)
        self.assertTrue(np.abs(1e-5 * g.r[len(g.mu) // 2] - 29534.49) < radius_tolerance_km)
        self.assertTrue(np.abs(1e-5 * g.h[len(g.mu) // 2] - 77.36) < radius_tolerance_km)

if __name__ == '__main__':
    unittest.main()
