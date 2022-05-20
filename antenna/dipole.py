import functools


import numpy as np


from antenna import Antenna


class Dipole(Antenna):
    def __init__(self, wavelength, current, height, z0=0):
        self.wavelength = wavelength
        self.beta = 2 * np.pi / self.wavelength

        self.height = height
        self.length = self.height / 2
        self.z0 = z0
        self._resolution = 1001
        self.z = (
            np.linspace(-self.length, self.length, self._resolution) + self.z0)
        self.dz = self.z[1] - self.z[0]

        self.current_function = current
        self.current = self.current_function(self.z - self.z0)

        self._angular_resolution = 1000
        self.theta = np.linspace(0, np.pi, 2 * self._angular_resolution + 1)
        self.d_theta = self.theta[1]
        self.phi = np.linspace(0, 2 * np.pi, 4 * self._angular_resolution + 1)
        self.d_phi = self.phi[1]

    @functools.cached_property
    def N(self):
        return (
            -np.sum(
                self.current
                * np.exp(
                    1j * self.beta * self.z[np.newaxis, :]
                    * np.cos(self.theta)[:, np.newaxis]), axis=1)
            * self.dz * np.sin(self.theta))

    @functools.cached_property
    def Fa(self) -> np.ndarray:
        """The caracteristic equation of the antenna (Fa)."""
        return (np.broadcast_to(
            np.abs(self.N)[..., np.newaxis],
            self.theta.shape + self.phi.shape) / self.N_max)

    @property
    def K(self):
        return 15 * np.pi / self.wavelength**2 * self.N**2

    @ property
    def radiation_resistance(self):
        return 2 * self.transmitted_power / np.max(self.current)**2
