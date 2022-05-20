import functools


import numpy as np


from .antenna import Antenna


class Array(Antenna):
    def __init__(self, element, rs, currents, wavelength):
        """TODO: summary

        TODO: wavelength not class property

        Arguments:
            element -- _description_
            rs -- _description_
            currents -- _description_
            wavelength -- _description_
        """
        super().__init__(wavelength)
        self.element = element
        self.currents = currents
        self.rs = rs

    @property
    def Fe(self):
        return self.element.Fa

    @functools.cached_property
    def f_res(self):
        delta_r = -np.einsum(
            '...j, jkl->...kl',
            self.rs,
            np.array(
                [np.sin(self.Theta) * np.cos(self.Phi),
                 np.sin(self.Theta) * np.sin(self.Phi),
                 np.cos(self.Theta)]))
        return np.sum((
            self.currents[..., np.newaxis, np.newaxis]
            * np.exp(-1j * self.beta * delta_r)
        ).reshape(-1, *self.Theta.shape), axis=0)

    @functools.cached_property
    def Fa(self):
        fa = self.Fe * np.abs(self.f_res)
        return fa / np.max(fa)
