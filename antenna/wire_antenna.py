import functools

import numpy as np

from .antenna import Antenna


class Wire_antenna(Antenna):
    _lin_resolution = 31

    def __init__(self, height, orientation):
        """TODO: summary

        Arguments:
            height -- _description_
            orientation -- _description_
        """
        super().__init__()

        self.height = height
        self.length = self.height / 2

        orientation = np.array(orientation)
        self.orientation = orientation / np.linalg.norm(orientation)

        self.r = (
            np.linspace(
                -self.length,
                self.length,
                self._lin_resolution)[:, np.newaxis]
            * self.orientation[np.newaxis, :])
        self.current = np.abs(np.sin(
            2 * np.pi * (self.length - np.linalg.norm(self.r, axis=-1))))
        self.d_r = self.r[1] - self.r[0]

    # @property
    def N_cart(self, wavelength):
        beta = 2 * np.pi / wavelength
        delta_r = np.einsum(
            'ij, jkl->ikl',
            self.r,
            np.array(
                [np.sin(self.Theta) * np.cos(self.Phi),
                 np.sin(self.Theta) * np.sin(self.Phi),
                 np.cos(self.Theta)]))
        return (np.sum(
            self.current[:, np.newaxis, np.newaxis]
            * np.exp(-1j * beta * delta_r), axis=0)[..., np.newaxis]
            * self.d_r)

    # @property
    def N_spherical(self, wavelength):
        cart2spherical = np.moveaxis(np.array([
            [
                np.sin(self.Theta) * np.cos(self.Phi),
                np.sin(self.Theta) * np.sin(self.Phi),
                np.cos(self.Theta)
            ],
            [
                np.cos(self.Theta) * np.cos(self.Phi),
                np.cos(self.Theta) * np.sin(self.Phi),
                -np.sin(self.Theta)
            ],
            [
                -np.sin(self.Phi),
                np.cos(self.Phi),
                0 * self.Phi
            ]
        ]), (0, 1), (-2, -1))
        return np.einsum(
            '...ij, ...j->...i',
            cart2spherical,
            self.N_cart(wavelength))
