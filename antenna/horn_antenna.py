import functools


import numpy as np


from .antenna import Antenna


class Horn_antenna(Antenna):
    _lin_resolution = 31

    def __init__(self, A, B, R1, R2, wavelength):
        """TODO: summary

        TODO: wavelength not class property

        Arguments:
            A -- _description_
            B -- _description_
            R1 -- _description_
            R2 -- _description_
            wavelength -- _description_
        """
        super().__init__(wavelength)

        self.A = A
        self.B = B
        self.R1 = R1
        self.R2 = R2

        self.x = np.linspace(-self.A / 2, self.A / 2, self._lin_resolution)
        self.y = np.linspace(-self.B / 2, self.B / 2, self._lin_resolution)

        self.S = np.moveaxis(np.meshgrid(self.x, self.y, indexing='ij'), 0, -1)
        self.d_S = np.prod(self.S[1, 1] - self.S[0, 0])

        self.E = np.moveaxis((
            np.zeros(self.S.shape[:-1]),
            np.cos(np.pi * self.S[:, :, 0] / self.A) * np.exp(
                1j * (self.beta / 2)
                * (self.S[:, :, 0]**2 / self.R1
                   + self.S[:, :, 1]**2 / self.R2)),
            np.zeros(self.S.shape[:-1])), 0, -1)

    @staticmethod
    def optimal_opening(f, D, a, b, is_D_dB: bool = True):
        def most_real_pos(array: np.ndarray):
            """_summary_

            Args:
                array (np.ndarray): _description_

            Returns:
                float: The closest value from the array to be real and
                    is positive
            """
            array = array[np.greater(array.real, 0)]

            min_im = np.inf
            most_real_pos_value = np.nan
            for value in array:
                if min_im > np.abs(value.imag):
                    min_im = np.abs(value.imag)
                    most_real_pos_value = np.abs(value)
            return most_real_pos_value
        c = 3e8
        wavelength = c / f
        if is_D_dB:
            D = 10**(D / 10)

        AB = (D * wavelength**2 / (0.51 * 4 * np.pi))

        A = most_real_pos(
            np.polynomial.Polynomial(
                [- 3 / 2 * AB**2, 3 / 2 * b * AB, 0, -a, 1]).roots())
        B = AB / A

        R1 = A**2 / (3 * wavelength)
        R2 = B**2 / (2 * wavelength)

        Rp = (((A - a) / A * R1)
              + ((B - b) / B * R2)) / 2

        lh = np.sqrt(R1**2 + (A / 2)**2)
        le = np.sqrt(R2**2 + (B / 2)**2)

        return Horn_antenna(A, B, R1, R2, wavelength)

    @property
    def N_cart(self):
        delta_r = np.einsum(
            'ijk, klm->ijlm',
            self.S,
            np.array(
                [np.sin(self.Theta) * np.cos(self.Phi),
                 np.sin(self.Theta) * np.sin(self.Phi), ]))
        #  np.cos(self.Theta)]))
        N_cart = (np.sum(
            self.E[..., np.newaxis, np.newaxis, :]
            * np.exp(-1j * self.beta * delta_r)[..., np.newaxis], axis=(0, 1))
            * self.d_S)
        N_cart *= (self.Theta <= np.pi / 2)[..., np.newaxis]
        return N_cart

    @property
    def N_spherical(self):
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
        return np.einsum('...ij, ...j->...i', cart2spherical, self.N_cart)
