"""TODO: summary

TODO: wavelength not class property
"""
import functools


import numpy as np


from .array import Array


def Tchebychev_coeff(m, n):
    if n > m:
        return 0
    coeffs = np.zeros(m + 1)
    coeffs[-1] = 1
    return np.polynomial.chebyshev.cheb2poly(coeffs)[n]


def Tchebychev_poly(m, x):
    if m == 0:
        return 1
    if m == 1:
        return x
    return 2 * x * Tchebychev_poly(m - 1, x) - Tchebychev_poly(m - 2, x)


class Dolph_Tchebychev_array(Array):
    # def __init__(self, element, orientation, N, R0, psi, wavelength):
    #     x0 = np.cosh(np.arccosh(R0) / (N - 1))

    #     gamma = np.pi / wavelength * d * cos(psi)
    #     x_min = x0 * np.cos(np.pi / wavelength * d)
    #     if N % 2 == 0:
    #         M = N // 2

    #         Coeff_matrix = np.array([
    #             [
    #                 Tchebychev_coeff(2 * m + 1, 2 * n + 1) / (x0**(2 * n + 1))
    #                 for m in range(M)]
    #             for n in range(M)])

    #         Y_matrix = np.array(
    #             [Tchebychev_coeff(N - 1, 2 * n + 1) for n in range(M)])

    #         currents = np.linalg.inv(Coeff_matrix) @ Y_matrix
    #         currents = np.array([*currents[::-1], *currents])
    #     else:
    #         M = (N - 1) // 2
    #         Coeff_matrix = np.array([
    #             [
    #                 Tchebychev_coeff(2 * m, 2 * n) / (x0**(2 * n))
    #                 for m in range(M + 1)]
    #             for n in range(M + 1)])
    #         print(Coeff_matrix)
    #         Y_matrix = np.array(
    #             [Tchebychev_coeff(N - 1, 2 * n) for n in range(M + 1)])
    #         print(Y_matrix)

    #         currents = np.linalg.inv(Coeff_matrix) @ Y_matrix
    #         currents = np.array([*currents[::-1], *currents[1:]])

    #     print(currents)

    #     d =

    #     # super().__init__(self, element, rs, currents, wavelength)

    @property
    def f_res(self):
        N = 7
        R0 = 8
        x0 = np.cosh(np.arccosh(R0) / (N - 1))
        d = 0.4
        psi_max = np.pi / 3
        return np.abs(Tchebychev_poly(
            N - 1,
            x0 * np.cos(np.pi * d * np.cos(self.Theta - np.pi / 2 - psi_max))))
