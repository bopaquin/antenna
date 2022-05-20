import functools

import numpy as np

from .wire_antenna import Wire_antenna


class Elementary_antenna(Wire_antenna):
    def __init__(self, height, orientention, wavelength):
        """TODO: summary

        TODO: wavelength not class property

        Arguments:
            height -- _description_
            orientention -- _description_
            wavelength -- _description_
        """
        super().__init__(height, orientention, wavelength)
        self.current = 0 * self.current + 1
