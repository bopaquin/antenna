from .antenna import Antenna


class Isotropic_antenna(Antenna):
    @property
    def Fa(self):
        return 0 * self.Theta + 1
