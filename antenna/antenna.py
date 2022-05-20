import functools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class Antenna(object):
    _angular_resolution = 100

    def __init__(self):
        """TODO: summary
        """
        self.theta = np.linspace(0, np.pi, 2 * self._angular_resolution + 1)
        self.d_theta = self.theta[1]
        self.phi = np.linspace(0, 2 * np.pi, 4 * self._angular_resolution + 1)
        self.d_phi = self.phi[1]
        self.Theta, self.Phi = np.meshgrid(self.theta, self.phi, indexing='ij')

    def __add__(self, other):
        return None

    @functools.lru_cache()
    def solid_angle(self, wavelength) -> float:
        """The solid angle (Omega_a) of the cone through which all the
        emitted power is concentrated at a constant value K_{max}."""
        return np.sum(
            self.Fa(wavelength)**2 * np.sin(self.theta)[:, np.newaxis],
            axis=(0, 1)) * self.d_theta * self.d_phi

    @functools.lru_cache()
    def directivity(self, wavelength) -> float:
        """The directivity (D) of the antenna compared to an isotropic
        antenna."""
        return 4 * np.pi / self.solid_angle(wavelength)

    def N_cart(self, *args, **kwargs):
        raise NotImplementedError()

    def N_spherical(self, *args, **kwargs):
        raise NotImplementedError()

    @functools.lru_cache()
    def Fa(self, wavelength) -> np.ndarray:
        """The caracteristic equation of the antenna (Fa)."""
        N_norm = np.linalg.norm(self.N_spherical(wavelength)[..., 1:], axis=-1)
        return (N_norm / np.max(N_norm))

    @functools.lru_cache()
    def K(self, wavelength):
        return (15 * np.pi / wavelength**2
                * np.linalg.norm(self.N_spherical(wavelength)[..., 1:],
                                 axis=-1))

    @functools.lru_cache()
    def radiation_resistance(self, wavelength):
        return 2 * self.transmitted_power / np.max(self.current)**2

    @functools.lru_cache()
    def transmitted_power(self, wavelength) -> float:
        """The total power transmitted by the antenna (Pt)."""
        return (2 * np.pi * np.sum(np.abs(self.K) * np.sin(self.theta), axis=0)
                * self.d_theta)

    def _plot_cross_sections(self, field, axs):
        # plan xy
        axs[0].title.set_text(r'Plan xy ($\phi$, $\theta=\frac{\pi}{2}$)')
        axs[0].plot(self.phi, field[self._angular_resolution, :])
        axs[0].set_theta_offset(-np.pi / 2.0)

        # plan xz
        axs[1].title.set_text(r'Plan xz ($\phi=0$, $\theta$)')
        axs[1].plot(
            [*self.theta, *(self.theta) + np.pi],
            [*field[:, 0],
             *field[::-1, 2 * self._angular_resolution]])
        axs[1].set_theta_direction(-1)
        axs[1].set_theta_offset(np.pi / 2)

        # plan yz
        axs[2].title.set_text(r'Plan yz ($\phi=\frac{\pi}{2}$, $\theta$)')
        axs[2].plot(
            [*self.theta, *(self.theta + np.pi)],
            [*field[:, self._angular_resolution],
             *field[::-1, 3 * self._angular_resolution]])
        axs[2].set_theta_direction(-1)
        axs[2].set_theta_offset(np.pi / 2)

        return axs

    def _plot_3d(self, ax, field, dB=False, elevation=None,
                 azimutal=None):
        ax.title.set_text('Vue 3D')

        dB_min = -30

        norm = mpl.colors.Normalize(vmin=0, vmax=np.max(field))
        if dB:
            norm = mpl.colors.Normalize(vmin=dB_min, vmax=np.max(field))

        cmap = mpl.cm.ScalarMappable(norm, mpl.cm.Spectral)
        fcolors = cmap.to_rgba(field)

        if dB:
            field_to_plot = field - dB_min
            field_to_plot /= np.max(field_to_plot)
            field_to_plot[field_to_plot < 0] = np.nan
        else:
            # if np.min(field) < 0:
            #     print('Negative values cropped')
            field_to_plot = field

        Theta, Phi = np.meshgrid(self.theta, self.phi, indexing='ij')

        X, Y, Z = (
            field_to_plot * np.sin(Theta) * np.cos(Phi),
            field_to_plot * np.sin(Theta) * np.sin(Phi),
            field_to_plot * np.cos(Theta)
        )

        ax.plot_surface(X, Y, Z, facecolors=fcolors, shade=False, rcount=100,
                        ccount=100, antialiased=True)

        ax.set_axis_off()

        ax.quiver(-1, -1, -1, 2, 0, 0, linewidth=2, color='red')
        ax.quiver(-1, -1, -1, 0, 2, 0, linewidth=2, color='green')
        ax.quiver(-1, -1, -1, 0, 0, 2, linewidth=2, color='blue')

        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)

        ax.view_init(elevation, azimutal)

        if dB:
            plt.colorbar(
                cmap, ax=ax,
                # orientation='horizontal',
                label='Directivité [dBi]')

    def plot_Fa(self, wavelength,
                show: bool = True, save: bool = False, name: str = None,
                **kwargs) -> plt.figure:
        """Plot of the caracteristic function of the antenna in the xy,
        xz and yz plane and in 3 dimension.

        Args:
            show (bool, optional): Whether to show the output plot.
                Defaults to True.
            save (bool, optional): Whether to save the output plot.
                Defaults to False.
            name (str, optional): Where to save the output plot.
                Defaults to None.

        Returns:
            plt.figure: The plot generated by matplotlib.
        """
        fig = plt.figure(figsize=(12, 9))
        axs = (
            plt.subplot(221, projection='polar', aspect='equal'),
            plt.subplot(222, projection='polar', aspect='equal'),
            plt.subplot(223, projection='polar', aspect='equal'),
            plt.subplot(224, projection='3d')
        )

        self._plot_cross_sections(self.Fa(wavelength), axs[:-1])
        axs[0].set_rmin(0)
        axs[0].set_rmax(1.1)
        axs[1].set_rmin(0)
        axs[1].set_rmax(1.1)
        axs[2].set_rmin(0)
        axs[2].set_rmax(1.1)

        # 3D
        self._plot_3d(axs[3], self.Fa(wavelength), **kwargs)

        fig.tight_layout()
        if save:
            if name is None:
                print('No name specified saving as "plot"')
                name = 'plot'
            fig.savefig(name, dpi=600)
        if show:
            plt.show()
        return fig

    def plot_D(self, wavelength,
               show: bool = True, save: bool = False, name: str = None,
               **kwargs) -> plt.figure:
        """Plot of the directivity pattern of the antenna in the xy, xz
        and yz plane and in 3 dimension.

        Args:
            show (bool, optional): Whether to show the output plot.
                Defaults to True.
            save (bool, optional): Whether to save the output plot.
                Defaults to False.
            name (str, optional): Where to save the output plot.
                Defaults to None.

        Returns:
            plt.figure: The plot generated by matplotlib.
        """
        fig = plt.figure(figsize=(12, 9))
        axs = (
            plt.subplot(221, projection='polar', aspect='equal'),
            plt.subplot(222, projection='3d'),
            plt.subplot(223, projection='polar', aspect='equal'),
            plt.subplot(224, projection='polar', aspect='equal'),
        )
        D = 10 * np.log10(self.directivity(wavelength))
        Directivity = 20 * np.log10(self.Fa(wavelength)) + D

        self._plot_cross_sections(Directivity, [axs[0], axs[3], axs[2]])

        rmin = D - 30
        rmax = D + 3
        axs[0].arrow(0, rmin, 0, rmax - rmin,
                     linewidth=2, color='red')
        axs[0].arrow(np.pi / 2, rmin, 0, rmax - rmin,
                     linewidth=2, color='green')
        axs[2].arrow(0, rmin, 0, rmax - rmin,
                     linewidth=2, color='blue')
        axs[2].arrow(np.pi / 2, rmin, 0, rmax - rmin,
                     linewidth=2, color='green')
        axs[3].arrow(0, rmin, 0, rmax - rmin,
                     linewidth=2, color='blue')
        axs[3].arrow(np.pi / 2, rmin, 0, rmax - rmin,
                     linewidth=2, color='red')

        for ax in [axs[0], axs[3], axs[2]]:
            ax.set_rmin(rmin)
            ax.set_rmax(rmax)

        # 3D
        self._plot_3d(axs[1], Directivity, dB=True, ** kwargs)

        fig.tight_layout()
        if save:
            if name is None:
                print('No name specified saving as "plot"')
                name = 'plot'
            fig.savefig(name, dpi=600)
        if show:
            plt.show()
        return fig

    # def plot_Fa_dB(self, min_dB=-20, show: bool = True,
    #                save: bool = False,
    #                name: str = None,
    #                **kwargs) -> plt.figure:
    #     """Plot of the caracteristic function of the antenna in the xy,
    #     xz and yz plane and in 3 dimension.

    #     Args:
    #         show (bool, optional): Whether to show the output plot.
    #             Defaults to True.
    #         save (bool, optional): Whether to save the output plot.
    #             Defaults to False.
    #         name (str, optional): Where to save the output plot.
    #             Defaults to None.

    #     Returns:
    #         plt.figure: The plot generated by matplotlib.
    #     """
    #     Fa = (20 * np.log10(self.Fa))
    #     # print(np.max(Fa))
    #     fig = plt.figure(figsize=(12, 9))

    #     # plan xy
    #     ax = plt.subplot(221, projection='polar', aspect='equal')
    #     ax.title.set_text(r'Plan xy ($\phi$, $\theta=\frac{\pi}{2}$)')
    #     ax.plot(self.phi, Fa[self._angular_resolution, :])
    #     ax.set_theta_offset(-np.pi / 2.0)
    #     # ax.set_rscale('log')
    #     ax.set_rmax(0)
    #     ax.set_rmin(min_dB)

    #     # plan xz
    #     ax = plt.subplot(222, projection='polar', aspect='equal')
    #     ax.title.set_text(r'Plan xz ($\phi=0$, $\theta$)')
    #     ax.plot(
    #         [*self.theta, *(self.theta) + np.pi],
    #         [*Fa[:, 0], *Fa[::-1, 2 * self._angular_resolution]])
    #     ax.set_theta_offset(np.pi)
    #     ax.set_thetamin(0)
    #     # ax.set_thetamax(180)
    #     ax.set_rmax(0)
    #     ax.set_rmin(min_dB)

    #     # plan yz
    #     ax = plt.subplot(223, projection='polar', aspect='equal')
    #     ax.title.set_text(r'Plan yz ($\phi=\frac{\pi}{2}$, $\theta$)')
    #     ax.plot(
    #         [*self.theta, *(self.theta + np.pi)],
    #         [*Fa[:, self._angular_resolution],
    #          *Fa[::-1, 3 * self._angular_resolution]])
    #     ax.set_theta_direction(-1)
    #     ax.set_theta_offset(np.pi / 2.0)
    #     ax.set_thetamin(0)
    #     # ax.set_thetamax(180)
    #     ax.set_rmax(0)
    #     ax.set_rmin(min_dB)

    #     # 3D
    #     ax = plt.subplot(224, projection='3d')
    #     ax.title.set_text('Vue 3D')
    #     self._plot_3d(ax, **kwargs)

    #     fig.tight_layout()
    #     if save:
    #         if name is None:
    #             print('No name specified saving as "plot"')
    #             name = 'plot'
    #         fig.savefig(name, dpi=600)
    #     if show:
    #         plt.show()
    #     return fig
