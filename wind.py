from default_vals import default_wind_vals
from misc import check_units
import numpy as np
from astropy import constants as const, units as un
import matplotlib.pyplot as plt

class Wind:
    def __init__(self, **kwargs):
        """
        Class for making spherical wind 
        :param **kwargs: mass loss rate (Mdot), wind speed (vwind), linear extent
        of the grid (linear_extent), number of pixels in each grid dimension (dim)
        :type **kwargs: TYPE
        :return: Wind class instance
        """
        for k in default_wind_vals:
            if k in kwargs:
                self.__dict__.update({k:check_units({k:kwargs[k]}, default_wind_vals)[k]})
            else:
                self.__dict__.update({k:default_wind_vals[k]})
                
        if self.dim%2 == 0:
            print(f"Dimension needs to be odd, changing to {self.dim+1}")
            self.dim += 1
        self.pix_res = self.linear_extent/self.dim 
        return
    
    
    def make_grids(self):
        """
        Makes grids for distances from center, particle scattering area, and
        number of scattering particles
        """
        if "grid_distances" not in self.__dict__:
            center = int(self.dim/2)
            x_grid = np.array([np.abs(np.linspace(0, self.dim-1, self.dim) - center)]*self.dim)            
            y_grid = x_grid.transpose()
            self.grid_distances = ((x_grid/self.dim)**2 + (y_grid/self.dim)**2)**0.5 * self.linear_extent
            
        # not particles per se, but essentially the fraction of the pixel area that is covered by particle area?
        self.grid_scattering_factor = (const.sigma_T * self.Mdot/(8 * self.grid_distances * self.vwind * const.u.cgs)).to("")
        self.grid_scattering_area = self.grid_scattering_factor * self.pix_res**2
        self.grid_particles = self.grid_scattering_area/const.sigma_T
        return
    
    
    def plot(self, unit:str = 'particles', logscale:bool = True):
        """
        Plots the wind
        :param unit: either 'particles', 'area', or 'fractional area', defaults to 'particles'
        :type unit: str, optional
        :param logscale: Defines whether to plot log 10 scale, defaults to True
        :type logscale: bool, optional
        :return: figure and axis instances
        :rtype: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        unit = unit.lower()
        assert(unit in ['particles', 'area', 'fractional area'])
        
        if unit == 'particles':
            dat = self.grid_particles.value
            cbar_label = "Number of electrons"
        if unit == 'area':
            dat = self.grid_scattering_area.to('cm**2').value
            cbar_label = r"Scattering area [cm$^2$]"
        elif unit == 'fractional area':
            dat = self.grid_scattering_factor.value
            cbar_label = r"Scattering fraction"
            
        if logscale:
            dat = np.log10(dat)
            cbar_label = r"$\log_{10}$("+cbar_label+")"
            
        d = self.linear_extent.to('AU').value
        axis_label = 'Distance [AU]'            

        extents = [-d/2, d/2,-d/2, d/2]
        fig, ax = plt.subplots(1,1)
        im = ax.imshow(dat, extent=extents)

        ax.set_xlabel(axis_label)
        ax.set_ylabel(axis_label)
        cbar = plt.colorbar(im)
        cbar.set_label(cbar_label)
        return fig, ax
