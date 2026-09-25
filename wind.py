from default_vals import default_wind_vals
from misc import check_units
import numpy as np
from astropy import constants as const

class Wind:
    def __init__(self, **kwargs):
        for k in default_wind_vals:
            if k in kwargs:
                self.__dict__.update({k:kwargs[k]})
            else:
                self.__dict__.update({k:default_wind_vals[k]})
                
        return_dict = check_units(vwind=self.vwind, Mdot=self.Mdot, linear_extent=self.linear_extent)
        if self.dim%2 == 0:
            self.dim += 1
            
        self.Mdot = return_dict['Mdot']
        self.vwind = return_dict['vwind']
        self.linear_extent = return_dict["linear_extent"]
        return
    
    
    def make_grids(self):
        if "grid_distances" not in self.__dict__:
            x_grid = np.zeros((self.dim, self.dim))
            center = [int(self.dim/2+1), int(self.dim/2 +1)]
            for i in range(x_grid.shape[0]):
                x_grid[i][:] = np.abs(np.linspace(0, x_grid.shape[0]-1, x_grid.shape[0]) - center[0])            
            y_grid = x_grid.transpose()
            
            self.grid_distances = ((x_grid/self.dim)**2 + (y_grid/self.dim)**2)**0.5 * self.linear_extent
        # not particles per se, but essentially the fraction of the pixel area that is covered by particle area?
        self.grid_scattering_factor = (const.sigma_T * self.Mdot/(8 * self.grid_distances * self.vwind * const.u.cgs)).to("")
        self.grid_scattering_factor[center,center] = np.nan
        return
