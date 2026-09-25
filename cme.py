import numpy as np
from default_vals import default_cme_vals
from misc import check_units
from scipy.ndimage import gaussian_filter
from astropy import units as un, constants as const
import matplotlib.pyplot as plt

class CME:
    def __init__(self, **kwargs):
        """
        Class for modelling CME
        :param **kwargs: number of pixels in each dimension of each grid (dim)

        """
        for k in default_cme_vals:
            if k in kwargs:
                self.__dict__.update({k:check_units({k:kwargs[k]}, default_cme_vals)[k]})
            else:
                self.__dict__.update({k:default_cme_vals[k]})
                
        if self.dim%2 == 0:
            print(f"Dimension needs to be odd, changing to {self.dim+1}")
            self.dim += 1
        
        return
    
        
    def build_cme_normalized_grid(self, smear=False, smear_sig=3, **kwargs):
        for k in kwargs:
            self.__dict__.update({k:check_units(self.__dict__, default_cme_vals)[k]})
            
        assert(self.dR_factor < 1)

        Rc = int(self.dim/2) + 1 # dimension of CME subset grid
        
        dR = Rc * self.dR_factor # fraction of the quadrant that the CME radii encompass
        grid = np.zeros((Rc,Rc)) # start with a quadrant of the grid to enforce symmetry
        
        for j in range(int(dR/2)):
            Rmin = dR +j + dR*self.horizontal_height_factor
            Rmax = dR*2 - j + dR*self.horizontal_height_factor
            
            phi_hw = self.phi_HW
            a = np.pi/2 /phi_hw
            
            phi = np.linspace(0, np.pi/4, Rc)
            rmin = Rmin * np.cos(a * phi)
            rmax = Rmax * np.cos(a * phi)
            
            xmin, ymin = ((rmin*np.cos(phi)).astype(int), (self.vertical_height_factor*rmin*np.sin(phi)).astype(int))
            xmax, ymax = ((rmax*np.cos(phi)).astype(int), (self.vertical_height_factor*rmax*np.sin(phi)).astype(int))
                     
            xmin = xmin[::-1]
            ymin = ymin[::-1] 
            xmax = xmax[::-1]
            ymax = ymax[::-1] 
            
            grid_subset = np.zeros(grid.shape)
            for i in range(int(Rmax)):
                idxmax = np.where(xmax == i)[0]
                idxmin = np.where(xmin == i)[0]
            
                if len(idxmax) != 0:
                    idxmaxsave = idxmax[0]
                    yf_max = ymax[idxmax[0]]
                    if i > xmin.max():
                        yf_min = 0
                    if len(idxmin) != 0:
                        yf_min = ymin[idxmin[0]]
                        idxminsave = idxmin[0]
                        
                    if len(idxmin) == 0 and i < xmin.max():
                        yf_min = int((ymin[idxminsave] + ymin[idxminsave + 1])/2)
                
                if len(idxmax) == 0:
                    if len(idxmin) != 0:
                        yf_max = int((ymax[idxmaxsave] + ymax[idxmaxsave + 1])/2)
                        yf_min = ymin[idxmin[0]]
                        idxminsave = idxmin            
                        
                    if len(idxmin) == 0 and i < xmin.max():
                        yf_max = int((ymax[idxmaxsave] + ymax[idxmaxsave + 1])/2)
                        yf_min = int((ymin[idxminsave] + ymin[idxminsave + 1])/2)
                        
                    if len(idxmin) == 0 and i >= xmin.max():
                        yf_max = int((ymax[idxmaxsave] + ymax[idxmaxsave + 1])/2)
                        yf_min = 0
                try:
                    dy = np.abs(yf_max - yf_min)
                    grid_subset[i, yf_min:yf_max] = dy
                    
                    if yf_max == yf_min:
                        grid_subset[i, yf_max] = 1
                except:
                    pass
                    
            # grid += np.transpose(grid_subset) * ((self.depth_factor*(Rmin-Rmax)) * xmax )
            grid += np.transpose(grid_subset) * np.abs((Rmin-Rmax)  * xmax)** 0.5 * j 
                    
        grid_full = np.zeros((self.dim, self.dim))
        grid_full[Rc-1:,Rc-1:] = grid
        grid_full[0:Rc,Rc-1:] = np.flip(grid, axis = 0)
        if smear:
            grid_full = gaussian_filter(grid_full, sigma=smear_sig)
        self.grid_norm = grid_full/np.nansum(grid_full)
        return
    
        
    def make_grids(self, **kwargs):
        for k in kwargs:
            self.__dict__.update({k:kwargs[k]})
            
        if "grid_norm" not in self.__dict__:
            self.build_cme_normalized_grid()
            
        center = (self.grid_norm.shape[0]/2, self.grid_norm.shape[0]/2)

        x_grid = np.zeros(self.grid_norm.shape)
        center = [int(self.dim/2+1), int(self.dim/2 +1)]
        for i in range(x_grid.shape[0]):
            x_grid[i][:] = np.abs(np.linspace(0, x_grid.shape[0]-1, x_grid.shape[0]) - center[0])            
        y_grid = x_grid.transpose()
        
        self.grid_distances = ((x_grid/self.dim)**2 + (y_grid/self.dim)**2)**0.5 * self.linear_extent
        self.grid_mass = self.grid_norm * self.mass
        self.grid_number = (self.grid_mass/const.u.cgs).to('')
        return 
    
    
    def plot(self, unit:str = "particle", logscale:bool = True):
        if unit.lower() == 'mass':
            dat = self.grid_mass.to('g').value
            cbar_label = r'Mass/g'
        elif unit.lower() == 'particle':
            dat = self.grid_number
            cbar_label = r'Number of electrons'
            

        d = self.linear_extent.to('AU').value
        axis_label = 'Distance [AU]'            
                    
        if logscale:
            dat = np.log10(dat.value)
            cbar_label = f"log10({cbar_label})"
            
        extents = [-d/2, d/2,-d/2, d/2]
        fig, ax = plt.subplots(1,1)
        im = ax.imshow(dat, extent=extents)

        ax.set_xlabel(axis_label)
        ax.set_ylabel(axis_label)
        cbar = plt.colorbar(im)
        cbar.set_label(cbar_label)
        return