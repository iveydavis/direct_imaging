from astropy import units as un, constants as const
from misc import check_units
from default_vals import default_star_vals, default_units
import numpy as np
from scipy import integrate
from cme import CME
from wind import Wind
import matplotlib.pyplot as plt


def spectral_radiance(wavelength, temp):
    """
    Calculates the stellar radiance (erg/s/cm^2/nm) from Planck's law at a given wavelength for a given temperature
    :param wavelength: wavelength to evaluate at; assumed to be in units of nm
    :type wavelength: float 
    :param temp: temperature to evaluate at; assumed to be in untis of K
    :type temp: float
    :return: spectral intensity in units erg/s/cm^2/nm
    :rtype: float

    """
    if type(temp) != un.Quantity:
        temp *= un.K
    if type(wavelength) != un.Quantity:
        wavelength *= un.nm
        
    spectralIntensityUnits = 'erg*s**-1*cm**-2*nm**-1'
    h = const.h
    c = const.c
    k = const.k_B
    numerator = 2 * h*c**2/wavelength**5
    denominator = np.exp((h*c/(wavelength*k*temp)).to('')) -1
    spectral_intensity = (numerator/denominator).to(spectralIntensityUnits)
    return spectral_intensity.value


def calculate_stellar_luminosity(wavelength_range, temp, radius):
    """
    Calculates the stellar luminosity over a wavelength range from Planck's law
    :param wavelength_range: range of wavelength to calculate luminsoity over; wavelengths assumed to have unit nm if not defined
    :type wavelength_range: list or numpy.array of floats (length 2)
    :param temp: temperature, assumed to have unit K if not defined
    :type temp: float
    :param radius: radius of the star, assumed to be units of solar radii if not defined
    :type radius: float
    :return: luminosity in erg/s
    :rtype: float

    """
    
    assert(len(wavelength_range) == 2)
    
    radiance = calculate_stellar_radiance(wavelength_range, temp)
    radius = check_units({'radius': radius}, default_units)['radius']
    lum = radiance * 4 * np.pi * un.sr * radius.to('cm')**2
    return lum


def calculate_stellar_radiance(wavelength_range, temp):
    """
    Evaluate spectral radiance over wavelength range
    :param wavelength_range: range of wavelength to calculate luminsoity over; wavelengths assumed to have unit nm if not defined
    :type wavelength_range: list or numpy.array of floats (length 2)
    :param temp: temperature, assumed to have unit K if not defined
    :type temp: float
    :return: Stellar radiance with unit erg/s/sr/cm^2
    :rtype: astropy.quantity.Quantity

    """
    assert(len(wavelength_range) == 2)
    
    # Check units of inputs
    wav_min = check_units({'wavelength':wavelength_range[0]}, default_units)['wavelength'].value
    wav_max = check_units({'wavelength':wavelength_range[1]}, default_units)['wavelength'].value
    temp = check_units({'temp':temp}, default_units)['temp'].value
    
    radiance, toss = integrate.quad(spectral_radiance, wav_min, wav_max, (temp))
    return radiance*un.erg/un.s/un.sr/un.cm**2


class Star:
    def __init__(self, **kwargs):
        """
        Class for determining observed stellar fluxes
        :param **kwargs: stellar radius (radius), stellar temperature (temp), 
        wavelength range (wavelength_range), distance of star (distance), the grid
        dimension (dim), the physical extent of the grid (linear_extent)
        :return: Star class instance

        """
        for k in default_star_vals:
            if k in kwargs:
                self.__dict__.update({k:check_units({k:kwargs[k]}, default_star_vals)[k]})
            else:
                self.__dict__.update({k:default_star_vals[k]})
                
        assert(len(self.wavelength_range) == 2), "wavelength_range should be list or numpy.ndarray of length 2"
        
        if self.dim % 2 == 0:
            print(f"Dimension needs to be odd, changing to {self.dim+1}")
            self.dim += 1
            
        self.luminosity = calculate_stellar_luminosity(self.wavelength_range, self.temp, self.radius)
        self.radiance = calculate_stellar_radiance(self.wavelength_range, self.temp)
        self.centre_wave = (self.wavelength_range[0] + self.wavelength_range[1])/2 
        self.photon_energy = (const.h*const.c/self.centre_wave).to('erg')
        return
    
    
    def make_stellar_grids(self, linear_extent=None, dim:int=None, wavelength_range=None):
        """
        Makes grids for physical distances centered on star, angular distances, energy flux, and photon flux
        :param linear_extent: Physical extent of the grid. If None, uses the class instance's value. defaults to None
        :type linear_extent: astropy.quantity.Quantity, optional
        :param dim: Number of pixels in each grid dimension. If None, uses the class instance's value, defaults to None
        :type dim: int, optional
        :param wavelength_range: wavelength range to evaluate stellar energy and photon fluxes. If None, uses the class instance's value, defaults to None
        :type wavelength_range: list or numpy.array of astropy.quantity.Quantity, optional
        """
        if linear_extent is not None:
            self.linear_extent = check_units({'linear_extent':linear_extent}, default_star_vals)['linear_extent']
        if dim is not None:
            self.dim = check_units({'dim':dim}, default_star_vals)['dim']
        if wavelength_range is not None:
            assert(len(wavelength_range) == 2)
            self.centre_wave = (self.wavelength_range[0] + self.wavelength_range[1])/2 
            self.photon_energy = (const.h*const.c/self.centre_wave).to('erg')
            self.radiance = calculate_stellar_radiance(self.wavelength_range, self.temp)
        self.pix_res = self.linear_extent/self.dim
        
        
        center = int(self.dim/2)
        x_grid = np.array([np.abs(np.linspace(0, self.dim-1, self.dim) - center)]*self.dim)            
        y_grid = x_grid.transpose()
        self.grid_distances = ((x_grid/self.dim)**2 + (y_grid/self.dim)**2)**0.5 * self.linear_extent
        
        flux = (self.radiance * np.pi*un.sr * self.radius**2/self.grid_distances**2).to('erg/s/cm**2') # should there be a 4pi here?
        flux[self.grid_distances < self.radius] = self.radiance * np.pi*un.sr # should there be a 4pi here?

        self.grid_stellar_energy_fluxes = flux
        self.grid_stellar_photon_fluxes = self.grid_stellar_energy_fluxes/self.photon_energy
        self._linear_to_angular()
        return
    
    
    def make_cme(self, smear:bool = False, smear_sig=3, **kwargs):
        """
        Makes CME class instance and related grids
        :param smear: If True, smooths over the CME grid using guassian filtering, defaults to False
        :type smear: bool, optional
        :param smear_sig: Signficance of smearing/smoothing, defaults to 3
        :type smear_sig: float, optional
        """
        cme = CME(dim=self.dim)
        for k in kwargs:
            cme.__dict__.update({k:kwargs[k]})
            
        cme.build_cme_normalized_grid(smear=smear, smear_sig=smear_sig)
        cme.make_grids()
        self.cme = cme
        return
    
    
    def make_wind(self, **kwargs):
        """
        Makes Wind class instance and related grids
        """
        wind = Wind()
        for k in kwargs:
            wind.__dict__.update({k:kwargs[k]})
        wind.dim = self.dim
        wind.make_grids()
        self.wind = wind
        return
    
    
    def particle_to_photon_flux(self):
        """
        Converts the particle area of the wind and CME to photon fluxes
        """
        if "grid_stellar_photon_fluxes" not in self.__dict__:
            self.make_stellar_grids()
        if "cme" not in self.__dict__:
            print("CME has not been made; constructing now")
            self.make_cme()
        if "wind" not in self.__dict__:
            print("Wind has not been made; constructing now")
            self.make_wind()
            
        self.grid_wind_photons = self.wind.grid_scattering_factor * self.grid_stellar_photon_fluxes
        self.grid_cme_photons = (self.cme.grid_number * const.sigma_T * self.grid_stellar_photon_fluxes/self.pix_res**2).to("s**-1 * cm**-2")
        return 
    
    
    def _linear_to_angular(self):
        """
        Converts from linear scale to angular scale
        """
        dist_norm = self.grid_distances/(self.linear_extent)
        theta = (self.linear_extent/self.distance).to('') * un.rad.to('arcsec')
        self.grid_distances_angular = dist_norm*theta
        self.pix_res_ang = theta*un.arcsec/self.dim
        self.angular_extent = theta*un.arcsec
        return
    
    def calc_pol(self):
        return
    
    
    def plot_stellar_flux(self, unit:str = 'photon', scale:str = 'linear', logscale:bool = True):
        """
        Plots stellar fluxes either in terms of # of photons or energy at the Earth 
        :param unit: defines whether to plot the 'photon' or 'energy' flux, defaults to 'photon'
        :type unit: str, optional
        :param scale: defines whether to plot in 'angular' or 'linear' units, defaults to 'linear'
        :type scale: str, optional
        :param logscale: defines whether to plot logscale or not, defaults to True
        :type logscale: bool, optional
        :return: figure and axis instances
        :rtype: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot

        """
        assert(scale.lower() in ['linear', 'angular'])
        if unit.lower() == 'photon':
            units = "s**-1 * cm**-2"
            dat = self.grid_stellar_photon_fluxes
            cbar_label = r'photons/s/cm$^2$'
        elif unit.lower() == 'energy':
            units = "erg * s**-1 * cm**-2"
            dat = self.grid_stellar_energy_fluxes
            cbar_label = r'erg/s/cm$^2$'
            
        if scale.lower() == 'linear':
            # convert to flux received by a detector
            dat = (dat * self.pix_res**2/ (4 * np.pi * self.distance**2)).to(units)
            d = self.linear_extent.to('AU').value
            axis_label = 'Distance [AU]'            
        elif scale.lower() == 'angular':
            d = self.angular_extent.to('arcsec').value
            axis_label = 'Distance [arcsec]'
            
            
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
        return fig, ax
    
    
    def plot_particle_fluxes(self, wind:bool = True, cme:bool = True, logscale:bool = True, scale:str = 'linear'):
        """
        plots the photons produced by the wind and/or cme
        :param wind: Defines whether to include the wind in the image, defaults to True
        :type wind: bool, optional
        :param cme: Defines whether to include the CME in the image, defaults to True
        :type cme: bool, optional
        :param logscale: defines whether to plot logscale or not, defaults to True
        :type logscale: bool, optional
        :param scale: defines whether to plot in 'angular' or 'linear' units, defaults to 'linear'
        :type scale: str, optional
        :return: figure and axis instances
        :rtype: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot

        """
        self.particle_to_photon_flux()
        dat = np.zeros(self.grid_cme_photons.shape) * (un.s * un.cm**2)**-1
        title = f"System distance : {self.distance}"
        if wind:
            dat += self.grid_wind_photons
            title = f"{title}\nMdot={self.wind.Mdot}, v_wind={self.wind.vwind}"
        if cme:
            dat += self.grid_cme_photons
            title = f"{title}\n CME mass={self.cme.mass}"
        
        if scale.lower() == 'linear':
            # convert to flux received by a detector
            dat = (dat * self.pix_res**2/ (4 * np.pi * self.distance**2)).to('s**-1 * cm**-2')
            d = self.linear_extent.to('AU').value
            axis_label = 'Distance [AU]'
        elif scale.lower() == 'angular':
            d = self.angular_extent.to('arcsec').value
            axis_label = 'Distance [arcsec]'
        cbar_label = r'photons/s/cm$^2$'
            
        if logscale:
            dat = np.log10(dat.value)
            cbar_label = f"log10({cbar_label})"
            
        extents = [-d/2, d/2,-d/2, d/2]
        fig, ax = plt.subplots(1,1)
        im = ax.imshow(dat, extent=extents)

        ax.set_xlabel(axis_label)
        ax.set_ylabel(axis_label)
        ax.set_title(title)
        cbar = plt.colorbar(im)
        cbar.set_label(cbar_label)
        return fig, ax