from astropy import units as un, constants as const
from misc import check_units, make_distance_grid
from default_vals import default_star_vals
import numpy as np
from cme import CME2D, CME3D
from wind import Wind2D, Wind3D
import matplotlib.pyplot as plt
spectralIntensityUnits = 'erg*s**-1*cm**-2*nm**-1'


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
        
    h = const.h
    c = const.c
    k = const.k_B
    numerator = 2 * h*c**2/wavelength**5
    denominator = np.exp((h*c/(wavelength*k*temp)).to('')) -1
    spectral_intensity = (numerator/denominator).to(spectralIntensityUnits)
    return spectral_intensity.value


def make_bb_spectrum(temp, wavelength_range = [100,1000], spectral_resolution=1):
    """
    Makes a blackbody spectrum; updates the spectrum and wavelengths properties of the class
    :param wavelength_range: The wavelength range to make the spectrum over, defaults to [100,1000] (units of nanometres)
    :type wavelength_range: array or list of ints,floats, or astropy.units.quantity.Quantity, optional
    :param spectral_resolution: spectral resolution for the wavelength array, defaults to 1 (units of nanometres)
    :type spectral_resolution: int or float, optional

    """
    assert(len(wavelength_range) == 2)
    
    if type(wavelength_range[0]) == un.quantity.Quantity:
        wavelength_range = [w.to('nm').value for w in wavelength_range]
    
    if type(spectral_resolution) == un.quantity.Quantity:
        spectral_resolution = spectral_resolution.to('nm').value
    
    wavelengths = np.arange(wavelength_range[0], wavelength_range[1]+spectral_resolution, spectral_resolution)*un.nm
    sp_radiance = spectral_radiance(wavelengths, temp)* un.quantity.Unit(spectralIntensityUnits)/un.sr
    spectrum = sp_radiance
    wavelengths = wavelengths
    return Spectrum(wavelengths, spectrum)


class Spectrum:
    def __init__(self, wavelengths, spectrum, wavelength_unit='nm', spectrum_unit = 'erg * nm**-1 * s**-1 * sr**-1 * cm**-2'):
        self.wavelength_unit = un.Unit(wavelength_unit)
        self.spectrum_unit = un.Unit(spectrum_unit)
        
        if type(spectrum[0]) != un.quantity.Quantity:
            self.spectrum = spectrum * self.spectrum_unit
        else:
            try:
                self.spectrum = spectrum.to(self.spectrum_unit)
            except Exception as e:
                print(e)
        
        if type(wavelengths[0]) != un.quantity.Quantity:
            self.wavelengths = wavelengths * self.wavelength_unit
        else:
            try:
                self.wavelengths = wavelengths.to(self.wavelength_unit)
            except Exception as e:
                print(e)
        
        self.photon_spectrum = self.spectrum/(const.h * const.c/wavelengths).to('erg')
        return
    
    
    def plot(self, band=[100*un.nm, 1000*un.nm], logy:bool = False, plot_band:bool = True, wavelength_unit='nm', yunit='energy'):
        """
        Plots the spectrum
        :param logy: if True, plots the y axis in log scale, defaults to False
        :type logy: bool, optional
        :param plot_band: if True, will fill in the wavelengths corresponding to the wavelength_range property, defaults to True
        :type plot_band: bool, optional
        :param wavelength_unit: unit to plot the wavelengths in, defaults to 'nm'
        :type wavelength_unit: str or astropy.units.quantity.Unit, optional
        :param yunit: unit to plot the spectrum, either 'energy' or 'photon', defaults to 'energy'
        :type yunit: str, optional
        :return: the figure and axis objects of the plot
        :rtype: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot

        """
        assert(yunit.lower == 'energy' or 'photon')
        assert("spectrum" in self.__dict__), "Spectrum has not been made yet. Run make_bb_spectrum"
        
        if yunit.lower() == 'energy':
            ylabel = f"Radiance ({self.spectrum.unit.to_string()})"
            y = self.spectrum
        elif yunit.lower() == 'photon':
            y = self.spectrum/(const.h*const.c/self.wavelengths).to('erg')
            ylabel = f"Photon spectral flux (photons {y.unit.to_string()})"
            
        fig, ax = plt.subplots()
        w = self.wavelengths.to(wavelength_unit)
        
        if logy:
            ax.semilogy(w, y)
        else:
            ax.plot(w, y)
            
        if plot_band:
            idxs = np.where((w>=band[0]) & (w<= band[1]))[0]
            ax.fill_between(w[idxs].value, y[idxs].value, y2=np.nanmin(y.value), alpha = 0.3)
            
        ax.set_xlabel(f"Wavelength ({w.unit.to_string()})")
        ax.set_ylabel(ylabel)
        return fig, ax
    

    
class Star:
    def __init__(self, ndim:int = 2, custom_spectrum:Spectrum = None, **kwargs):
        """
        Class for determining observed stellar fluxes
        :param **kwargs: stellar radius (radius), stellar temperature (temp), 
        wavelength range (wavelength_range), distance of star (distance), the grid
        dimension (dim), the physical extent of the grid (physical_diameter)
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
        if custom_spectrum == None:    
            self.spectrum = make_bb_spectrum(self.temp, self.wavelength_range)
            
        self.ndim = ndim
        self.integrate_over_band()
        self.luminosity = self.radiance * 4 * np.pi * un.sr * self.radius.to('cm')**2
        self.centre_wave = (self.wavelength_range[0] + self.wavelength_range[1])/2 
        self.photon_energy = (const.h*const.c/self.centre_wave).to('erg')
        self.make_stellar_grids()
        return
    
    
    def integrate_over_band(self, band=None):
        """
        Integrates the blackbody spectrum over the wavelength_range (band). Updates the radiance property
        :param band: wavelength range to integrate over. If None, uses the class instance's wavelength_range, defaults to None
        :type band: list or array of length two, optional
        """
        assert("spectrum" in self.__dict__), "Spectrum has not been made yet. Run make_bb_spectrum"
        
        if band is not None:
            assert(len(band) == 2)
            if all(isinstance(w,(int,float)) for w in band):
                band = [b*un.nm for b in band]
            self.wavelength_range = band
        else:
            band = self.wavelength_range
            
        assert(band[0] >= self.spectrum.wavelengths[0] and band[-1] <= self.spectrum.wavelengths[-1])    
        idxs = np.where((self.spectrum.wavelengths >= band[0]) & (self.spectrum.wavelengths <= band[1]))
        self.radiance = np.nansum(self.spectrum.spectrum[idxs]) * (self.spectrum.wavelengths[1] - self.spectrum.wavelengths[0])
        self.photon_radiance = np.nansum(self.spectrum.photon_spectrum[idxs]) * (self.spectrum.wavelengths[1] - self.spectrum.wavelengths[0])
        return
    
    
    def make_distance_grid(self, physical_diameter=None, dim=None):
        if physical_diameter is not None:
            self.physical_diameter = check_units({'physical_diameter':physical_diameter}, default_star_vals)['physical_diameter']
        if dim is not None:
            self.dim = check_units({'dim':dim}, default_star_vals)['dim']
        self.grid_distances = make_distance_grid(self.dim, self.physical_diameter, ndim=self.ndim)
        return
    
    
    def make_stellar_grids(self, physical_diameter=None, dim:int=None, wavelength_range=None):
        """
        Makes grids for physical distances centered on star, angular distances, energy flux, and photon flux
        :param physical_diameter: Physical extent of the grid. If None, uses the class instance's value. defaults to None
        :type physical_diameter: astropy.quantity.Quantity, optional
        :param dim: Number of pixels in each grid dimension. If None, uses the class instance's value, defaults to None
        :type dim: int, optional
        :param wavelength_range: wavelength range to evaluate stellar energy and photon fluxes. If None, uses the class instance's value, defaults to None
        :type wavelength_range: list or numpy.array of astropy.quantity.Quantity, optional
        """
        if physical_diameter is not None or dim is not None or 'grid_distances' not in self.__dict__:
            self.make_distance_grid(physical_diameter=physical_diameter, dim=dim)
            
        if wavelength_range is not None:
            assert(len(wavelength_range) == 2)
            if (any(isinstance(w, (int,float))) for w in wavelength_range ):
                assert((all(isinstance(w, (int,float))) for w in wavelength_range)==True), "Mixed datatype for wavelength_range"
                wavelength_range = [w*un.nm for w in wavelength_range]
                
            self.wavelength_range = wavelength_range
            self.centre_wave = (self.wavelength_range[0] + self.wavelength_range[1])/2 
            self.photon_energy = (const.h*const.c/self.centre_wave).to('erg')
            self.integrate_over_band(self.wavelength_range)
            
        self.pix_res = self.physical_diameter/self.dim
        
        energy_flux = (self.radiance * np.pi*un.sr * self.radius**2/self.grid_distances**2).to('erg/s/cm**2') # should there be a 4pi here?
        energy_flux[self.grid_distances < self.radius] = self.radiance * np.pi*un.sr # should there be a 4pi here?
        
        photon_flux = (self.photon_radiance * np.pi*un.sr * self.radius**2/self.grid_distances**2).to('s**-1 * cm**-2') # should there be a 4pi here?
        photon_flux[self.grid_distances < self.radius] = self.photon_radiance * np.pi*un.sr 
        
        self.grid_stellar_energy_fluxes = energy_flux
        self.grid_stellar_photon_fluxes = photon_flux
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
        if self.ndim == 2:
            cme = CME2D(dim=self.dim, physical_diameter=self.physical_diameter, **kwargs)
            cme.build_cme_normalized_grid(smear=smear, smear_sig=smear_sig)
        elif self.ndim == 3:
            cme = CME3D(dim=self.dim, physical_diameter=self.physical_diameter, **kwargs)
            cme.build_cme_normalized_grid()
            
        cme.make_grids()
        self.cme = cme
        return
    
    
    def make_wind(self, **kwargs):
        """
        Makes Wind class instance and related grids
        """
        if self.ndim == 2:
            wind = Wind2D(dim=self.dim, physical_diameter=self.physical_diameter, **kwargs)
        elif self.ndim == 3:
            wind = Wind3D(dim=self.dim, physical_diameter=self.physical_diameter, **kwargs)
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
            
        self.grid_wind_photons = (self.wind.grid_particles * const.sigma_T.cgs * self.grid_stellar_photon_fluxes).cgs
        self.grid_cme_photons = (self.cme.grid_particles * const.sigma_T * self.grid_stellar_photon_fluxes).cgs#/self.pix_res**2).to("s**-1 * cm**-2")
        return 
    
    
    def _linear_to_angular(self):
        """
        Converts from linear scale to angular scale
        """
        dist_norm = self.grid_distances/(self.physical_diameter)
        theta = (self.physical_diameter/self.distance).to('') * un.rad.to('arcsec')
        self.grid_distances_angular = dist_norm*theta
        self.pix_res_ang = theta*un.arcsec/self.dim
        self.angular_extent = theta*un.arcsec
        return
    
    
    def calc_pol(self):
        return
    
    
    def _plot_properties(self, unit:str = 'photon', scale:str = 'linear', logscale:bool = True, index=None, axis=0):
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
            d = self.physical_diameter.to('AU').value
            axis_label = 'Distance [AU]'            
        elif scale.lower() == 'angular':
            d = self.angular_extent.to('arcsec').value
            axis_label = 'Distance [arcsec]'
            
            
        if logscale:
            dat = np.log10(dat.value)
            cbar_label = f"log10({cbar_label})"
            
        extents = [-d/2, d/2,-d/2, d/2]
        fig, ax = plt.subplots(1,1)
        return fig, ax, dat, extents, cbar_label, axis_label
    
    
    def _plot_stellar_flux_2D(self, unit:str = 'photon', scale:str = 'linear', logscale:bool = True):
        fig, ax, dat, extents, cbar_label, axis_label = self._plot_properties(unit=unit, scale=scale, logscale=logscale)

        im = ax.imshow(dat, extent=extents)
        ax.set_xlabel(axis_label)
        ax.set_ylabel(axis_label)
        cbar = plt.colorbar(im)
        cbar.set_label(cbar_label)
        return fig, ax
    
    
    def _plot_stellar_flux_3D(self, unit:str = 'photon', scale:str = 'linear', logscale:bool = True, index:int=0, axis:int = 0):
        fig, ax, dat, extents, cbar_label, axis_label = self._plot_properties(unit=unit, scale=scale, logscale=logscale)
 
        if axis == 0:
            dat = dat[index, :, :]
        if axis == 1:
            dat = dat[:, index, :]
        if axis == 2:
            dat = dat[:, :, index]
            
        im = ax.imshow(dat, extent=extents)
        ax.set_xlabel(axis_label)
        ax.set_ylabel(axis_label)
        cbar = plt.colorbar(im)
        cbar.set_label(cbar_label)
        fig.suptitle(f"Surface distance from star: {self.grid_distances[index].min():.2f}")
        return fig, ax
    
    
    def plot_stellar_flux(self, unit:str = 'photon', scale:str = 'linear', logscale:bool = True, index:int=0, axis:int = 0):
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
        if self.ndim == 2:
            self._plot_stellar_flux_2D(unit=unit, scale=scale, logscale=logscale)
        elif self.ndim == 3:
            self._plot_stellar_flux_3D(unit=unit, scale=scale, logscale=logscale, index=index, axis=axis)
        return 
    
    
    def plot_particle_fluxes(self, wind:bool = True, cme:bool = True, logscale:bool = True, scale:str = 'linear', index=None, axis=None):
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
            d = self.physical_diameter.to('AU').value
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
    

    
