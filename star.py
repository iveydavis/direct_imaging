from astropy import units as un, constants as const
from misc import check_units
from default_vals import default_star_vals
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
    :param wavelength_range: range of wavelength
    :type wavelength_range: TYPE
    :param temp: DESCRIPTION
    :type temp: TYPE
    :param radius: DESCRIPTION
    :type radius: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    assert(len(wavelength_range) == 2)
    wav_min = wavelength_range[0].to('nm').value
    wav_max = wavelength_range[1].to('nm').value
    d = check_units(temp=temp, radius=radius)
    radiance, toss = integrate.quad(spectral_radiance, wav_min, wav_max, (d['temp'].value))*un.erg/un.s/un.sr/un.cm**2
    lum = radiance * 4 * np.pi * un.sr * radius.to('cm')**2
    return lum


def calculate_stellar_radiance(wavelength_range, temp):
    """
    
    :param wavelength_range: DESCRIPTION
    :type wavelength_range: TYPE
    :param temp: DESCRIPTION
    :type temp: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    wav_min = wavelength_range[0].to('nm').value
    wav_max = wavelength_range[1].to('nm').value
    d = check_units(temp=temp)
    radiance, toss = integrate.quad(spectral_radiance, wav_min, wav_max, (d['temp'].value))
    return radiance*un.erg/un.s/un.sr/un.cm**2


class Star:
    def __init__(self, **kwargs):
        for k in default_star_vals:
            if k in kwargs:
                self.__dict__.update({k:kwargs[k]})
            else:
                self.__dict__.update({k:default_star_vals[k]})
                
        assert(len(self.wavelength_range) == 2), "wavelength_range should be list or numpy.ndarray of length 2"
        return_dict = check_units(temp=self.temp, radius=self.radius, distance=self.distance, linear_extent=self.linear_extent)
        self.temp = return_dict['temp']
        self.radius = return_dict['radius']
        self.luminosity = calculate_stellar_luminosity(self.wavelength_range, self.temp, self.radius)
        self.radiance = calculate_stellar_radiance(self.wavelength_range, self.temp)
        
        self.linear_extent = return_dict['linear_extent']
        self.distance = return_dict['distance']
        
        if self.dim % 2 == 0:
           self.dim += 1
           
        self.pix_res = self.linear_extent/self.dim
        self.center_wave = (self.wavelength_range[0] + self.wavelength_range[1])/2 
        self.photon_energy = (const.h*const.c/self.center_wave).to('erg')
        
        self.cme = None
        self.wind = None
        return
    
    
    def reset_spectral_properties(self, wavelength_range):
        self.wavelength_range = wavelength_range
        self.center_wave = (wavelength_range[0] + wavelength_range[1])/2 
        self.photon_energy = (const.h*const.c/self.center_wave).to('erg')
        self.make_stellar_grids()
        return
    
    
    def make_stellar_grids(self):
        x_grid = np.ones((self.dim, self.dim))
        center = [int(self.dim/2), int(self.dim/2)]
        x_grid = x_grid*np.abs(np.linspace(0, x_grid.shape[0]-1, x_grid.shape[0]) - center[0])            
        y_grid = x_grid.transpose()
        
        self.grid_distances = ((x_grid/self.dim)**2 + (y_grid/self.dim)**2)**0.5 * self.linear_extent
        
        # thetamaxes = np.arctan((self.radius/np.sqrt(self.grid_distances**2 - self.radius**2)).to('')) 
        # interior_angles = np.pi/2 * un.rad - thetamaxes
        # solid_angles = np.abs(4* np.pi * np.cos(2 * interior_angles) )* un.sr
        # flux = (self.radiance * solid_angles * self.radius**2 / self.grid_distances**2).to('erg/s/cm**2')
        flux = (self.radiance * np.pi*un.sr * self.radius**2/self.grid_distances**2).to('erg/s/cm**2')
        idxs = np.where(self.grid_distances < self.radius)
        flux[idxs] = np.nan
        self.grid_stellar_energy_fluxes = flux
        self.grid_stellar_photon_fluxes = self.grid_stellar_energy_fluxes/self.photon_energy
        self.linear_to_angular()
        return
    
    
    def make_cme(self, smear=False, smear_sig=3, **kwargs):
        cme = CME(dim=self.dim)
        for k in kwargs:
            cme.__dict__.update({k:kwargs[k]})
            
        cme.build_cme_normalized_grid(smear=smear, smear_sig=smear_sig)
        cme.make_grids()
        self.cme = cme
        return
    
    
    def make_wind(self, **kwargs):
        wind = Wind()
        for k in kwargs:
            wind.__dict__.update({k:kwargs[k]})
        wind.dim = self.dim
        wind.make_grids()
        self.wind = wind
        return
    
    
    def particle_to_photon_flux(self):
        if self.cme is None:
            self.make_cme()
        if self.wind is None:
            self.make_wind()
            
        self.grid_wind_photons = self.wind.grid_scattering_factor * self.grid_stellar_photon_fluxes
        self.grid_cme_photons = (self.cme.grid_number * const.sigma_T * self.grid_stellar_photon_fluxes/self.pix_res**2).to("s**-1 * cm**-2")
        return 
    
    
    def linear_to_angular(self):
        dist_norm = self.grid_distances/(self.linear_extent)
        theta = (self.linear_extent/self.distance).to('') * un.rad.to('arcsec')
        self.grid_distances_angular = dist_norm*theta
        self.pix_res_ang = theta*un.arcsec/self.dim
        self.angular_extent = theta*un.arcsec
        return
    
    
    def plot(self, wind=True, cme=True, logscale=True, scale='linear'):
        
        dat = np.zeros(self.grid_cme_photons.shape) * (un.s * un.cm**2)**-1
        title = f"System distance : {self.distance}"
        if wind:
            dat += self.grid_scattering_factor
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
        