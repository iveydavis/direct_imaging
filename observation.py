#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from star import Star
from instrument import Instrument
from default_vals import default_star_vals, default_instrument, bands
from misc import check_units, get_band_info
from hcipy import make_focal_grid
import numpy as np
from astropy import units as un
import warnings

class Observation:
    def __init__(self, instrument:Instrument, wavelength_range=None, band:str = None, star:Star = None, source_dim:int = 513, normalise=True):
        if star is not None:
            assert(star.dim == source_dim), f"Star's dim property ({star.dim})should match the source_dim quantity ({source_dim})"
        
        if wavelength_range is not None:
            assert(len(wavelength_range) == 2)
        
        if band is None:
            assert(wavelength_range is not None), "Either band or wavelength_range needs to be defined"
        elif band is not None:
            if wavelength_range is not None:
                warnings.warn("Both band and wavelength range are defined. Using user-defined wavelength range, which might not match traditional values")
                self.wavelength_range = wavelength_range
                self.centre_wave = (wavelength_range[0] + wavelength_range[1])/2
            elif wavelength_range is None:
                self.wavelength_range, self.centre_wave = get_band_info(band)
        
        if self.centre_wave != instrument.band.centre_wave:
            warnings.warn(f"Observation centre wavelength ({self.centre_wave}) does not match instrument centre wavelength ({instrument.band.centre_wave})")
        
        self.source_dim = source_dim
        self.star = star
        self.instrument = instrument

        if self.instrument.pupil_grid.shape[0] != source_dim or normalise != self.instrument.normalise:
            self._update_optical_elements(pupil_diam_pix=source_dim, normalise=normalise)
            
        if 'coronagraph' not in self.instrument.__dict__:
            self.instrument.add_coronagraph_optics()
        return
    

    def add_star(self, **kwargs):
        star = Star(dim=self.source_dim, wavelength_range=self.wavelength_range)
        for k in kwargs:
            if k in default_star_vals and (k != "dim" or "wavelength_range"):
                star.__dict__.update({k:kwargs[k]})
                
        star.make_stellar_grids()
        self.star = star
        
        return
    
    
    def _update_optical_elements(self, **kwargs):
        self.instrument._update_grids(**kwargs)
        if 'charge' in kwargs or 'lyot_fraction' in kwargs:
            self.instrument.add_coronagraph_optics(charge=kwargs['charge'], 
                                                   lyot_fraction=kwargs['lyot_fraction'])
        return
    
    
    def make_obs_source(self, force_star:bool = True):
        self.star.particle_to_photon_flux()
        cme_photons = (self.star.grid_cme_photons * self.star.pix_res**2).to('1/s')
        wind_photons = (self.star.grid_wind_photons* self.star.pix_res**2).to('1/s')
        source = np.nansum((cme_photons, wind_photons), axis=0)
        
        if self.star.pix_res < self.star.radius:
            star_photons = self.star.radiance * np.pi * un.sr*self.star.pix_res.cgs**2/self.star.photon_energy
        else:
            star_photons = (self.star.radiance * 2 * np.pi * un.sr * self.star.radius**2/self.star.photon_energy).to('1/s')
        self.star_photons = star_photons
        source[np.isnan(source)] = star_photons
        source[np.isinf(source)] = star_photons
        if force_star:
            idx = np.where(self.star.grid_distances.value == 0)
            source[idx] = (star_photons).to('1/s').value
        self.source = source
        return
    
    
    def conduct_observation(self, force_star = True, stokes=['I']):        
        self.make_obs_source(force_star=force_star)
        self.instrument.source_to_image(self.source, stokes=stokes)
        self.stokes = stokes
        return
