#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:30:49 2026

@author: idavis
"""
from star import Star
from instrument import Instrument
from default_vals import default_star_vals, default_instrument, bands
from misc import check_units, get_band_info
from hcipy import make_focal_grid
import numpy as np

class Observation:
    def __init__(self, instrument:Instrument, wavelength_range=None, band:str = None, star:Star = None, source_dim:int = 513):
        if star is not None:
            assert(star.dim == source_dim), f"Star's dim property ({star.dim})should match the source_dim quantity ({source_dim})"
        assert(len(wavelength_range) == 2)
        if band is None:
            assert(wavelength_range is not None), "Either band or wavelength_range needs to be defined"
        elif band is not None:
            if wavelength_range is not None:
                raise Warning("Both band and wavelength range are defined. Using user-defined wavelength range, which might not match traditional values")
                self.wavelength_range = wavelength_range
                self.centre_wave = (wavelength_range[0] + wavelength_range[1])/2
            elif wavelength_range is None:
                self.wavelength_range, self.centre_wave = get_band_info(band)
        
        self.star = star
        self.instrument = instrument
        self._update_focal_grid()
        
        if "pupil_grid" in self.instrument.__dict__:
            if self.instrument.pupil_grid.shape[0] != source_dim:
                self._update_optical_elements()
        elif "pupil_grid" not in self.instrument.__dict__:
            self._update_optical_elements()
        return
    
    
    def _update_focal_grid(self, res_sampling=None, pupil_diameter=None, focal_length=None, n_pix=None):
        if res_sampling is None:
            res_sampling = self.instrument.res_sampling
        if pupil_diameter is None:
            pupil_diameter = self.instrument.ap_diam
        if n_pix is None:
            n_pix = self.instrument.n_pix
        if focal_length is None:
            focal_length = self.instrument.focal_length
        radius = int(n_pix/2)
        
        fg = make_focal_grid(q=res_sampling, 
                        num_airy = np.ceil(radius/res_sampling).astype(int),
                        pupil_diameter = pupil_diameter,
                        focal_length = focal_length,
                        reference_wavelength=self.centre_wave)
        self.instrument.focal_grid = fg
        return
    
    
    def add_star(self, **kwargs):
        star = Star(dim=self.source_dim, wavelength_range=self.wavelength_range)
        for k in kwargs:
            if k in default_star_vals and (k != "dim" or "wavelength_range"):
                star.__dict__.update({k:kwargs[k]})
                
        star.make_stellar_grids()
        self.star = star
        
        return
    
    def _update_optical_elements(self, charge:int=6, super_sample=True, super_samp_fact=4, lyot_fraction=0.95, pupil_factor=1.125):
        self.instrument.add_optical_elements(n_pix=self.source_dim, 
                                             charge=charge, 
                                             super_sample=super_sample, 
                                             super_samp_fact=super_samp_fact, 
                                             lyot_fraction=lyot_fraction, 
                                             pupil_factor=pupil_factor)
        return
    
    
    def conduct_observation(self):
        self.star.particle_to_photon_flux()
        source = (np.nansum((self.star.grid_cme_photons, self.star.grid_wind_photons), axis=0)*self.star.pix_res**2 ).to('1/s')
        self.wf = self.instrument.source_to_wavefront(source)

        lyot_plane = self.instrument.coro(self.wf)
        img_ref = self.prop(self.wf)

        post_lyot_mask = self.lyot_stop(lyot_plane)
        img = self.prop(post_lyot_mask).intensity
        
        self.img_ref = img_ref
        self.img = img
        return
    
    