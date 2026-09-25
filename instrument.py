#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from hcipy import (make_pupil_grid, 
                   make_focal_grid,
                   evaluate_supersampled,
                   make_circular_aperture,
                   FraunhoferPropagator,
                   VortexCoronagraph,
                   Apodizer,
                   Wavefront)
from misc import get_band_info, check_units
from default_vals import instruments, default_instrument, bands
import numpy as np


class Band:
    def __init__(self, bandname:str=None, **kwargs):
        self.bandname = bandname.upper()
        if bandname is not None:
            self.bandpass, self.centre_wave = get_band_info(self.bandname)
                
        elif bandname is None:
            assert("bandpass" in kwargs), "Either band name or bandpass needs to be defined"
            assert(len("bandpass") == 2), "Bandpass needs to be of length 2"
            self.bandpass = kwargs['bandpass']    
            self.centre_wave = (self.bandpass[1] + self.bandpass[0])/2
            self.bandname = ''
        return

        
class Instrument:
    def __init__(self, band, name:str = None, res_sampling=6, **kwargs):
        # one band per instrument instance
        check_units(kwargs, default_instrument)
        
        if 'N' in kwargs:
            self.iwa_fac = kwargs['N']
        else:
            self.iwa_fac = 1
            
        if name is not None:
            name = name.upper()
            assert(name in instruments), f"{name} not a recognized instrument ({instruments})"
            assert(band in instruments[name]['bands'])
            self.name = name
            self.ap_diam = instruments[name]['ap_diam']
            self.n_pix = instruments[name]['n_pix']
            self.focal_length = instruments[name]['focal_length']
            
        else:
            assert('ap_diam' in kwargs)
            assert('n_pix' in kwargs)
            self.name = ''
            self.ap_diam = kwargs['ap_diam']
            self.n_pix = kwargs['n_pix']
            self.focal_length = kwargs['focal_length']
        
        assert(type(band) == str or Band), "band should either be str of band name or Band class instance"
        if type(band) == str:
            self.band = Band(band)
        elif type(band) == Band:
            self.band = band
            
        self.iwa = self.iwa_fac * self.band.centre_wave / self.ap_diam
        self.ang_res = 1.22 * self.band.centre_wave / self.ap_diam # assumes diffraction limited telescope
        self.fov = self.ang_res * self.n_pix

        spat_res = self.band.centre_wave * self.focal_length/self.ap_diam
        radius = int(self.n_pix/2)
        self.res_sampling = res_sampling
        self.radius = radius
        self.focal_grid = make_focal_grid(q=res_sampling,
                                          num_airy = np.ceil(radius/res_sampling).astype(int),
                                          pupil_diameter = self.ap_diam,
                                          focal_length = self.focal_length,
                                          reference_wavelength=self.band.centre_wave)
        
        return
    
    
    def add_optical_elements(self, n_pix:int=201, charge:int=6, super_sample=True, super_samp_fact=4, lyot_fraction=0.95, pupil_factor=1.125):
        self.pupil_grid = make_pupil_grid(n_pix, diameter=self.ap_diam*pupil_factor)
        self.prop = FraunhoferPropagator(self.pupil_grid, self.focal_grid)
        
        ap = make_circular_aperture(self.ap_diam.to('m').value, )
        lyot = make_circular_aperture(self.ap_diam.to('m').value*lyot_fraction)
        if super_sample:
            self.ap = evaluate_supersampled(ap, 
                                            self.pupil_grid, 
                                            super_samp_fact)
            self.lyot = evaluate_supersampled(lyot, 
                                            self.pupil_grid, 
                                            super_samp_fact)
        elif not super_sample:
            self.ap = ap(self.pupil_grid)
            self.lyot = lyot(self.pupil_grid)
        
        self.coronagraph = VortexCoronagraph(self.pupil_grid, charge=charge)
        self.lyot_stop = Apodizer(lyot)
        return

    
    def source_to_wavefront(self, source):
        assert("pupil_grid" in self.__dict__), "Need to run add_optical_elements before calculating wavefront"
        assert(source.shape == self.pupil_grid.shape), "The source image should be the same dimensions as the pupil_grid"
        
        arg = 0 + 0j
        xgrid = self.pupil_grid.x.reshape(self.pupil_grid.shape)
        ygrid = self.pupil_grid.y.reshape(self.pupil_grid.shape)
        
        for i in range(xgrid.shape[0]):
            for j in range(xgrid.shape[1]):    
                xfac = xgrid[i,j]
                yfac = ygrid[i,j]
                arg += source[i,j]*np.exp(xfac * np.pi *self.pupil_grid.x + yfac*np.pi*self.pupil_grid.y)
                
        wf = Wavefront(self.ap * arg, wavelength=self.band.centre_wave.to('m'))
        return wf
    
