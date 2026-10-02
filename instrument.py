#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from hcipy import (make_pupil_grid, 
                   make_focal_grid,
                   make_focal_grid_from_pupil_grid,
                   evaluate_supersampled,
                   make_circular_aperture,
                   FraunhoferPropagator,
                   VortexCoronagraph,
                   Apodizer,
                   Wavefront)
from misc import get_band_info, check_units, un
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
    def __init__(self, band, name:str = None, res_sampling:int = 6, 
                 normalise:bool = True, 
                 pupil_factor=1.125,
                 super_sample:bool = True,
                 super_samp_factor: int = 4,
                 pupil_diam_pix: int = 201,
                 **kwargs):
        
        # one band per instrument instance
        kwargs = check_units(kwargs, default_instrument)
        if not normalise and band.upper() == 'NORM':
            raise Warning("Band is normalised but not the optical path")
        self.normalise = normalise
        self.pupil_factor = pupil_factor
        self.super_sample = super_sample
        self.super_samp_factor = super_samp_factor
        self.pupil_diam_pix = pupil_diam_pix
            
        if name is not None:
            name = name.upper()
            assert(name in instruments), f"{name} not a recognized instrument ({instruments})"
            assert(band in instruments[name]['bands'] or band.upper()=='NORM')
            self.name = name
            self.ap_diam = instruments[name]['ap_diam']
            self.n_pix = instruments[name]['n_pix']
            self.focal_length = instruments[name]['focal_length']
            
        elif name is None:
            assert('ap_diam' in kwargs)
            assert('n_pix' in kwargs)
            self.name = ''
            self.ap_diam = kwargs['ap_diam']
            self.n_pix = kwargs['n_pix']
            self.focal_length = kwargs['focal_length']
            
        if self.normalise:
            self.focal_length = 1*un.m
            band = 'NORM'
        
        assert(type(band) == str or Band), "band should either be str of band name or Band class instance"
        if type(band) == str:
            self.band = Band(band)
        elif type(band) == Band:
            self.band = band
        
        self.reference_wavelength = self.band.centre_wave
        self.res_sampling = res_sampling
        self.pupil_grid = make_pupil_grid(pupil_diam_pix, diameter=self.ap_diam.to('m').value*pupil_factor)
        
        ap = make_circular_aperture(self.ap_diam.to('m').value)
        if super_sample:
            self.ap = evaluate_supersampled(ap, 
                                            self.pupil_grid, 
                                            super_samp_factor)
        elif not super_sample:
            self.ap = ap(self.pupil_grid)
            
        
        self.focal_grid = make_focal_grid_from_pupil_grid(self.pupil_grid,
                                                          q=self.res_sampling,
                                          num_airy=np.ceil(self.n_pix/self.res_sampling/2).astype(int),
                                          focal_length=self.focal_length.to('m').value,
                                          wavelength=self.reference_wavelength.to('m').value)
        self.prop = FraunhoferPropagator(self.pupil_grid, self.focal_grid)
        return
    

    def _update_grids(self, **kwargs):
        if 'normalise' in kwargs:
            normalise = kwargs['normalise']
            band = self.band
            if 'band' in kwargs and normalise:
                self.band = Band('NORM')
            elif 'band' in kwargs and not normalise:
                assert(type(band) == str or Band), "band should either be str of band name or Band class instance"
                if type(self.band) == str:
                    self.band = Band(band)
                elif type(band) == Band:
                    self.band = band                
                    
        for k in kwargs:
            if k in list(self.__dict__.keys()) and k != 'band':
                if k in list(default_instrument.keys()):
                    chk = check_units({k:kwargs[k]}, default_instrument)
                    val = chk[k]
                elif k not in list(default_instrument.keys()):
                    val = kwargs[k]
                self.__dict__.update({k: val})
                
        if self.normalise:
            self.focal_length = 1*un.m
            self.reference_wavelength = 1*un.m
                
        self.pupil_grid = make_pupil_grid(self.pupil_diam_pix, diameter=self.ap_diam*self.pupil_factor)
        ap = make_circular_aperture(self.ap_diam.to('m').value)
        if self.super_sample:
            self.ap = evaluate_supersampled(ap, 
                                            self.pupil_grid, 
                                            self.super_samp_factor)
        elif not self.super_sample:
            self.ap = ap(self.pupil_grid)
            
        
        self.focal_grid = make_focal_grid_from_pupil_grid(self.pupil_grid,
                                                          q=self.res_sampling,
                                          num_airy=np.ceil(self.n_pix/self.res_sampling/2).astype(int),
                                          focal_length=self.focal_length.to('m').value,
                                          wavelength=self.reference_wavelength.to('m').value)
        self.prop = FraunhoferPropagator(self.pupil_grid, self.focal_grid)
        return
    
    
    def add_coronagraph_optics(self, charge:int=6, lyot_fraction=0.95):
        self.charge = charge
        self.lyot_fraction = lyot_fraction
        lyot = make_circular_aperture(self.ap_diam.to('m').value*lyot_fraction)
        
        self.coronagraph = VortexCoronagraph(self.pupil_grid, charge=charge)
        self.lyot_stop = Apodizer(lyot)
        return


    def source_to_image(self, source, stokes:list = ['I']):
        assert(np.array(source.shape)[0] == self.pupil_grid.shape[0]), "The source image should be the same dimensions as the pupil_grid"
        assert('coronagraph' in self.__dict__), "Coronagraph hasn't been made yet"
        
        stokes_dict = {}
        for i,s in enumerate(stokes):
            assert(s.upper() in ['I', 'Q', 'U', 'V']), f"{s} not recognised stokes parameter"
            stokes_dict.update({s: i})
        
        xgrid = self.pupil_grid.x.reshape(self.pupil_grid.shape)
        ygrid = self.pupil_grid.y.reshape(self.pupil_grid.shape)
        
        img_shape = (len(stokes), self.focal_grid.size)
        img_ref = np.zeros(img_shape)
        img = np.zeros(img_shape)
        rw = self.reference_wavelength.to('m').value
        
        for i in range(xgrid.shape[0]):
            for j in range(xgrid.shape[1]):    
                xfac = xgrid[i,j]
                yfac = ygrid[i,j]
                arg = np.exp(xfac * np.pi*2j *self.pupil_grid.x + yfac*np.pi*2j*self.pupil_grid.y)
                
                wf = Wavefront(self.ap * arg, wavelength=rw)
                wf.total_power = source[i,j]
                img_ref_wf = self.prop(wf)
                
                lyot_plane = self.coronagraph(wf)
                post_lyot_mask = self.lyot_stop(lyot_plane)
                img_wf = self.prop(post_lyot_mask)
                
                for si, s in enumerate(stokes):
                    if s == 'I':
                        img[stokes_dict[s]] += img_wf.I
                        img_ref[stokes_dict[s]] += img_ref_wf.I
                    elif s == 'Q':
                        img[stokes_dict[s]] += img_wf.Q
                        img_ref[stokes_dict[s]] += img_ref_wf.Q
                    elif s == 'U':
                        img[stokes_dict[s]] += img_wf.U
                        img_ref[stokes_dict[s]] += img_ref_wf.U
                    elif s == 'V':
                        img[stokes_dict[s]] += img_wf.V
                        img_ref[stokes_dict[s]] += img_ref_wf.V
                    
        self.img = img.reshape((len(stokes), self.focal_grid.shape[0], self.focal_grid.shape[1]))
        self.img_ref = img_ref.reshape((len(stokes), self.focal_grid.shape[0], self.focal_grid.shape[1]))
        return 
    
