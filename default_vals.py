#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from astropy import units as un, constants as const
from numpy import pi

default_cme_vals = {"dim":513,
                    "dR_factor":0.2, 
                    "phi_HW":pi/4,
                    "horizontal_height_factor":0,
                    "vertical_height_factor":1.5,
                    "depth_factor":1,
                    "mass":1e19*un.g,
                    "linear_extent": 6*un.au}

default_wind_vals = {"dim":513, 
                     "Mdot":6e-13*un.M_sun/un.yr, 
                     "vwind":400*un.km/un.s, 
                     "linear_extent":6*un.au}

default_star_vals = {"temp":5000*un.K, 
                     "radius":0.7*const.R_sun, 
                     "distance":3.2*un.pc, 
                     "dim":513, 
                     "linear_extent":6*un.au, 
                     "wavelength_range":[100*un.nm,1000*un.nm]}


bands = {'I': {'center': 806*un.nm, 'bandwidth':149*un.nm},
         'Y': {'center': 1020*un.nm, 'bandwidth':120*un.nm},
         'J': {'center': 1220*un.nm, 'bandwidth':213*un.nm},
         'H': {'center': 1630*un.nm, 'bandwidth':307*un.nm},
         'K': {'center': 2190*un.nm, 'bandwidth':390*un.nm}}


instruments = {"GPI":{"bands":['Y', 'J', 'H', 'K'],
                       "n_pix":200,
                       "ap_diam":8*un.m,
                       "focal_length": 0.23 * un.m, # https://www.gemini.edu/observing/telescopes-and-sites/telescopes
                       "iwa_fac": 1
                       }, # limiting magnitude is 8, https://arxiv.org/pdf/0704.1454
               
               "HWO": {"bands":['I', 'Y', 'J', 'H', 'K'],
                      "n_pix":200,
                      "ap_diam": 8*un.m,
                      "focal_length": 0.23 * un.m, # I'm guessing here
                      "iwa_fac": 1
                          }
            }

default_instrument = instruments["HWO"]
