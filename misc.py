#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from default_vals import *
from astropy import units as un

def check_units(kwarg_dict, default_dict):
    for k in kwarg_dict.keys():
        if type(kwarg_dict[k]) == un.quantity.Quantity:
            assert(kwarg_dict[k].unit.is_equivalent(default_dict[k].unit)), f"{kwarg_dict[k]} should be in units of {default_dict[k].unit.physical_type}"
            
        elif type(kwarg_dict[k]) != un.quantity.Quantity:
            if type(default_dict[k]) == un.quantity.Quantity:
                kwarg_dict[k] = kwarg_dict[k] * default_dict[k].unit
    return kwarg_dict

def get_band_info(bandname: str):
    assert(bandname in bands), f"{bandname} not recognised band name ({list(bands)})"
    bandpass = bands[bandname]
    c, b = bandpass.values()
    wmin = c - b/2
    wmax = c + b/2
    
    return [wmin, wmax], c