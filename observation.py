#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:30:49 2026

@author: idavis
"""
from star import Star
from instrument import Instrument


class Observation:
    def __init__(self, star:Star = None, instrument:Instrument = None, source_dim:int = 513):
        if star is not None:
            assert(star.dim == source_dim), f"Star's dim property ({star.dim})should match the source_dim quantity ({source_dim})"
            
        self.star = star
        self.instrument = instrument
        return
    
    
    def add_star(self):
        return
    
    
    def add_instrument(self):
        return
    
    
    