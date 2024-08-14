"""
Created on Fri Aug 09 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Wrap POLLUX code in python.

"""
import os, sys
import numpy as np
import fmodpy

pollux = fmodpy.fimport("Polx2.f")

pollux.outpt1()
