"""
Created on Mon Jul 15 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Simple simulation of a Biermann battery.

Methods:
    beam:
        defines beam shape
    density:
        defines density profile of plasma
    grad:
        finds gradient vector of scalar field
    biermann_field:
        finds magnetic field given density and beam fields
"""
import numpy as np
from scipy.constants import Boltzmann as k, elementary_charge as e

def beam(xyz, amp, spec_amp, width):
    '''
    Beam intensity function based on super Gaussian.

    Args:
        xy: 2-d position in beam in m
        amp: amplitude of beam in J
        spec_amp: amplitude of specles in J
        width: standard deviation of beam Gaussian in m

    Returns:
        beam intensity at xy location in J
    '''
    xy = np.array(xyz[:2])
    # Base beam
    base_beam = amp * np.exp(-(np.linalg.norm(xy)**2/(2*width**2))**5)

    # Specs
    spec_loc = 1/2 # fraction of width
    spec_size = 1/4 # fraction of width
    spec1_pos = np.array([-width*spec_loc, 0])
    spec2_pos = np.array([width*spec_loc, 0])
    spec1 = spec_amp * np.exp(-(np.linalg.norm(xy - spec1_pos)**2/(2*(width*spec_size)**2))**5)
    spec2 = spec_amp * np.exp(-(np.linalg.norm(xy - spec2_pos)**2/(2*(width*spec_size)**2))**5)

    return base_beam + spec1 + spec2

def density(xyz, rho0, decay_length, beam_func) -> float:
    '''
    Density decay function.

    Args:
        z: distance away from target surface in m
        rho0: maximum density at surface
        decay_length: length scale over which density decays by factor of 1/e

    Returns:
        density at z location in kg/m^3
    '''
    z = xyz[2]
    beam_val = beam_func(xyz)
    if z <= 0:
        return rho0
    return rho0 * np.exp(-z/decay_length)*beam_val

def grad(func, coord):
    '''
    Gradient of function.

    Args:
        func: function of interest
        coord: position of gradient

    Returns:
        gradient at position
    '''
    res = 1e-1
    gradient = [0, 0, 0]
    for l in range(3):
        coord = np.array(coord)
        dx = np.zeros(3)
        dx[l] = res

        diff = (func(coord + dx) - func(coord - dx))/(2*res)
        gradient[l] = diff
    return gradient

def biermann_field(xyz, beam_shape, density_func):
    '''
    Determines magnetic field due to Biermann battery.

    Args:
        xyz: 3-d position
        beam_shape: function of beam intensity distribution
        density_func: function of density distribution

    Returns:
        magnetic field
    '''
    grad_density = np.array(grad(density_func, xyz))
    grad_temp = np.array(grad(beam_shape, xyz))
    magnetic_field = k/(e*density_func(xyz))*np.cross(grad_density, grad_temp)

    return magnetic_field
