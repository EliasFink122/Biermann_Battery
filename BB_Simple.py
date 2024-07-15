"""
Created on Mon Jul 15 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Simple simulation of a Biermann battery.

Methods:
    beam:
        defines beam shape
    density:
        defines density profile of plasma
"""
import numpy as np
import matplotlib.pyplot as plt

def beam(xy, amp, spec_amp, width) -> float:
    '''
    Beam intensity function based on super Gaussian.
    
    Args:
        xy: 2-d position in beam in mm
        amp: amplitude of beam in J
        spec_amp: amplitude of specles in J
        width: standard deviation of beam Gaussian in mm

    Returns:
        beam intensity at xy location in J
    '''
    xy = np.array(xy)
    # Base beam
    base_beam = amp * np.exp(-(np.linalg.norm(xy)**2/(2*width**2))**5)

    # Specs
    spec_loc = 1/2 # fraction of width
    spec_size = 1/5 # fraction of width
    spec1_pos = np.array([-width*spec_loc, 0])
    spec2_pos = np.array([width*spec_loc, 0])
    spec1 = spec_amp * np.exp(-np.linalg.norm(xy - spec1_pos)**2/(2*(width*spec_size)**2)**5)
    spec2 = spec_amp * np.exp(-np.linalg.norm(xy - spec2_pos)**2/(2*(width*spec_size)**2)**5)

    return base_beam + spec1 + spec2

def density(z, rho0, decay_length) -> float:
    '''
    Density decay function.

    Args:
        z: distance away from target surface in mm
        rho0: maximum density at surface
        decay_length: length scale over which density decays by factor of 1/e

    Returns:
        density at z location in kg/m^3
    '''
    if z <= 0:
        return rho0
    return rho0 * np.exp(-z/decay_length)

def grad(func, coord, *args) -> np.array:
    '''
    Gradient of function.

    Args:
        func: function of interest
        coord: position of gradient
        *args: other arguments of function

    Returns:
        gradient at position
    '''
    res = 1e-3
    gradient = []
    if isinstance(coord, (int, float)):
        coord = [coord]
    for i, _ in enumerate(coord):
        arr1, arr2 = np.array(coord), np.array(coord)
        arr1[i] -= res
        arr2[i] += res

        diff = (func(arr2, *args) - func(arr1, *args))/(2*res)
        gradient.append(diff)
    return gradient

def biermann_field(xyz, beam_shape, density_func) -> np.array:
    '''
    Determines magnetic field due to Biermann battery.

    Args:
        xyz: 3-d position
        temp_func: function of temperature distribution
        density_func: function of density distribution

    Returns:
        magnetic field
    '''
    temperature = beam_shape
    grad_density = np.array([0, 0] + grad(density_func, xyz[2]))
    grad_temp = np.array(grad(temperature, xyz[:2]) + [0])
    magnetic_field = np.cross(grad_density, grad_temp)

    return magnetic_field

if __name__ == "__main__":
    # Parameters
    RHO0 = 1
    DECAY_LENGTH = 1
    AMP = 10
    SPEC_AMP = 1
    WIDTH = 10


    density_distr = lambda z: density(z, rho0 = RHO0, decay_length = DECAY_LENGTH)
    beam_sh = lambda xy: beam(xy, amp = AMP, spec_amp = SPEC_AMP, width = WIDTH)
