"""
Created on Mon Jul 15 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Realistic simulation of a Biermann battery.

Methods:
    beam:
        defines beam shape
    density:
        defines density profile of plasma
    grad:
        finds gradient vector of scalar field
    magnetic_field:
        finds magnetic field given density and beam fields
    electric_field:
        finds magnetic field given density and beam fields
"""
import numpy as np
from scipy.constants import Boltzmann as kb, elementary_charge as e
from scipy.special import erf
from BB_Tools import grad, div, curl, integrate_dx

def beam(amp, width, mod_amp, mod_freq, num):
    '''
    Adds modulation to the beam shape.
    Change this for more control about the initial beam shape.

    Args:
        mod_amp: amplitude of modulation
        width: beam width
        mod_freq: frequency of modulation
        num: resolution

    Returns:
        value of modulation
    '''
    xs = np.linspace(-width*1.5, width*1.5, num)
    xyz = np.zeros((len(xs), len(xs), len(xs), 3))
    phase = np.random.rand(num, num)*2*np.pi
    phase_xyz = np.zeros(np.shape(xyz)[:3])
    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            for k, z in enumerate(xs):
                xyz[i, j, k] = [x, y, z]
                phase_xyz[i, j, k] = phase[i, j]
    exp = mod_amp * np.sin(mod_freq*(np.sqrt(xyz[:, :, :, 0]**2 + xyz[:, :, :, 1]**2)))**2
    modulation = np.exp(exp) * np.exp(1j*phase_xyz)
    ideal_beam = amp*np.exp(-((xyz[:, :, :, 0]**2 + xyz[:, :, :, 1]**2)/(2*width**2))**5)
    return np.abs(ideal_beam * modulation)

def density(time, rho0, decay_length, num, temp = None, d = 1) -> np.ndarray:
    '''
    Density distribution.

    Args:
        time: time in simulation
        rho0: maximum density at surface
        decay_length: length scale over which density decays by factor of 1/e
        num: resolution
        temp: temperature distribution
        d: diffusion coefficient

    Returns:
        density at z location in kg/m^3
    '''
    zs = np.linspace(-5*decay_length, 5*decay_length, num)
    density_arr = np.zeros((len(zs)))
    if time == 0:
        for i, z_pos in enumerate(zs):
            if z_pos >= 0:
                density_arr[i] = rho0
            else:
                density_arr[i] = 0
    else:
        density_arr = rho0*(erf(zs/(2*np.sqrt(d*time)))+1)
    density_xyz = np.zeros((num, num, num))
    for i, _ in enumerate(zs):
        for j, _ in enumerate(zs):
            for k, _ in enumerate(zs):
                density_xyz[i, j, k] = density_arr[k]
    if temp is None:
        return density_xyz
    return density_xyz*np.sqrt(temp/np.max(temp))

def temperature(time, beam_sh, c_tilde = 1, temp_init = None,
                alpha = 1, width = None, dens = None) -> np.ndarray:
    '''
    Temperature distribution.

    Args:
        time: time in simulation
        beam_sh: laser beam
        c_tilde: heat capacity per area
        temp_init: previous temperature distribution
        alpha: heat transmission coefficient
        width: beam width
        dens: density distribution

    Returns:
        temperature distribution in K
    '''
    if temp_init is None:
        temp = time*beam_sh/c_tilde
    elif dens is None:
        temp = time*(beam_sh/(c_tilde) + alpha*div(grad(temp_init, width), width))
    else:
        temp = time*(beam_sh/(c_tilde*dens/np.max(dens)) + alpha*div(grad(temp_init, width), width))
    return temp

def magnetic_field(time, temp_distr, density_distr, width) -> np.ndarray:
    '''
    Determines magnetic field due to Biermann battery.

    Args:
        time: time in simulation
        temp_func: function of temperature distribution
        density_func: function of density distribution
        width: beam width

    Returns:
        magnetic field
    '''
    grad_temp = grad(temp_distr, width)
    grad_density = grad(density_distr, width)
    magnetic = time*kb/(e*density_distr)*np.cross(grad_temp, grad_density, axis = 3)
    return magnetic

def electric_field(temp_distr: np.ndarray, density_distr: np.ndarray, width: float) -> np.ndarray:
    '''
    Determines electric field due to Biermann battery.

    Args:
        time: time in simulation
        temp_distr: function of temperature distribution
        density_distr: function of density distribution
        width: beam width

    Returns:
        electric field
    '''
    grad_temp = grad(temp_distr, width)
    grad_density = grad(density_distr, width)
    lapl_electric = -kb/(e*density_distr)*curl(np.cross(grad_temp, grad_density, axis = 3), width)
    electric = integrate_dx(integrate_dx(lapl_electric.transpose(2, 0, 1, 3),
                                         width), width).transpose(1, 2, 0, 3)
    return electric
