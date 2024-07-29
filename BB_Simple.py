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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.constants import Boltzmann as k, elementary_charge as e

def beam(xyz, amp, spec_amp, width) -> float:
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

def density(xyz, rho0, decay_length) -> float:
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
    if z <= 0:
        return rho0
    return rho0 * np.exp(-z/decay_length)

def grad(func, coord) -> np.array:
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

def biermann_field(xyz, beam_shape, density_func) -> np.array:
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

if __name__ == "__main__":
    # Parameters
    RHO0 = 10
    DECAY_LENGTH = 0.1
    AMP = 10
    SPEC_AMP = 1
    WIDTH = 0.1

    density_distr = lambda xyz: density(xyz, rho0 = RHO0, decay_length = DECAY_LENGTH)
    beam_sh = lambda xyz: beam(xyz, amp = AMP, spec_amp = SPEC_AMP, width = WIDTH)

    NUM = 20
    NUMZ = 10
    xs = np.linspace(-1.5*WIDTH, 1.5*WIDTH, NUM)
    zs = np.linspace(0, 1.5*DECAY_LENGTH, NUMZ)

    bf = np.zeros((NUM, NUM, NUMZ, 3))
    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            for k, z in enumerate(zs):
                bf[i, j, k] = biermann_field(xyz = [x, y, z], beam_shape = beam_sh,
                                    density_func = density_distr)

    NUMB = 50
    beam_xs = np.linspace(-1.5*WIDTH, 1.5*WIDTH, NUMB)
    beam_arr = np.zeros((len(beam_xs), len(beam_xs)))
    for i, x in enumerate(beam_xs):
        for j, y in enumerate(beam_xs):
            beam_arr[i, j] = beam_sh([x, y, 0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.meshgrid(xs, xs, zs)
    ax.quiver(x, y, z, bf[:, :, :, 1], bf[:, :, :, 0], bf[:, :, :, 2], length=0.00001,
              linewidth = 2, arrow_length_ratio = 0.3)
    x, y = np.meshgrid(beam_xs, beam_xs)
    ax.plot_surface(x, y, beam_arr, cmap = "Oranges")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
