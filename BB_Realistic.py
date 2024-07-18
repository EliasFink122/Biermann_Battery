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
    biermann_field:
        finds magnetic field given density and beam fields
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def beam(amp, width, mod_amp, mod_freq, num) -> np.ndarray:
    '''
    Adds modulation to the beam shape.
    Change this for more control about the initial beam shape.

    Args:
        mod_amp: amplitude of modulation
        mod_freq: frequency of modulation
        num: resolution

    Returns:
        value of modulation
    '''
    xs = np.linspace(-width*1.5, width*1.5, num)
    xys = np.zeros((len(xs), len(xs), 2))
    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            xys[i, j] = [x, y]
    phase = np.random.rand(num, num)*2*np.pi
    modulation = np.exp(mod_amp * np.sin(mod_freq*np.linalg.norm(xys, axis = 2))**2 * np.exp(1j*phase))
    ideal_beam = amp*np.exp(-((np.linalg.norm(xys, axis = 2)**2)/(2*width**2))**5)
    return np.abs(ideal_beam * modulation)

def density(rho0, decay_length, num) -> np.ndarray:
    '''
    Density decay function.

    Args:
        rho0: maximum density at surface
        decay_length: length scale over which density decays by factor of 1/e
        num: resolution

    Returns:
        density at z location in kg/m^3
    '''
    zs = np.linspace(0, 5*decay_length, num)
    density_arr = np.zeros((len(zs)))
    for i, z_pos in enumerate(zs):
        density_arr[i] = rho0 * np.exp(-z_pos/decay_length)
    return density_arr

def grad(arr, width) -> np.array:
    '''
    Gradient of function.

    Args:
        arr: array of beam or density
        width: beam width

    Returns:
        gradient of array
    '''
    spacing = 3*width/len(arr)

    gradient_12d = np.array(np.gradient(arr, spacing))

    if len(np.shape(arr)) == 1:
        gradient = np.zeros((len(gradient_12d), len(gradient_12d), len(gradient_12d), 3))
        for i, row1 in enumerate(gradient):
            for j, row2 in enumerate(row1):
                for k, _ in enumerate(row2):
                    gradient[i, j, k] = np.array([0, 0, gradient_12d[k]])
    elif len(np.shape(arr)) == 2:
        gradient = np.zeros((len(gradient_12d[0]), len(gradient_12d[0]), len(gradient_12d[0]), 3))
        for i, row1 in enumerate(gradient):
            for j, row2 in enumerate(row1):
                for k, _ in enumerate(row2):
                    gradient[i, j, k] = np.array([gradient_12d[0, i, j], gradient_12d[1, i, j], 0])

    return gradient

def biermann_field(beam_sh, density_distr, width) -> np.array:
    '''
    Determines magnetic field due to Biermann battery.

    Args:
        xyz: 3-d position
        temp_func: function of temperature distribution
        density_func: function of density distribution
        width: beam width

    Returns:
        magnetic field
    '''
    grad_beam = grad(beam_sh, width)
    grad_density = grad(density_distr, width)
    grad_temp = grad_beam
    magnetic_field = np.cross(grad_temp, grad_density, axis = 3)
    return magnetic_field

if __name__ == "__main__":
    # Parameters
    RHO0 = 10
    DECAY_LENGTH = 0.1
    AMP = 10
    MOD_AMP = 0.5
    MOD_FREQ = 1
    WIDTH = 10
    NUM = 20

    densities = density(RHO0, DECAY_LENGTH, NUM)
    mod_beam = beam(AMP, WIDTH, MOD_AMP, MOD_FREQ, NUM)

    bf = biermann_field(mod_beam, densities, WIDTH)

    x_arr = np.linspace(-1.5*WIDTH, 1.5*WIDTH, NUM)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.meshgrid(x_arr, x_arr, x_arr)
    ax.quiver(x, y, z, bf[:, :, :, 1], bf[:, :, :, 0], bf[:, :, :, 2], length=1,
              linewidth = 2, arrow_length_ratio = 0.3)
    x, y = np.meshgrid(x_arr, x_arr)
    ax.plot_surface(x, y, mod_beam, cmap = "Oranges")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
