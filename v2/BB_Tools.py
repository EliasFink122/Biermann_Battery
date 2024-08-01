"""
Created on Thu Aug 01 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Tools for simulations.

Methods:
    grad:
        finds gradient vector of scalar field
    maxwell:
        random speed values according to M-B distribution

Classes:
    Proton:
        base class for proton properties and movement
"""
from scipy.constants import elementary_charge as e, proton_mass as m_p
import numpy as np

def grad(arr, width) -> np.ndarray:
    '''
    Gradient of scalar field.

    Args:
        arr: array of beam or density
        width: beam width

    Returns:
        gradient of array
    '''
    spacing = 3*width/len(arr)

    gradient_12d = np.array(np.gradient(arr, spacing))

    gradient = np.zeros((len(gradient_12d[0]), len(gradient_12d[0]), len(gradient_12d[0]), 3))
    for i, row1 in enumerate(gradient):
        for j, row2 in enumerate(row1):
            for k, _ in enumerate(row2):
                gradient[i, j, k] = np.array([gradient_12d[0, i, j, k],
                                                gradient_12d[1, i, j, k],
                                                gradient_12d[2, i, j, k]])
    return gradient

def curl(arr, width) -> np.ndarray:
    '''
    Curl of 3D vector field.
    
    Args:
        arr: array of beam or density
        width: array width

    Returns:
        curl of array
    '''
    arr = np.array(arr)
    vx = arr[:, :, :, 0]
    vy = arr[:, :, :, 1]
    vz = arr[:, :, :, 2]

    curl_arr = np.zeros((len(arr), len(arr[0]), len(arr[0, 0]), 3))
    for i in range(1, len(arr)-1):
        for j in range(1, len(arr[0])-1):
            for k in range(1, len(arr[0, 0])-1):
                cx = (vz[i, j+1, k] - vz[i, j-1, k]) - (vy[i, j, k+1] - vy[i, j, k-1])
                cy = (vx[i, j, k+1] - vx[i, j, k-1]) - (vz[i+1, j, k] - vz[i-1, j, k])
                cz = (vy[i+1, j, k] - vy[i-1, j, k]) - (vx[i, j+1, k] - vx[i, j-1, k])
                curl_arr[i, j, k] = [cx*len(arr), cy*len(arr[0]), cz*len(arr[0, 0])]/width
    return curl_arr

def integrate_dx(arr, width) -> np.ndarray:
    '''
    Integrate field dx.

    Args:
        arr: array of beam or density
        width: array width

    Returns:
        spatial integral of array
    '''
    arr = np.array(arr)
    dx = width/len(arr)
    integral = np.zeros((len(arr), len(arr[0]), len(arr[0, 0]), 3))
    for i, row in enumerate(row):
        for j in range(i):
            integral[i] += arr[j]*dx
    return integral

def maxwell(temp = 1e7):
    '''
    Randomly Maxwellian-distributed values

    Args:
        temp: temperature in eV

    Returns:
        array with num values
    '''
    vx = np.random.normal()
    vy = np.random.normal()
    vz = np.random.normal()
    return np.sqrt((vx*vx + vy*vy + vz*vz)*(temp*e/m_p))

class Proton():
    '''
    Proton base class

    Methods:
        pos:
            get hidden position attribute
        vel:
            get hidden velocity attribute
        move:
            update position according to velocity and velocity according to force
    '''
    def __init__(self, pos = [0., 0., 1.], vel = [0., 0., -1.]):
        '''
        Args:
            pos: initial proton position (by default 1 m above target surface)
            vel: initial proton velocity (by default 1 m/s towards target surface)
        '''
        self.__pos = np.array(pos)
        self.__vel = np.array(vel)
    def pos(self):
        '''
        Proton position

        Returns:
            proton position array
        '''
        return self.__pos
    def vel(self):
        '''
        Proton velocity

        Returns:
            proton velocity array
        '''
        return self.__vel
    def move(self, dt, b_field, e_field = [0, 0, 0]) -> np.ndarray:
        '''
        Moves proton and updated velocity

        Args:
            dt: time increment
            b_field: magnetic field
            e_field: electric field
            rc: whether to engage relativistic correction
        '''
        self.__pos = self.__pos + dt*self.__vel
        self.__vel = self.__vel + dt*e/m_p*(np.array(e_field) + np.cross(self.__vel, b_field))
