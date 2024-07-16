"""
Created on Mon Jul 15 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Integrate magnetic field to find proton beam path.
"""
import numpy as np
from BB_Simple import beam, density, biermann_field
from scipy.constants import elementary_charge as e
from scipy.constants import proton_mass as m_p

def maxwell(num = 1, temp = 1e7) -> np.ndarray:
    '''
    Randomly Maxwellian-distributed values

    Args:
        num: number of values to generate
        temp: temperature in eV

    Returns:
        array with num values
    '''
    vx = np.random.normal(size=num)
    vy = np.random.normal(size=num)
    vz = np.random.normal(size=num)
    return np.sqrt(vx*vx + vy*vy + vz*vz)/m_p*temp

class Proton():
    '''
    Proton base class
    '''
    def __init__(self, pos = [0, 0, 10], vel = [0, 0, 1]):
        self.__pos = np.array(pos)
        self.__vel = np.array(vel)
    def pos(self) -> np.ndarray:
        '''
        Proton position

        Returns:
            proton position array
        '''
        return self.__pos
    def vel(self) -> np.ndarray:
        '''
        Proton velocity

        Returns:
            proton velocity array
        '''
        return self.__vel
    def move(self, dt, b_field) -> np.ndarray:
        '''
        Moves proton and updated velocity
        '''
        self.__pos += dt*self.__vel
        self.__vel += dt*e/m_p*np.cross(self.__vel, b_field)

class ProtonBeam():
    '''
    Proton beam
    '''
    RHO0 = 1
    DECAY_LENGTH = 10
    AMP = 10
    SPEC_AMP = 1
    WIDTH = 10
    TIME_INCREMEMT = 1e-2

    def __init__(self, n_protons = 100, temperature = 10):
        self.__protons: list[Proton] = []
        temperature *= 1e6
        for _ in range(n_protons):
            v_z = maxwell(temp = temperature)
            self.__protons.append(Proton(vel = [0, 0, v_z]))
    def protons(self) -> list:
        '''
        Proton array

        Returns:
            array of proton objects
        '''
        return self.__protons
    def propagate(self):
        '''
        Propagate proton beam along
        '''
        for proton in self.__protons:
            magnetic = biermann_field(proton.pos(), self.beam_sh, self.density_distr)
            proton.move(ProtonBeam.TIME_INCREMEMT, magnetic)
    def density_distr(self, xyz):
        '''
        Density distribution

        Args:
            xyz: 3-d position

        Returns:
            density value
        '''
        return density(xyz, rho0 = ProtonBeam.RHO0, decay_length = ProtonBeam.DECAY_LENGTH)
    def beam_sh(self, xyz):
        '''
        Beam shape

        Args:
            xyz: 3-d position
        
        Returns:
            beam intensity value
        '''
        return beam(xyz, amp = ProtonBeam.AMP, spec_amp = ProtonBeam.SPEC_AMP,
                    width = ProtonBeam.WIDTH)
