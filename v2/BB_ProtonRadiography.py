"""
Created on Thu Aug 01 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Simulate proton radiography.

Classes:
    ProtonBeam:
        collection of protons that can be fired at a target through a specified magnetic field
"""
from multiprocessing import Pool
import numpy as np
from BB_Tools import maxwell, Proton
from BB_Realistic import beam, density, biermann_field

class ProtonBeam():
    '''
    Proton beam

    Methods:
        create_proton:
            create proton with certain speed (thermal MB distribution)
        protons:
            get hidden protons array attribute
        plot_spectrum:
            plots proton speed spectrum and beam distribution at target
        propagate_one (MCP):
            propagate only one proton by one time step
        shoot_at_target (MCP):
            propagate one proton repeatedly until it hits the target and record position
        send_beam_mp (MCP):
            shoot protons one at a time and plot final positions
    '''
    RHO0 = 10 # base density
    DECAY_LENGTH = 0.5 # density decay length scale
    AMP = 0 # beam amplitude
    WIDTH = 0.1 # beam width
    TIME_INCREMEMT = 1e-11 # simulation time step
    E_FIELD = [0, 0, 0] # electric field to keep protons from turning around

    MOD_AMP = 1 # modulation amplitude
    MOD_FREQ = 10 # modulation frequency
    NUM = 50 # physics resolution

    beam_sh_real = beam(AMP, WIDTH, MOD_AMP, MOD_FREQ, NUM)
    density_distr_real = density(RHO0, DECAY_LENGTH, NUM, beam_sh_real)
    biermann = biermann_field(beam_sh_real, density_distr_real, WIDTH)

    def __init__(self, n_protons = 100, temperature = 10, distribution = 'even'):
        '''
        Args:
            n_protons: number of protons
            temperature: proton temperature in MeV
            distribution: even, central or edge for different proton distributions
        '''
        self.__temperature = temperature*1e6
        self.__distribution = distribution
        self.__n_protons = n_protons
    def create_proton(self, i):
        '''
        Create one proton only

        Returns:
            proton object
        '''
        origin = [0, 0, 1]
        speed = maxwell(temp = self.__temperature)
        if self.__distribution == 'even':
            spread = np.sqrt(np.random.rand())*np.pi/20
        elif self.__distribution == 'central':
            spread = np.random.rand()*np.pi/20
        elif self.__distribution == 'edge':
            spread = np.sqrt(np.sqrt(np.random.rand()))*np.pi/20
        traj = np.random.rand()*2*np.pi
        vel = [speed*np.sin(spread)*np.cos(traj), speed*np.sin(spread)*np.sin(traj),
                -speed*np.cos(spread)]
        return Proton(pos = origin, vel = vel)
    def propagate_one(self, proton: Proton, time: float) -> Proton:
        '''
        Propagate one proton by one time step

        Args:
            proton: proton object to propagate
            time: time in simulation
            rc: whether to engage relativistic correction

        Returns:
            updated proton object
        '''
        xs = np.linspace(-1.5*ProtonBeam.WIDTH, 1.5*ProtonBeam.WIDTH, ProtonBeam.NUM)
        x_coord = np.argmin(xs - proton.pos()[0])
        y_coord = np.argmin(xs - proton.pos()[1])
        z_coord = np.argmin(xs - proton.pos()[2])
        magnetic = time*ProtonBeam.biermann[x_coord, y_coord, z_coord]
        proton.move(ProtonBeam.TIME_INCREMEMT, magnetic, ProtonBeam.E_FIELD)
        return proton
    def shoot_at_target(self, i) -> list[float]:
        '''
        Shoot one proton at target

        Args:
            proton: proton object to shoot
            rc: whether to engage relativistic correction

        Returns:
            final position of proton
        '''
        time = 0
        proton = self.create_proton(i)
        while True:
            time += ProtonBeam.TIME_INCREMEMT
            proton = self.propagate_one(proton, time = time)
            if proton.pos()[2] <= 0:
                #print(f"Proton detected at {proton.pos()[:2]}.")
                return proton.pos()[:2]

            moving_backwards = proton.vel()[2] >= 0
            out_of_screen = not (np.abs(proton.pos()[0]) < 1.5 and np.abs(proton.pos()[1]) < 1.5)
            if moving_backwards or out_of_screen:
                print("Warning: Proton out of scope")
                return None
    def send_beam_mp(self):
        '''
        Send beam through magnetic field and record on RCF behind target using multiprocessing.

        Args:
            plot: whether to plot

        Returns:
            final positions of protons
        '''
        with Pool() as pool:
            positions = pool.map(self.shoot_at_target, range(self.__n_protons))

        # Remove None values
        for_removal = []
        for i, pos in enumerate(positions):
            if pos is None or None in pos:
                for_removal.append(i)
        for i in reversed(for_removal):
            positions.pop(i)

        positions = np.array(positions)
        return positions
