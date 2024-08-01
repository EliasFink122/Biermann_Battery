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
from BB_Fields import beam, density, temperature, magnetic_field, electric_field

class ProtonBeam():
    '''
    Proton beam

    Methods:
        calculate_fields:
            determine fields for all timesteps
        create_proton:
            create proton with certain speed (thermal MB distribution)
        propagate_one:
            propagate only one proton by one time step
        shoot_at_target:
            propagate one proton repeatedly until it hits the target and record position
        send_beam:
            shoot protons one at a time and plot final positions
    '''
    RHO0 = 10 # base density
    DECAY_LENGTH = 0.5 # density decay length scale
    AMP = 0 # beam amplitude
    WIDTH = 0.1 # beam width
    TIME_INCREMEMT = 1e-10 # simulation time step

    MOD_AMP = 1 # modulation amplitude
    MOD_FREQ = 10 # modulation frequency
    NUM = 50 # physics resolution

    def __init__(self, n_protons = 100, temp = 10, distribution = 'even'):
        '''
        Args:
            n_protons: number of protons
            temperature: proton temperature in MeV
            distribution: even, central or edge for different proton distributions
        '''
        self.__temperature = temp*1e6
        self.__distribution = distribution
        self.__n_protons = int(n_protons)
        self.__fname = f'results_1e{np.log10(n_protons):.1f}_{distribution}.txt'

        self.__b_fields = []
        self.__e_fields = []

        np.savetxt(self.__fname, np.array([]))
    def calculate_fields(self):
        '''
        Calculate fields for all time steps
        '''
        print("Calculating fields...")

        beam_sh = beam(ProtonBeam.AMP, ProtonBeam.WIDTH, ProtonBeam.MOD_AMP,
                       ProtonBeam.MOD_FREQ, ProtonBeam.NUM)

        density_distr = density(0, ProtonBeam.RHO0, ProtonBeam.DECAY_LENGTH,
                                     ProtonBeam.NUM, d = 0.0001) + 1
        temp_distr = temperature(0, beam_sh, width = ProtonBeam.WIDTH) + 1


        for n in range(100):
            print(n)
            time = n*ProtonBeam.TIME_INCREMEMT
            density_distr = density(time, ProtonBeam.RHO0, ProtonBeam.DECAY_LENGTH,
                                     ProtonBeam.NUM, temp_distr) + 1
            temp_distr = temperature(ProtonBeam.TIME_INCREMEMT, beam_sh, temp_init = temp_distr,
                                     width = ProtonBeam.WIDTH, dens = density_distr) + 1
            magnetic_arr = magnetic_field(time, temp_distr, density_distr,
                                      ProtonBeam.WIDTH)
            electric_arr = electric_field(temp_distr, density_distr,
                                        ProtonBeam.WIDTH)
            self.__b_fields.append(magnetic_arr)
            self.__e_fields.append(electric_arr)
    def create_proton(self):
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
        # Position
        xs = np.linspace(-1.5*ProtonBeam.WIDTH, 1.5*ProtonBeam.WIDTH, ProtonBeam.NUM)
        x_coord = np.argmin(xs - proton.pos()[0])
        y_coord = np.argmin(xs - proton.pos()[1])
        z_coord = np.argmin(xs - proton.pos()[2])

        # Motion
        magnetic_arr = self.__b_fields[time]
        magnetic = magnetic_arr[x_coord, y_coord, z_coord]
        electric_arr = self.__e_fields[time]
        electric = electric_arr[x_coord, y_coord, z_coord]
        proton.move(ProtonBeam.TIME_INCREMEMT, magnetic, electric)

        return proton
    def shoot_at_target(self, i):
        '''
        Shoot one proton at target

        Args:
            proton: proton object to shoot
            rc: whether to engage relativistic correction

        Returns:
            final position of proton
        '''
        print(i)
        with open(self.__fname, 'w', encoding='utf-8') as f:
            time = 0
            proton = self.create_proton()
            while True:
                proton = self.propagate_one(proton, time = time)
                if proton.pos()[2] <= 0:
                    np.savetxt(f, proton.pos()[:2])
                    break

                moving_backwards = proton.vel()[2] >= 0
                out_of_screen = np.abs(proton.pos()[0]) > 1.5 or np.abs(proton.pos()[1]) > 1.5
                timeout = time >= len(self.__b_fields)
                if moving_backwards or out_of_screen or timeout:
                    print(f"Warning: Proton {i} out of scope")
                    break
                time += 1
    def send_beam(self):
        '''
        Send beam through magnetic field and record on RCF behind target using multiprocessing.

        Args:
            plot: whether to plot

        Returns:
            final positions of protons
        '''
        print("Sending beam...")
        with Pool() as pool:
            pool.map(self.shoot_at_target, range(self.__n_protons))

if __name__ == "__main__":
    proton_beam = ProtonBeam(1e2, 10, 'even')
    proton_beam.calculate_fields()
    proton_beam.send_beam()
