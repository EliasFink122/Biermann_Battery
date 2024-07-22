"""
Created on Tue Jul 16 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Integrate magnetic field to find proton beam path.

Methods:
    maxwell:
        random speed values according to M-B distribution

Classes:
    Proton:
        base class for proton properties and movement
    ProtonBeam:
        collection of protons that can be fired at a target through a specified magnetic field
"""
from multiprocessing import Pool
import numpy as np
from scipy.constants import elementary_charge as e, proton_mass as m_p
MODE = "realistic"
if MODE == "simple":
    from BB_Simple import beam, density, biermann_field
elif MODE == "realistic":
    from BB_Realistic import beam, density, biermann_field

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
    def move(self, dt, b_field, e_field = [0, 0, 0]):
        '''
        Moves proton and updated velocity
        '''
        self.__pos = self.__pos + dt*self.__vel
        self.__vel = self.__vel + dt*e/m_p*(np.array(e_field) + np.cross(self.__vel, b_field))

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
        density_distr:
            density distribution function
        beam_sh:
            beam shape function
        propagate (SCP):
            propagate whole proton beam by one time step
        send_beam (SCP):
            shoot all protons at the same time and increment all at once each time step
        propagate_one (MCP):
            propagate only one proton by one time step
        shoot_at_target (MCP):
            propagate one proton repeatedly until it hits the target and record position
        send_beam_mp (MCP):
            shoot protons one at a time and plot final positions
    '''
    RHO0 = 1000 # base density
    DECAY_LENGTH = 0.2 # density decay length scale
    AMP = 0.1 # beam amplitude
    WIDTH = 0.1 # beam width
    TIME_INCREMEMT = 1e-11 # simulation time step
    E_FIELD = [0, 0, -10] # electric field to keep protons from turning around

    # Simple mode
    SPEC_AMP = 0.05 # speckle amplitude

    # Realistic mode
    MOD_AMP = 2 # modulation amplitude
    MOD_FREQ = 10 # modulation frequency
    NUM = 50 # physics resolution

    if MODE == "realistic":
        density_distr_real = density(RHO0, DECAY_LENGTH, NUM)
        beam_sh_real = beam(AMP, WIDTH, MOD_AMP, MOD_FREQ, NUM)
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

        with Pool() as pool:
            self.__protons = pool.map(self.create_proton, range(int(n_protons)))
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
    def protons(self):
        '''
        Proton array

        Returns:
            array of proton objects
        '''
        return self.__protons
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
        beam(xyz, amp = ProtonBeam.AMP, spec_amp = ProtonBeam.SPEC_AMP,
                        width = ProtonBeam.WIDTH)

    # Single core processing
    def propagate(self):
        '''
        Propagate proton beam along
        '''
        for proton in self.__protons:
            if MODE == "simple":
                magnetic = biermann_field(proton.pos(), self.beam_sh, self.density_distr)
            elif MODE == "realistic":
                xs = np.linspace(-1.5*ProtonBeam.WIDTH, 1.5*ProtonBeam.WIDTH, ProtonBeam.NUM)
                x_coord = np.argmin(xs - proton.pos()[0])
                y_coord = np.argmin(xs - proton.pos()[1])
                z_coord = np.argmin(xs - proton.pos()[2])
                magnetic = ProtonBeam.biermann[x_coord, y_coord, z_coord]
            proton.move(ProtonBeam.TIME_INCREMEMT, magnetic, ProtonBeam.E_FIELD)
    def send_beam(self):
        '''
        Send beam through magnetic field and record on RCF behind target.

        Args:
            plot: whether to plot

        Returns:
            final positions of protons
        '''
        positions = []
        detected = 0
        while True:
            self.propagate()
            for i, proton in enumerate(self.__protons):
                for_removal = []
                if proton.pos()[2] <= 0:
                    if np.abs(proton.pos()[0]) < 0.3 and np.abs(proton.pos()[1]) < 0.3:
                        detected += 1
                        positions.append(proton.pos()[:2])
                    for_removal.append(i)
                elif proton.vel()[2] >= 0:
                    for_removal.append(i)
                    print("Warning: Proton moving backwards")
            for i in reversed(for_removal):
                self.__protons.pop(i)
            if len(self.__protons) == 0:
                break
        positions = np.array(positions)
        return positions

    # Multi core processing
    def propagate_one(self, proton: Proton):
        '''
        Propagate one proton by one time step
        '''
        if MODE == "simple":
            magnetic = biermann_field(proton.pos(), self.beam_sh, self.density_distr)
        elif MODE == "realistic":
            xs = np.linspace(-1.5*ProtonBeam.WIDTH, 1.5*ProtonBeam.WIDTH, ProtonBeam.NUM)
            x_coord = np.argmin(xs - proton.pos()[0])
            y_coord = np.argmin(xs - proton.pos()[1])
            z_coord = np.argmin(xs - proton.pos()[2])
            magnetic = ProtonBeam.biermann[x_coord, y_coord, z_coord]
        proton.move(ProtonBeam.TIME_INCREMEMT, magnetic, ProtonBeam.E_FIELD)
        return proton
    def shoot_at_target(self, proton: Proton):
        '''
        Shoot one proton at target
        '''
        while True:
            proton = self.propagate_one(proton)
            if proton.pos()[2] <= 0:
                if np.abs(proton.pos()[0]) < 0.3 and np.abs(proton.pos()[1]) < 0.3:
                    return proton.pos()[:2]
                return None
            if proton.vel()[2] >= 0:
                print("Warning: Proton moving backwards")
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
            positions = pool.map(self.shoot_at_target, self.__protons)

        # Remove None values
        for_removal = []
        for i, pos in enumerate(positions):
            if pos is None or None in pos:
                for_removal.append(i)
        for i in reversed(for_removal):
            positions.pop(i)

        positions = np.array(positions)
        return positions

if __name__ == "__main__":
    print("Creating proton beam...")
    sample_beam = ProtonBeam(1e7, 10, 'even')
    print("Shooting proton beam...")
    position_arr = sample_beam.send_beam_mp()
    print("Saving result...")
    np.savetxt("results.txt", position_arr)
    print("Finished!")
