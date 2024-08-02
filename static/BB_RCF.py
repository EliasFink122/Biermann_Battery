"""
Created on Tue Jul 16 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Integrate magnetic field to find proton beam path.

Methods:
    maxwell:
        random speed values according to M-B distribution

Classes:
    F:
        electromagnetic field tensor
    Proton:
        base class for proton properties and movement
    ProtonBeam:
        collection of protons that can be fired at a target through a specified magnetic field
"""
from multiprocessing import Pool
import numpy as np
from scipy.constants import elementary_charge as e, proton_mass as m_p, speed_of_light as c
MODE = "simple"
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

class F():
    '''
    Electromagnetic field tensor

    Methods:
        matrix:
            get EM field tensor as matrix
        transform:
            Lorentz boost tensor into different frame
        get_e:
            extract electric field
        get_b:
            extract magnetic field
    '''
    def __init__(self, e_field: list, b_field: list):
        '''
        Args:
            e_field: electric field
            b_field: magnetic field
        '''
        self.__f = np.array([
                    [0, -e_field[0]/c, -e_field[1]/c, -e_field[2]/c],
                    [e_field[0]/c, 0, -b_field[2], b_field[1]],
                    [e_field[1]/c, b_field[2], 0, -b_field[0]],
                    [e_field[2]/c, -b_field[1], b_field[0], 0]
                    ])
    def matrix(self) -> np.ndarray:
        '''
        Get electromagnetic field tensor matrix

        Returns:
        electromagnetic field tensor
        '''
        return self.__f
    def transform(self, vel: np.ndarray):
        '''
        Transform EM field tensor into different frame.

        Args:
            boost: Lorentz boost
        '''
        gm = 1/np.sqrt(1 - np.dot(vel, vel)/c**2)
        v_x, v_y, v_z, v = (vel[0], vel[1], vel[2], np.linalg.norm(vel))
        lorentz = np.array([
            [gm, -gm/c*v_x, -gm/c*v_y, -gm/c*v_z],
            [-gm/c*v_x, 1+(gm-1)*v_x**2/v**2, (gm-1)*v_x*v_y/v**2, (gm-1)*v_x*v_z/v**2],
            [-gm/c*v_y, (gm-1)*v_x*v_y/v**2, 1+(gm-1)*v_y**2/v**2, (gm-1)*v_y*v_z/v**2],
            [-gm/c*v_z, (gm-1)*v_x*v_z/v**2, (gm-1)*v_y*v_z/v**2, 1+(gm-1)*v_z**2/v**2]
            ])
        self.__f = lorentz * self.__f * lorentz
    def get_e(self) -> np.ndarray:
        '''
        Get electric field.

        Returns:
            electric field
        '''
        return -np.array([self.__f[0, 1], self.__f[0, 2], self.__f[0, 3]])*c
    def get_b(self) -> np.ndarray:
        '''
        Get magnetic field.

        Returns:
            magnetic field
        '''
        return np.array([self.__f[3, 2], self.__f[1, 3], self.__f[2, 1]])

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
    def move(self, dt, b_field, e_field = [0, 0, 0], rc = False) -> np.ndarray:
        '''
        Moves proton and updated velocity

        Args:
            dt: time increment
            b_field: magnetic field
            e_field: electric field
            rc: whether to engage relativistic correction
        '''
        self.__pos = self.__pos + dt*self.__vel
        if rc:
            f = F(e_field, b_field)
            f.transform(self.__vel)
            e_field = f.get_e()
            self.__vel = self.__vel + dt*e/m_p*np.array(e_field)
        else:
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
    RHO0 = 0.1 # base density
    DECAY_LENGTH = 0.5 # density decay length scale
    AMP = 0.1 # beam amplitude
    WIDTH = 0.1 # beam width
    TIME_INCREMEMT = 1e-11 # simulation time step
    E_FIELD = [0, 0, 0] # electric field to keep protons from turning around

    # Simple mode
    SPEC_AMP = 0.05 # speckle amplitude

    # Realistic mode
    MOD_AMP = 1 # modulation amplitude
    MOD_FREQ = 10 # modulation frequency
    NUM = 50 # physics resolution

    if MODE == "realistic":
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
        return density(xyz, decay_length = ProtonBeam.DECAY_LENGTH,
                       beam_func = self.beam_sh)
    def beam_sh(self, xyz) -> float:
        '''
        Beam shape

        Args:
            xyz: 3-d position

        Returns:
            beam intensity value
        '''
        return beam(xyz, amp = ProtonBeam.AMP, spec_amp = ProtonBeam.SPEC_AMP,
                        width = ProtonBeam.WIDTH)

    # Single core processing
    def propagate(self, rc = False):
        '''
        Propagate proton beam along

        Args:
            rc: whether to engage relativistic correction
        '''
        for proton in self.__protons:
            if MODE == "simple":
                magnetic = ProtonBeam.RHO0*biermann_field(proton.pos(), self.beam_sh, self.density_distr)
            elif MODE == "realistic":
                xs = np.linspace(-1.5*ProtonBeam.WIDTH, 1.5*ProtonBeam.WIDTH, ProtonBeam.NUM)
                x_coord = np.argmin(xs - proton.pos()[0])
                y_coord = np.argmin(xs - proton.pos()[1])
                z_coord = np.argmin(xs - proton.pos()[2])
                magnetic = ProtonBeam.biermann[x_coord, y_coord, z_coord]
            proton.move(ProtonBeam.TIME_INCREMEMT, magnetic, ProtonBeam.E_FIELD, rc = rc)
    def send_beam(self, plot = True, rc = False) -> np.ndarray:
        '''
        Send beam through magnetic field and record on RCF behind target.

        Args:
            plot: whether to plot
            rc: whether to engage relativistic correction

        Returns:
            final positions of protons
        '''
        positions = []
        detected = 0
        time = 0
        while True:
            time += ProtonBeam.TIME_INCREMEMT
            self.propagate(rc = rc)
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
    def propagate_one(self, proton: Proton, rc = False) -> Proton:
        '''
        Propagate one proton by one time step

        Args:
            proton: proton object to propagate
            time: time in simulation
            rc: whether to engage relativistic correction

        Returns:
            updated proton object
        '''
        if MODE == "simple":
            magnetic = ProtonBeam.RHO0*biermann_field(proton.pos(), self.beam_sh, self.density_distr)
        elif MODE == "realistic":
            xs = np.linspace(-1.5*ProtonBeam.WIDTH, 1.5*ProtonBeam.WIDTH, ProtonBeam.NUM)
            x_coord = np.argmin(xs - proton.pos()[0])
            y_coord = np.argmin(xs - proton.pos()[1])
            z_coord = np.argmin(xs - proton.pos()[2])
            magnetic = ProtonBeam.biermann[x_coord, y_coord, z_coord]
        proton.move(ProtonBeam.TIME_INCREMEMT, magnetic, ProtonBeam.E_FIELD, rc = rc)
        return proton
    def shoot_at_target(self, proton: Proton, rc = False) -> list[float]:
        '''
        Shoot one proton at target

        Args:
            proton: proton object to shoot
            rc: whether to engage relativistic correction

        Returns:
            final position of proton
        '''
        time = 0
        while True:
            time += ProtonBeam.TIME_INCREMEMT
            proton = self.propagate_one(proton, rc = rc)
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
    sample_beam = ProtonBeam(1e5, 10, 'even')
    print("Shooting proton beam...")
    position_arr = sample_beam.send_beam_mp()
    print("Saving result...")
    np.savetxt("results.txt", position_arr)
    print("Finished!")
