import numpy as np
from numpy.random import default_rng
from scipy.constants import gravitational_constant as G
import matplotlib.pyplot as plt
from time import perf_counter


EPSILON = 0.001
N_YEARS = 1
N_PARTICLES = 8


class Particles:
    def __init__(self, pos_in, vel_in, mass_in, names_in=None):
        self.pos = pos_in
        self.pos_prev = pos_in.copy()
        self.vel = vel_in
        self.acc = np.zeros_like(self.pos)
        self.mass = mass_in
        self.names = names_in

        self.size = len(mass_in)


# class Particle:
    # def __init__(self, x, y, z, vx, vy, vz, mass, name=None):
        # self.pos = np.array([x, y, z], dtype=np.float64)
        # self.pos_prev = np.array([x, y, z], dtype=np.float64)
        # self.vel = np.array([vx, vy, vz], dtype=np.float64)
        # self.acc = np.array([0, 0, 0], dtype=np.float64)

        # self.mass = mass
        # self.name = name

        # self.tracker = []


def random(num, a=0., b=1.):
    rng = default_rng()
    return rng.random(num)*(b-a) + a


def generate_random_ics(num):
    r = random(num, 0.4, 20)
    theta = random(num, 0., np.pi)
    v_mag = 1./np.sqrt(r)

    pos = np.zeros((num, 2))
    vel = np.zeros((num, 2))

    pos[:,0] = r*np.sin(theta)
    pos[:,1] = r*np.cos(theta)

    vel[:,0] = -v_mag*np.cos(theta)
    vel[:,1] =  v_mag*np.sin(theta)

    mass = random(num, 1/6000000, 1/1000)

    pos[0,:] = 0.
    vel[0,:] = 0.
    mass[0] = 1.

    return Particles(pos, vel, mass)


def create_solar_system():
    # Solar system data from https://physics.stackexchange.com/questions/441608/solar-system-position-and-velocity-data
    names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Neptune", "Uranus"]
    m = np.array(([1],[1/6023600],[1/408524],[1/332946.038],[1/3098710],[1/1047.55],[1/3499],[1/22962],[1/19352]))
    r = np.array(([0,0],[0.4,0],[0,0.7],[1,0],[0,1.5],[5.2,0],[0,9.5],[19.2,0],[0,30.1]))
    v = np.array(([0,0],[0,-np.sqrt(1/0.4)],[np.sqrt(1/0.7),0],[0,-1],[np.sqrt(1/1.5),0],[0,-np.sqrt(1/5.2)],[np.sqrt(1/9.5),0],[0,-np.sqrt(1/19.2)],[np.sqrt(1/30.1),0]))

    return [Particle(r[i, 0], r[i,1], 0, v[i,0], v[i,1], 0, m[i,0], name=names[i]) for i in range(9)]


def calc_acceleration(particles):
    for i in range(particles.size):
        r = particles.pos - particles.pos[i]
        acc = r.T * particles.mass / (r[:, 0]**2 + r[:, 1]**2 + EPSILON**2)**(1.5)
        particles.acc[i] = np.sum(acc)

# def integrate_vel_verlet(particles):
    # for particle in particles:
        # particle.tracker += [particle.pos.copy()]
        # particle.pos += particle.vel * dt + 0.5 * particle.acc * dt**2
        # particle.vel += particle.acc * dt


def main():
    # particles = create_solar_system()
    particles = generate_random_ics(N_PARTICLES)

    dt = 0.01
    start = perf_counter()

    t = 0.0
    # total_time = N_YEARS*2*np.pi
    total_time = 100*dt

## STANDARD VERLET METHOD WITH INITIAL SETUP
    calc_acceleration(particles)
    particles.pos_prev[:] = particles.pos[:]
    particles.pos += particles.vel*dt + 0.5*particles.acc*dt**2
    # for particle in particles:
        # particle.tracker += [particle.pos.copy()]
        # particle.pos_prev[:] = particle.pos[:]
        # particle.pos += particle.vel * dt + 0.5 * particle.acc * dt**2
    t += dt

    while t < total_time:
        calc_acceleration(particles)
        temp_pos = particles.pos.copy()
        particles.pos = 2.0 * particles.pos - particles.pos_prev + particles.acc * dt**2
        particles.pos_prev[:] = temp_pos[:]
        # for particle in particles:
            # particle.tracker += [particle.pos.copy()]
            # temp_pos = particle.pos.copy()
            # particle.pos = 2.0 * particle.pos - particle.pos_prev + particle.acc * dt**2
            # particle.pos_prev[:] = temp_pos[:]
        t += dt

## VELOCITY VERLET METHOD:
    # while t < total_time:
        # calc_acceleration(particles)
        # for particle in particles:
            # particle.tracker += [particle.pos.copy()]
            # particle.pos += particle.vel * dt + 0.5 * particle.acc * dt**2
            # particle.vel += particle.acc * dt
        # t += dt

    end = perf_counter()

    print(f"Time to complete: {end-start}")

    # ax = plt.axes(projection='3d')
    # ax = plt.axes()
    # xmin, xmax = 0.0, 0.0
    # ymin, ymax = 0.0, 0.0
    # for p in particles:
        # xdata = np.array(p.tracker)[:,0]
        # ydata = np.array(p.tracker)[:,1]
        # xmin = min(np.min(xdata), xmin)
        # xmax = max(np.max(xdata), xmax)
        # ymin = min(np.min(ydata), ymin)
        # ymax = max(np.max(ydata), ymax)
        # ax.plot(xdata, ydata, label=p.name)

    # xmax = max(abs(xmax), abs(xmin))
    # ymax = max(abs(ymax), abs(ymin))
    # xmin = -xmax
    # ymin = -ymax
    # plt.xlim(xmin, xmax)
    # plt.ylim(ymin, ymax)
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    # main()
    for N_PARTICLES in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        print(N_PARTICLES)
        main()
