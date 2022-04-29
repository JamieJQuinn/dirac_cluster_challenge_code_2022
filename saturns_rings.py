import numpy as np
from numpy.random import default_rng
from scipy.constants import gravitational_constant as G
import matplotlib.pyplot as plt
from time import perf_counter

from numba import njit, prange


PARALLEL=False
EPSILON = 0.01
N_YEARS = 0.1
N_PARTICLES = 100
FP_TYPE = np.float64


def random(num, a=0., b=1.):
    rng = default_rng()
    return rng.random(num, dtype=FP_TYPE)*(b-a) + a


def calc_stable_orbit(r, theta):
    v_mag = 1./np.sqrt(r)

    pos = np.zeros((r.shape[0], 2))
    vel = np.zeros((r.shape[0], 2))

    pos[:,0] = r*np.sin(theta)
    pos[:,1] = r*np.cos(theta)

    vel[:,0] = -v_mag*np.cos(theta)
    vel[:,1] =  v_mag*np.sin(theta)

    return pos, vel


def generate_random_star_system(num):
    # Generate similar system to the solar system
    r = random(num, 0.4, 20)
    mass = random(num, 1/6000000, 1/1000)

    theta = random(num, 0., np.pi)
    pos, vel = calc_stable_orbit(r, theta)

    # Add central star
    pos[0,:] = 0.
    vel[0,:] = 0.
    mass[0] = 1.

    return pos, vel, mass


def create_solar_system():
    # Solar system data from https://physics.stackexchange.com/questions/441608/solar-system-position-and-velocity-data

    names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Neptune", "Uranus"]

    mass = np.array((1,1/6023600,1/408524,1/332946.038,1/3098710,1/1047.55,1/3499,1/22962,1/19352))
    r = np.array((0.1, 0.4, 0.7, 1, 1.5, 5.2, 9.5, 19.2, 30.1))
    pos, vel = calc_stable_orbit(r, random(len(r), 0., np.pi))

    pos[0,:] = 0.
    vel[0,:] = 0.
    mass[0] = 1.

    return pos, vel, mass


@njit(parallel=True)
def calc_acc(acc, pos, mass):
    for i in prange(len(pos)):
        acc[i,:] = 0.0
        for j in range(len(pos)):
            if i == j:
                # Skip self-comparison
                continue
            r = pos[j] - pos[i]
            acc[i,:] += r * mass[j] / (r[0]**2 + r[1]**2 + EPSILON**2)**(1.5)


# @njit
def advance_pos(acc, pos, pos_prev, pos_temp, dt):
    pos_temp[:] = pos[:]
    pos[:] = 2.0 * pos - pos_prev + acc * dt**2
    pos_prev[:] = pos_temp[:]


def main():

    dt = 0.01
    total_time = 10000*dt
    # total_time = N_YEARS*2*np.pi
    pos_tracker = []

    # Load initial conditions
    # pos, vel, mass = create_solar_system()
    pos, vel, mass = generate_random_star_system(N_PARTICLES)
    acc = np.zeros_like(pos)
    calc_acc(acc, pos, mass)
    pos_prev = pos - vel*dt - 0.5*acc*dt**2
    pos_temp = np.zeros_like(pos)

    start = perf_counter()

    t = 0
    while t < total_time:
        pos_tracker += [pos.copy()]
        calc_acc(acc, pos, mass)
        advance_pos(acc, pos, pos_prev, pos_temp, dt)
        t += dt

    end = perf_counter()

    print(f"Time to complete: {end-start}")

    positions_for_plotting = np.array(pos_tracker)

    ax = plt.axes(projection='3d')
    ax = plt.axes()
    xmin, xmax = 0.0, 0.0
    ymin, ymax = 0.0, 0.0
    for i in range(len(pos)):
        xdata = positions_for_plotting[:,i,0]
        ydata = positions_for_plotting[:,i,1]
        xmin = min(np.min(xdata), xmin)
        xmax = max(np.max(xdata), xmax)
        ymin = min(np.min(ydata), ymin)
        ymax = max(np.max(ydata), ymax)
        ax.plot(xdata, ydata)

    xmax = max(abs(xmax), abs(xmin))
    ymax = max(abs(ymax), abs(ymin))
    xmin = -xmax
    ymin = -ymax
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    # for N_PARTICLES in [8, 16, 32, 64, 128, 256, 512]:
        # print(N_PARTICLES)
        # main()