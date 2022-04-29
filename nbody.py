import numpy as np
from numpy.random import default_rng
from scipy.constants import gravitational_constant as G
import matplotlib.pyplot as plt
from time import perf_counter


EPSILON = 0.001
N_YEARS = 50
N_PARTICLES = 2
FP_TYPE = np.float64


def random(num, a=0., b=1.):
    rng = default_rng()
    return rng.random(num, dtype=FP_TYPE)*(b-a) + a


def generate_random_ics(num):
    r = random(num, 0.4, 20)
    theta = random(num, 0., np.pi)
    v_mag = 1./np.sqrt(r)

    pos = np.zeros((num, 2), dtype=FP_TYPE)
    vel = np.zeros((num, 2), dtype=FP_TYPE)

    pos[:,0] = r*np.sin(theta)
    pos[:,1] = r*np.cos(theta)

    vel[:,0] = -v_mag*np.cos(theta)
    vel[:,1] =  v_mag*np.sin(theta)

    mass = random(num, 1/6000000, 1/1000)

    pos[0,:] = 0.
    vel[0,:] = 0.
    mass[0] = 1.

    return pos, vel, mass


def create_solar_system():
    # Solar system data from https://physics.stackexchange.com/questions/441608/solar-system-position-and-velocity-data

    names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Neptune", "Uranus"]

    mass = np.array((1,1/6023600,1/408524,1/332946.038,1/3098710,1/1047.55,1/3499,1/22962,1/19352))
    pos = np.array(([0,0],[0.4,0],[0,0.7],[1,0],[0,1.5],[5.2,0],[0,9.5],[19.2,0],[0,30.1]))
    vel = np.array(([0,0],[0,-np.sqrt(1/0.4)],[np.sqrt(1/0.7),0],[0,-1],[np.sqrt(1/1.5),0],[0,-np.sqrt(1/5.2)],[np.sqrt(1/9.5),0],[0,-np.sqrt(1/19.2)],[np.sqrt(1/30.1),0]))

    return pos, vel, mass


def calc_acc(acc, pos, mass):
    for i in range(len(pos)):
        r = pos - pos[i]
        acc[i] = np.sum(r.T * mass / (r[:, 0]**2 + r[:, 1]**2 + EPSILON**2)**(1.5), axis=1)


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
    pos, vel, mass = create_solar_system()
    # pos, vel, mass = generate_random_ics(N_PARTICLES)
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
