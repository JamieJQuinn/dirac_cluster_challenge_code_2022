import numpy as np
from scipy.constants import gravitational_constant as G
import matplotlib.pyplot as plt


EPSILON = 0.001


class Particle:
    def __init__(self, x, y, z, vx, vy, vz, mass, name=None):
        self.pos = np.array([x, y, z], dtype=np.float64)
        self.vel = np.array([vx, vy, vz], dtype=np.float64)
        self.acc = np.array([0, 0, 0], dtype=np.float64)
        self.acc_prev = np.array([0, 0, 0], dtype=np.float64)

        self.mass = mass
        self.name = name

        self.tracker = []


def create_solar_system():
    # Solar system data from https://physics.stackexchange.com/questions/441608/solar-system-position-and-velocity-data
    names = ["Sun", "Mercury", "Venus", "Mars", "Earth", "Jupiter", "Saturn", "Neptune", "Uranus"]
    m = np.array(([1],[1/6023600],[1/408524],[1/332946.038],[1/3098710],[1/1047.55],[1/3499],[1/22962],[1/19352]))
    r = np.array(([0,0],[0.4,0],[0,0.7],[1,0],[0,1.5],[5.2,0],[0,9.5],[19.2,0],[0,30.1]))
    v = np.array(([0,0],[0,-np.sqrt(1/0.4)],[np.sqrt(1/0.7),0],[0,-1],[np.sqrt(1/1.5),0],[0,-np.sqrt(1/5.2)],[np.sqrt(1/9.5),0],[0,-np.sqrt(1/19.2)],[np.sqrt(1/30.1),0]))

    return [Particle(r[i, 0], r[i,1], 0, v[i,0], v[i,1], 0, m[i,0], name=names[i]) for i in range(len(names))]


def main():

    particles = create_solar_system()

    dt = 0.01
    # ax = plt.axes(projection='3d')
    ax = plt.axes()
    t = 0.0
    i = 0
    while t < 40:
        for p1 in particles:
            p1.acc[:] = 0.0
            for p2 in particles:
                r = p2.pos - p1.pos
                p1.acc += p2.mass * r \
                        / (np.linalg.norm(r)**2 + EPSILON**2)**(1.5)
        for particle in particles:
            particle.tracker += [particle.pos.copy()]
            particle.pos += particle.vel * dt + 0.5 * particle.acc * dt**2
            particle.vel += particle.acc * dt
        t += dt
        i += 1

    for p in particles:
        xdata = np.array(p.tracker)[:,0]
        ydata = np.array(p.tracker)[:,1]
        ax.plot(xdata, ydata, label=p.name)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
