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

    def integrate(self, dt):
        self.tracker += [self.pos.copy()]
        self.pos += self.vel * dt# + self.acc * dt * 0.5
        self.vel += self.acc * dt
        # self.vel += (self.acc_prev + self.acc) * dt * 0.5
        # self.acc_prev, self.acc = self.acc, self.acc_prev

    def calc_self_force(self, particles):
        self.acc[:] = 0.0
        for particle in particles:
            r = particle.pos - self.pos
            self.acc += particle.mass * r \
                    / (np.linalg.norm(r)**2 + EPSILON**2)**(1.5)

    def print(self):
        print(self.name)
        print(self.pos)
        print(self.vel)
        print(self.acc)




def create_solar_system():
    # Solar system data from https://physics.stackexchange.com/questions/441608/solar-system-position-and-velocity-data
    names = ["Sun", "Mercury", "Venus", "Mars", "Earth", "Jupiter", "Saturn", "Neptune", "Uranus", "Pluto"]
    m = np.array(([1],[1/6023600],[1/408524],[1/332946.038],[1/3098710],[1/1047.55],[1/3499],[1/22962],[1/19352]))
    r = np.array(([0,0],[0.4,0],[0,0.7],[1,0],[0,1.5],[5.2,0],[0,9.5],[19.2,0],[0,30.1]))
    v = np.array(([0,0],[0,-np.sqrt(1/0.4)],[np.sqrt(1/0.7),0],[0,-1],[np.sqrt(1/1.5),0],[0,-np.sqrt(1/5.2)],[np.sqrt(1/9.5),0],[0,-np.sqrt(1/19.2)],[np.sqrt(1/30.1),0]))

    return [Particle(r[i, 0], r[i,1], 0, v[i,0], v[i,1], 0, m[i,0], name=names[i]) for i in range(3)]


def main():

    particles = create_solar_system()

    for particle in particles:
        particle.calc_self_force(particles)
        particle.acc_prev[:] = particle.acc[:]

    dt = 0.001
    # ax = plt.axes(projection='3d')
    ax = plt.axes()
    t = 0.0
    i = 0
    while t < 4:
        for particle in particles:
            if i%10 == 0:
                particle.print()
            particle.calc_self_force(particles)
        for particle in particles:
            particle.integrate(dt)
        t += dt
        i += 1

    for p, color in zip(particles, ['red', 'gray', 'pink']):
        xdata = np.array(p.tracker)[:,0]
        print(xdata)
        ydata = np.array(p.tracker)[:,1]
        print(ydata)
        ax.plot(xdata, ydata, color=color)

    plt.show()


if __name__ == "__main__":
    main()
