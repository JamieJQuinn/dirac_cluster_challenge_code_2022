import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.random import default_rng
from time import perf_counter


SEED = 0


def handle_corners(x):
    x[ 0,  0] = 0.5*(x[ 1,  0] + x[ 0,  1])
    x[ 0, -1] = 0.5*(x[ 1, -1] + x[ 0, -2])
    x[-1,  0] = 0.5*(x[-2,  0] + x[-1,  1])
    x[-1, -1] = 0.5*(x[-2, -1] + x[-1, -2])


def set_dirichlet_bcs(x, bcs):
    """
    Apply Dirichlet boundary conditions; conditions that specify a particular value of x on the boundary.
    """
    left, top, right, bottom = bcs
    x[ 0, :] = -x[1,:] + 2.*left
    x[-1, :] = -x[-2,:] + 2.*right
    x[ :,  0] = -x[:,1] + 2.*bottom
    x[ :, -1] = -x[:,-2] + 2.*top
    handle_corners(x)


def set_von_neumann_bcs(x):
    """
    Apply von Neumann boundary conditions; conditions that specify a particular value of the *derivative* of x on the boundary.
    """
    x[ 0, :] = x[ 1, :]
    x[-1, :] = x[-2, :]
    x[:,  0] = x[:,  1]
    x[:, -1] = x[:, -2]
    handle_corners(x)


def set_periodic_bcs_x(x):
    """
    Apply periodic boundary conditions in x; speedy things go in, speedy things go out
    """
    x[ 0, :] = x[-2, :]
    x[-1, :] = x[1, :]


def set_periodic_bcs_y(x):
    """
    Apply periodic boundary conditions in y
    """
    x[:,  0] = x[:, -2]
    x[:, -1] = x[:, 1]


def set_periodic_bcs(x):
    """
    Apply periodic boundary conditions in x and y
    """
    set_periocic_bcs_x(x)
    set_periodic_bcs_y(x)
    handle_corners(x)


def set_u_bcs(u):
    bottom = 0.
    top = 0.
    u[ :,  0] = -u[:,1] + 2.*bottom
    u[ :, -1] = -u[:,-2] + 2.*top

    set_periodic_bcs_x(u)
    handle_corners(u)


def set_v_bcs(v):
    bottom = 0.
    top = 0.
    v[ :,  0] = -v[:,1] + 2.*bottom
    v[ :, -1] = -v[:,-2] + 2.*top

    set_periodic_bcs_x(v)
    handle_corners(v)


def set_pressure_bcs(p):
    set_von_neumann_bcs(p)


def set_tmp_bcs(tmp):
    # Top and bottom are at a fixed temp
    bottom = 0.
    top = -1.
    tmp[ :,  0] = -tmp[:,1] + 2.*bottom
    tmp[ :, -1] = -tmp[:,-2] + 2.*top
    # Left and right have zero heat flux
    tmp[ 0, :] = tmp[ 1, :]
    tmp[-1, :] = tmp[-2, :]


def diffuse(x, x0, dt, diff, nx, ny, dx, dy, set_boundary_fn, n_iterations=100):
    """
    Add diffusion term to x using Gauss-Siedel method
    """
    temp = np.zeros_like(x)
    D = (1.+2.*diff*dt*(1./dx**2 + 1./dy**2))
    for k in range(n_iterations):
        temp[1:-1,1:-1] = (x0[1:-1,1:-1] + diff*dt*(
            (x[:-2,1:-1] + x[2:,1:-1])/dx**2 + (x[1:-1,:-2] + x[1:-1,2:])/dy**2))/D
        x[:] = temp[:]
        set_boundary_fn(x)


def project(u, v, p, div, nx, ny, dx, dy, max_iterations = 100):
    """
    Calculate pressure required to keep fluid incompressible and add pressure force to velocity
    """
    # Calculate divergence of flow; this is what we want to make zero
    div[1:-1,1:-1] = 0.5*(u[2:,1:-1] - u[:-2,1:-1])/dx + 0.5*(v[1:-1,2:] - v[1:-1,:-2])/dy
    set_dirichlet_bcs(div, [0., 0., 0., 0.])

    # Initialise pressure
    p[:] = 0.
    set_pressure_bcs(p)

    temp = np.zeros_like(p)

    D = -2.*(1./dx**2 + 1./dy**2)

    # Perform Jacobi iteration to calculate pressure
    for k in range(max_iterations):
        temp[1:-1,1:-1] = (div[1:-1,1:-1] - (p[:-2,1:-1] + p[2:,1:-1])/dx**2 - (p[1:-1,:-2] + p[1:-1,2:])/dy**2)/D
        p[:] = temp[:]
        set_pressure_bcs(p)

    # Add pressure force to velocity
    u[1:-1,1:-1] -= 0.5*(p[2:,1:-1] - p[:-2,1:-1])/dx
    v[1:-1,1:-1] -= 0.5*(p[1:-1,2:] - p[1:-1,:-2])/dy

    set_u_bcs(u)
    set_v_bcs(v)


def advect(q, q0, u, v, dt, dx, dy, nx, ny, set_boundary_fn):
    """
    Advect the quantity q (original value in q0) using velocities u and v
    """
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            x = i-dt/dx*u[i,j]
            y = j-dt/dy*v[i,j]

            x = max(x, 0.5)
            x = min(x, nx+0.5)
            y = max(y, 0.5)
            y = min(y, ny+0.5)

            i0 = int(x)
            j0 = int(y)
            i1 = i0+1
            j1 = j0+1
            s1 = x-i0
            s0 = 1-s1
            t1 = y-j0
            t0 = 1-t1

            q[i,j] = \
                s0*(t0*q0[i0, j0] + t1*q0[i0, j1]) +\
                s1*(t0*q0[i1, j0] + t1*q0[i1, j1])

    set_boundary_fn(q)


def main():
    nx = 64
    ny = 64

    lx = 1.
    ly = 1.
    Pr = 10.
    Ra = 800.
    tmp_diff = 1.0

    dx = lx/nx
    dy = ly/ny

    epsilon = 0.01

    dt = 0.001
    # dt = 0.5*min(dx**2/visc, dx)
    dump_dt = 0.01
    total_time = 0.5

    FP_TYPE = np.float32

    print(f'Numerical diffusion: {dx**2/dt}')
    print(f'Explicit dt (diffusion): {dx**2/Pr}')
    print(f'Explicit dt (CFL): {dx}')
    print(f'dt: {dt}')

    u = np.zeros((nx+2, ny+2), dtype=FP_TYPE)
    v = np.zeros_like(u)
    u_prev = np.zeros_like(u)
    v_prev = np.zeros_like(u)
    tmp = np.zeros_like(u)
    tmp_prev = np.zeros_like(u)
    pressure = np.zeros_like(u)

    rng = default_rng(SEED)
    tmp += epsilon * (2 * rng.random(tmp.shape) - 1.0)

    x = np.linspace(-dx, lx+dx, nx+2)
    y = np.linspace(-dy, ly+dy, ny+2)
    X, Y = np.meshgrid(x, y, indexing='ij')

    set_u_bcs(u)
    set_u_bcs(u_prev)
    set_v_bcs(v)
    set_v_bcs(v_prev)

    u_conv = np.zeros_like(u)
    v_conv = np.zeros_like(u)

    av_u_conv = 0.
    av_v_conv = 0.

    fig, ax = plt.subplots()
    plots = []

    start = perf_counter()

    t = 0.
    step_counter = 0
    time_to_next_dump = 0.
    dump_counter = 0
    while t < total_time:
        if (t > time_to_next_dump):
            # plot = plt.quiver(x[1:-1], y[1:-1], u[1:-1,1:-1].T, v[1:-1,1:-1].T, animated=True)
            # vorticity = (v[2:,1:-1] - v[:-2,1:-1])/(2.*dx) - (u[1:-1,2:] - u[1:-1,:-2])/(2.*dy)
            # plot = plt.pcolormesh(X[1:-1,1:-1], Y[1:-1,1:-1], tmp[1:-1,1:-1], vmin=-1, vmax=1)
            plot = plt.pcolormesh(X[1:-1,1:-1], Y[1:-1,1:-1], tmp[1:-1,1:-1])
            # vlimit = max(abs(np.max(vorticity)), abs(np.min(vorticity)))
            # plot = plt.pcolormesh(X[1:-1,1:-1], Y[1:-1,1:-1], vorticity, cmap='RdBu', vmax=vlimit, vmin=-vlimit)
            plt.colorbar(plot)
            plt.quiver(x, y, u.T, v.T)
            ax.set_aspect('equal')
            # plt.xlim(0, lx)
            # plt.ylim(0, ly)

            fname = f"{dump_counter:04d}.png"
            plt.savefig(fname)
            plt.close()

            print(f"t={t} of {total_time}")
            dump_counter += 1
            time_to_next_dump += dump_dt
        # u_prev[(X-lx/2.)**2 + (Y-ly/2.)**2 < 0.1**2] = 1.
        v_prev[1:-1,1:-1] += Ra*Pr*tmp[1:-1,1:-1]
        diffuse(u, u_prev, dt, Pr, nx, ny, dx, dy, set_u_bcs)
        diffuse(v, v_prev, dt, Pr, nx, ny, dx, dy, set_v_bcs)
        diffuse(tmp, tmp_prev, dt, tmp_diff, nx, ny, dx, dy, set_tmp_bcs)
        project(u, v, u_prev, v_prev, nx, ny, dx, dy)
        u_prev[:] = u[:]
        v_prev[:] = v[:]
        tmp_prev[:] = tmp[:]
        advect(u, u_prev, u_prev, v_prev, dt, dx, dy, nx, ny, set_u_bcs)
        advect(v, v_prev, u_prev, v_prev, dt, dx, dy, nx, ny, set_v_bcs)
        advect(tmp, tmp_prev, u_prev, v_prev, dt, dx, dy, nx, ny, set_tmp_bcs)
        project(u, v, u_prev, v_prev, nx, ny, dx, dy)
        # pressure[:] = u_prev[:]
        u_prev[:] = u[:]
        v_prev[:] = v[:]
        tmp_prev[:] = tmp[:]
        t += dt
        step_counter += 1

    end = perf_counter()
    print(f"Time to complete: {end-start}")
    print(f"FPS: {(step_counter)/(end-start)}")


if __name__ == "__main__":
    main()
