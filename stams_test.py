import numpy as np
import numba
import matplotlib.pyplot as plt

# from pytest import approx
from stams import advect, set_dirichlet_bcs, solve, set_von_neumann_bcs

def test_advection():
    nx = 8
    ny = 5

    u = np.zeros((nx+2, ny+2))
    v = np.zeros_like(u)
    q = np.zeros_like(u)
    q0 = np.zeros_like(u)

    u[:] = 1.0
    v[:] = 2.0

    q0[2, 3] = 1.0

    advect(2, q, q0, u, v, 1.0, 1.0, 1.0, nx, ny)

    assert q[3, 5] == approx(q0[2, 3])


def test_dirichlet():
    nx = 8
    ny = 5

    u = np.zeros((nx+2, ny+2))

    set_dirichlet_bcs(u, (1.0, 2.0, -0.5, 3.0))

    assert 0.5*(u[1:-1,0]+u[1:-1,1]) == approx(3.0)
    assert 0.5*(u[1:-1,-1]+u[1:-1,-2]) == approx(2.0)
    assert 0.5*(u[0,1:-1]+u[1,1:-1]) == approx(1.0)
    assert 0.5*(u[-1,1:-1]+u[-2,1:-1]) == approx(-0.5)


def test_conjugate_gradient():
    nx = 16
    ny = 31
    lx = 1.0
    ly = 1.0
    dx = lx/nx
    dy = ly/ny

    x = np.linspace(-dx, lx+dx, nx+2)
    y = np.linspace(-dy, ly+dy, ny+2)
    X, Y = np.meshgrid(x, y, indexing='ij')

    u = np.sin(2*np.pi*X)*np.sin(2*np.pi*Y)
    # u = np.zeros((nx+2, ny+2))

    # print("Plot initial conditions")
    # im = plt.imshow(u.T, origin='lower')
    # plt.colorbar(im)
    # plt.show()

    nabla2 = numba.njit()(lambda x, i, j: (x[i+1,j] - 2.0*x[i,j] + x[i-1,j])/dx**2 + (x[i,j+1] - 2.0*x[i,j] + x[i,j-1])/dy**2)
    apply_bcs = numba.njit()(lambda x: set_dirichlet_bcs(x, [0, 0, 0, 0]))

    b = np.zeros_like(u)
    b[:,-2] = -2.0*1.0/dy**2

    temp1 = np.zeros_like(u)
    temp2 = np.zeros_like(u)

    solve(nabla2, u, b, temp1, temp2, nx, ny, apply_bcs, max_iterations=100)

    set_dirichlet_bcs(u, [0, 1, 0, 0])

    print("Plot result")
    im = plt.imshow(u.T, origin='lower')
    plt.colorbar(im)
    plt.show()



def main():
    # test_advection()
    # test_dirichlet()
    test_conjugate_gradient()

main()
