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


def main():
    # test_advection()
    # test_dirichlet()

main()
