# dependencies
from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from tIGAr import *
from tIGAr.BSplines import *
from scipy.interpolate import griddata
from pathlib import Path
from mpi4py import MPI

# useful functions
from utils import *

# building the mesh
def define_mesh(nel_x, nel_y, p=3):
    # define the number of elements in x and y direction
    # nel_x - number of elements in the x direction (int)
    # nel_y - number of elements in the y direction (int)
    # p - degree of the splines, default 3 for C1 condition (int)

    # define the knots in the x and y directions
    kx = uniformKnots(p, 0.0, 1.0, nel_x)
    ky = uniformKnots(p, 0.0, 1.0, nel_y)

    # build and return the bspline control mesh
    mesh = ExplicitBSplineControlMesh([p, p], [kx, ky]) 
    return mesh

# building finite element operators
def build_fe_operators(mesh,
                       Re=10,
                       Ro=1e-4,
                       p=3,
                       filename=Path('../results/solution')):

    # builds and solves the full-order model
    # Re - Reynolds number (float)
    # Ro - Rosby number (float)
    # p - degree of the splines (default 3 for C1 elements)
    # filename - path to file to save plotting results

    
    # define the bspline generator for the mesh
    spline_generator = EqualOrderSpline(1, mesh)

    # get the scalar spline with homogenous dirichlet bcs
    scalar_spline = spline_generator.getScalarSpline(0)
    for parametric_dir in [0, 1]:
        for side in [0, 1]:
            sideDofs = scalar_spline.getSideDofs(parametric_dir, side)
            spline_generator.addZeroDofs(0, sideDofs)

    # get a spline from the generator, quadrature degree of 2 * p
    spline = ExtractedSpline(spline_generator, 2*p)
    
    # define test and trial functions
    psi = Function(spline.V)
    v = TestFunction(spline.V)
    
    # define constants and some forms
    nu = Constant(Re**-1)
    mu = Constant(Ro**-1)

    # assemble finite-element operators
    a = variational_form_a(psi, v, spline)
    b = variational_form_b(psi, v, spline)
    mu_term = inner_prod_1(psi, v)
    forcing = inner_prod_2(v)

    # solve the system
    F = nu*a + b - mu*mu_term - mu*forcing
    dF = derivative(F, psi)
    spline.solveNonlinearVariationalProblem(F,dF, psi)

    # save for plotting    
    with XDMFFile(MPI.comm_world, str(filename.with_suffix(".xdmf"))) as xdmf:
        xdmf.write(psi)

    # convert psi to an array and get the mesh coordinates
    psi_array = psi.vector().vec().array
    coordinates = spline.V.tabulate_dof_coordinates().reshape((-1, 2))

    return psi_array, coordinates
    
# testing
if __name__ == "__main__":
    # define the mesh
    mesh = define_mesh(500, 500)

    # solve the fe-system
    build_fe_operators(mesh, Re = 5.27, Ro = 6.05e-4)   
