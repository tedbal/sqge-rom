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

class FiniteElementModel:
    def __init__(self, nel_x, nel_y, p = 3):
        # set the model parameters
        self.p = p   # degree of spline polynomials, 3 ensures C1 continuity

        # build the mesh
        self.define_mesh(nel_x, nel_y, p)

        
    # building the mesh
    def define_mesh(self, nel_x, nel_y, p):
        # define the number of elements in x and y direction
        # nel_x - number of elements in the x direction (int)
        # nel_y - number of elements in the y direction (int)
        # p - degree of the splines, default 3 for C1 condition (int)

        # define the knots in the x and y directions
        kx = uniformKnots(p, 0.0, 1.0, nel_x)
        ky = uniformKnots(p, 0.0, 1.0, nel_y)

        # build and return the bspline control mesh
        mesh = ExplicitBSplineControlMesh([p, p], [kx, ky]) 
        self.mesh = mesh

    # building finite element operators
    def build_fe_operators(self,
                           filename=Path('../results/solution')):
        # builds and solves the full-order model
        # filename - path to file to save plotting results

        # define the bspline generator for the self.mesh
        spline_generator = EqualOrderSpline(1, self.mesh)

        # get the scalar spline with homogenous dirichlet bcs
        scalar_spline = spline_generator.getScalarSpline(0)
        for parametric_dir in [0, 1]:
            for side in [0, 1]:
                # set neumann and dirichlet boundary conditions
                sideDofs = scalar_spline.getSideDofs(parametric_dir, side, nLayers = 2)
                spline_generator.addZeroDofs(0, sideDofs)

        # get a spline from the generator, quadrature degree of 2 * p
        spline = ExtractedSpline(spline_generator, 2*self.p)

        # define test and trial functions
        psi = Function(spline.V)
        v = TestFunction(spline.V)

        # define constants and some forms
        alpha_1 = Constant(1e3)
        alpha_2 = Constant(5e-2)

        # define finite-element operators
        a = variational_form_a(psi, v, spline)
        b = variational_form_b(psi, v, spline)
        mu_term = inner_prod_1(psi, v, spline)
        stab_term = stabilizing_term(psi, v, spline, alpha_1, alpha_2)
        forcing = inner_prod_2(v, spline)

        # assemble finite element operators
        self.fe_mass_matrix = assemble(inner(psi, v) * spline.dx).get_local().reshape((1, -1))
        self.fe_a_array = assemble(a).get_local().reshape((1, -1))
        self.fe_b_array = assemble(b).get_local().reshape((1, -1))
        self.fe_mu_array = assemble(mu_term).get_local().reshape((1, -1))
        self.fe_forcing_array = assemble(forcing).get_local().reshape((1, -1))

        # declare class parameters
        self.a = a
        self.b = b
        self.mu_term = mu_term
        self.stab_term = stab_term
        self.forcing = forcing
        self.psi = psi
        self.spline = spline

    def solve_fe_system(self, re, ro):
        # declare constants
        self.re = re
        self.ro = ro
        nu = Constant(re**-1)
        mu = Constant(ro**-1)

        # solve the system
        F = nu * self.a + self.b - mu * self.mu_term - mu * self.forcing + self.stab_term
        dF = derivative(F, self.psi)
        spline.solveNonlinearVariationalProblem(F, dF, self.psi)

        # save for plotting    
        with XDMFFile(MPI.comm_world, str(filename.with_suffix(".xdmf"))) as xdmf:
            xdmf.write(self.psi)

        # convert psi to an array and get the mesh coordinates
        psi_array = self.psi.vector().get_local()
        coordinates = self.spline.V.tabulate_dof_coordinates().reshape((-1, 2))

        self.psi_array = psi_array

        
# testing
if __name__ == "__main__":
    # constants for the model
    re = 10
    ros = [1, 0.1, 0.01, 0.001]
    nelx = nely = 64

    for ro in ros:
        # build the fe model
        fe_model = FiniteElementModel(re, ro, nelx, nely)

        # solve the fe system
        ro_str = str(ro).replace(".", ",")
        filename_str = f"../results/fe_re={re}_ro={ro_str}"
        fe_model.build_fe_operators(filename = Path(filename_str))
