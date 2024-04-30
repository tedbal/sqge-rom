# dependencies
from __future__ import print_function
from fenics import *
import numpy as np
from math import *
from tIGAr import *


def variational_form_a(psi, phi, spline):
    # returns the inner product (D^2 psi, D^2 phi)
    return inner(spline.div(spline.grad(psi)), spline.div(spline.grad(phi))) * spline.dx



def variational_form_b(psi, phi, spline):
    # returns the form 
    # \del psi (\partial_y psi  \partial_x phi - \partial_x psi \partial_y \phi)
    # integrated over the domain omega

    # define the laplacian of psi and the jacobian form
    del_psi = spline.div(spline.grad(psi))
    J = Dx(psi, 0) * Dx(del_psi, 1) - Dx(psi, 1) * Dx(del_psi, 0)

    # return the variational form
    return inner(J, phi) * spline.dx



def inner_prod_1(psi, phi, spline):
    return inner(Dx(psi, 0), phi) * spline.dx



def inner_prod_2(phi, spline, f = None):
    # if no forcing is passed, default to sin(pi*(y - 1))
    if f == None:
        x = spline.spatialCoordinates()
        f = sin(pi*(x[1] - 1))

    # return the variational form
    return inner(f, phi) * spline.dx


def stabilizing_term(psi, phi, spline, a1, a2):
    return a1 * inner(psi, phi) * spline.ds + a2 * inner(spline.grad(psi), spline.grad(phi)) * spline.ds
