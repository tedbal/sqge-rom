# dependencies
from __future__ import print_function
from fenics import *
import numpy as np
from math import *
from tIGAr import *


def variational_form_a(psi, phi, spline):
    # returns the inner product (D^2 psi, D^2 phi)
    return inner(spline.grad(spline.grad(psi)), spline.grad(spline.grad(phi)))*spline.dx



def variational_form_b(psi, phi, spline):
    # returns the form 
    # \del psi (\partial_y psi  \partial_x phi - \partial_x psi \partial_y \phi)
    # integrated over the domain omega
    return spline.div(spline.grad(psi)) * (Dx(psi, 1) * Dx(phi, 0)  - Dx(psi, 0) * Dx(phi, 1)) * spline.dx



def inner_prod_1(psi, phi):
    return inner(Dx(psi, 1), phi)*dx



def inner_prod_2(phi, F = sin(pi/4)):
    return inner(F, phi)*dx
