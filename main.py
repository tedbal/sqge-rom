import matplotlib.pyplot as plt
import numpy as np

from build_fe_operators import *
from build_rom_basis_and_operators import *

# ====================================================
#               DEFINE PROBLEM COSNTANTS
# ====================================================
# first we define the number of elements in the x and y direction
# using the square domain [0, 1] x [0, 1], so will use equal
# number of elements in both directions

NELX = NELY = 16

# additionally, we need to build the nondimensional numbers which
# define the problem, viz. Reynolds and Rosby numbers

RE = [1, 5, 10]
RO = [1, 0.1, 0.01, 0.001]

# finally, we define the number of ROM modes

ROM_MODES = 10

# ====================================================
#           BUILD THE FINITE ELEMENT MODEL
# ====================================================
# build the finite element model

fe_model = FiniteElementModel(RE, RO, NELX, NELY)

# status update

print(f"initialized problem on unit square with {NELX} x {NELY} elements")
print(f"initialized the QGE problem with Re = {RE:.2e} and Ro = {RO:.2e}")

# build the finite element operators

print("building finite element solution and operators...")
fe_model.build_fe_operators()


# ====================================================
#            BUILD THE REDUCED ORDER MODEL
# ====================================================
# build the ROM operators

print(f"building ROM with {ROM_MODES} modes...")
rom = build_rom(fe_model, ROM_MODES)

print(f"built the ROM model with reconstruction error = {rom.reconstruction_error:.2e}") 

# ====================================================
#                EVALUATE ON NEW REGIME
# ====================================================
# define some new regime

NEW_RE = 10
NEW_RO = 0.01

print(f"initialized the new QGE problem with Re = {NEW_RE:.0f} and Ro = {NEW_RO:.3f}") 

# get the ground truth of the new regime
print("getting new regime finite element solution...")

start = time.time()
new_regime_fe_model = FiniteElementModel(NEW_RE, NEW_RO, NELX, NELY)
new_regime_fe_model.build_fe_operators()
fe_time = time.time() - start

new_regime_soln = new_regime_fe_model.psi_array

# get the ROM evaluation of the new regime
print("getting rom solution of new regime...")
start = time.time()
rom_error = rom.evaluate(new_regime_soln)
rom_time = time.time() - start

# print status
print("finished!")
print("============ TIME RESULTS ============")
print(f"finite element time: {fe_time:.2e}")
print(f"rom time:            {rom_time:.2e}")
print("============ ERROR RESULTS ============")
print(f"rom proj error:      {rom_error:.2e}")
