# SQGE ROM
Ted Balabanski

The following repository was made for the final project of MATH 5414 - Reduced Order Modelling for Fluids at Virginia Tech.
The stationary Quasi-Geostrophic Equation (SQGE) can be written (in streamfunction form) as

$$ \nu \Delta^2 \psi + J(\psi, \Delta \psi) - \mu \frac{\partial \psi}{\partial x} = \mu F \, \, \mathrm{in} \Omega$$

With homogenous Dirichlet and Neumann boundary conditions. Because of the fourth-order differential operator ($\Delta^2$),
the finite elements used must be at least $C^1$. To this end, this project makes use of tIGAr to implement B-Splines.

## Dependencies
In order to reproduce this code, FEnics and tIGAr are required. 
