# build rom basis and operators
# dependencies
import numpy as np
from scipy.optimize import fsolve
from pathlib import Path
import matplotlib.pyplot as plt

# wrapper to initialize reduced order model
def build_rom(fe_model, rom_dim, savepath=Path('../results/rom')):
    return ReducedOrderModel(rom_dim,
                             fe_model.re,
                             fe_model.ro,
                             fe_model.fe_mass_matrix,
                             fe_model.psi_array,
                             fe_model.fe_a_array,
                             fe_model.fe_b_array,
                             fe_model.fe_mu_array,
                             fe_model.fe_forcing_array,
                             savepath)

# implementation of the reduced order model class
class ReducedOrderModel:
    def __init__(self,
                 rom_dim,
                 fe_re,
                 fe_ro,
                 fe_mass_matrix,
                 fe_snapshots,
                 fe_a_array,
                 fe_b_array,
                 fe_mu_array,
                 fe_forcing_array,
                 savepath):
        
        # initialize the class variables
        self.rom_dim = rom_dim
        self.fe_snapshots = fe_snapshots.reshape((1, -1))
        self.savepath = savepath

        # build rom basis
        print("computing ROM basis...")
        self.build_rom_basis(fe_mass_matrix)
        self.plot_eigenvalues()
        
        # build rom operators
        print("building ROM operators...")
        self.build_rom_operators(fe_a_array,
                                 fe_b_array,
                                 fe_mu_array,
                                 fe_forcing_array)

        # evaluate reconstruction error
        print("evaluating rom reconstruction error...")
        rom_soln = self.evaluate(fe_re, fe_ro)
        error = np.norm(self.fe_snapshots - rom_soln)/np.norm(self.fe_snapshots)
        self.reconstruction_error = error
        
        # update the user
        print("done!")


    def plot_eigenvalues(self):
        # get the normalized eigenvalues
        lambda_norm = self.lambdas / self.lambdas[0]

        # format the plot
        plt.rcParams.update({'font.size': 14})
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(lambda_norm, 'r-')

        # save the plot
        plt.savefig(self.savepath.with_suffix('.png'))

        
    def build_rom_basis(self, fe_mass_matrix):
        # solve the eigenvalue problem
        lambdas, vs = np.linalg.eig(self.fe_snapshots.T * fe_mass_matrix * self.fe_snapshots)

        # compute the matrix C
        C = np.reciprocal(np.sqrt(lambdas[:self.rom_dim])) * self.fe_snapshots * vs[:, :self.rom_dim]

        # update the class to include C
        self.C = C
        self.lambdas = lambdas


    def build_rom_operators(self,
                            fe_a_array,
                            fe_b_array,
                            fe_mu_array,
                            fe_forcing_array):
        # generate the linear rom operators
        print(self.C.T.shape, fe_a_array.shape, self.C.shape)
        print((fe_a_array @ self.C).T.shape)
        rom_a = self.C @ (fe_a_array @ self.C).T
        rom_mu = np.matmul(self.C.T, np.matmul(fe_mu_array, self.C))
        rom_forcing = np.matmul(self.C.T, np.matmul(fe_forcing_array, self.C))

        # DEPRICATED:
        # deim for nonlinear term
        # Algorithm 7 - Volkwein, 2013
        """
        # svd on snapshots
        pod_basis, sing_vals, _ = np.linalg.svd(self.fe_snapshots)

        # initialize the matrix U
        U = np.zeros((pod_basis.shape[0], self.rom_dim))
        U[:, 0] = pod_basis[:, 0]

        # set the initial index and index vector
        idx = np.argmax(pod_basis[:, 0])
        idx_vec = np.full((self.rom_dim, 1), False)
        idx_vec[idx] = True

        # start the greedy procedure    
        for i in range(1, self.rom_dim):
            # solve the linear system Uc = u and get residual
            u = pod_basis[:, i]
            c = np.linalg.solve(U[idx_vec], u[idx_vec])
            resid = u - U*c

            # greedily choose next index
            idx = np.argmax(r)

            # update the matrix U and index_vec
            U[:, i] = u
            idx_vec[idx] = True
        """
        
        rom_b = np.matmul(self.C.T, np.matmul(self.C.T, np.matmul(fe_b_array,self.C)))

        # update the class with the rom model
        self.rom_a = rom_a
        self.rom_b = rom_b
        self.rom_mu = rom_mu
        self.rom_forcing = rom_forcing

        
    def evaluate(self, evaluation_re, evaluation_ro):
        # solve the linear system
        # 0 = b + A*a + a.T * B * a
        # define the necessary variants
        # constant variant
        b = (evaluation_ro**-1) @ self.rom_forcing
        
        # linear variant
        A = (evaluation_ro**-1) @ self.rom_mu - (evaluation_re**-1) @ self.rom_a
        
        # quadratic variant
        B = self.rom_b

        # calculate the rom-space solution
        eval_func = lambda a: b + A @ a + a.T @ B @ a
        a_soln = fsolve(eval_func, self.C @ self.fe_snapshots)
        
        # return the fe-space solution
        return self.C.T * a
