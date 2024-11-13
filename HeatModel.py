#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:28:29 2024

@author: ramsonmunoz
"""

from scipy import sparse
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns


class HeatModel:

    def __init__(self, size):
        self.size = size

        self.designMatrix = self.create_Model().tocsr()

        self.b = self.constructVectorB()
 
    def create_diagonal_matrix(self, diagonalElement, offDiagonalElement):
        """


        Parameters
        ----------
        diagonalElement : float
            main diagonal element for sparse diagonal element.
        offDiagonalElement : float
            general off diagonal element. our case uses ones.

        Raises
        ------
        ValueError
            returns error if constructor is called with non-positive int.
        TypeError
            if constructor input is not int

        Returns
        -------
        sparse.diagonal object
            a sparse matrix containing diagonal elements of the input element and off-diagonal elements as specified

        """

        try:

            int(self.size)

            if (self.size < 1):
                raise ValueError("Ensure model size is at least 1")
            elif (self.size == 1):
                return np.array([[-4]])

            data = [offDiagonalElement * np.ones(self.size - 1), diagonalElement*np.ones(
                self.size), offDiagonalElement * np.ones(self.size - 1)]
            offsets = [-1, 0, 1]

            return sparse.diags(data, offsets, shape=(self.size, self.size))

        except ValueError as e:
            print(e)
            sys.exit()
        except TypeError as e:
            print(e)
            sys.exit()

    def create_large_diagonal_matrix(self):
        """
        Generates the B matrix for our large design matrix

        Returns
        -------
        Sparse.diag matrix

        """
        return self.create_diagonal_matrix(-4, 1)

    def create_Model(self):
        """


        Returns
        -------
        scipy sparse matrix object
            our design matrix as a scipy.sparse object.

            implemented as a block diagonal matrix such that main diagonal is an n x n matrix of the from B

            where B =[[-4.,  1.,  0.],
                      [ 1., -4.,  1.],
                      [ 0.,  1., -4.]] for 3 x 3

                  [[-4.,  1.,  0.,  0.,  0.,  0.,  0.],
                   [ 1., -4.,  1.,  0.,  0.,  0.,  0.],
                   [ 0.,  1., -4.,  1.,  0.,  0.,  0.],
                   [ 0.,  0.,  1., -4.,  1.,  0.,  0.],
                   [ 0.,  0.,  0.,  1., -4.,  1.,  0.],
                   [ 0.,  0.,  0.,  0.,  1., -4.,  1.],
                   [ 0.,  0.,  0.,  0.,  0.,  1., -4.]] for B(7 x 7)

        generalized to nxn for positive values n

        and on the of diagonal blocks an Identity matrix of size n


        final matrix is n^2 x n^2



        """

        diagonalMatrix = self.create_large_diagonal_matrix()

        finalMatrix = sparse.block_diag([diagonalMatrix] * self.size)

        # creates Identity above main diagonal
        finalMatrix.setdiag(
            np.ones((self.size * self.size) - self.size), self.size)

        # creates Identity below main diagonal
        finalMatrix.setdiag(
            np.ones((self.size * self.size) - self.size), -self.size)

        return finalMatrix

    def constructVectorB(self):
        """
        
        
        Returns
        -------
        A vector size n^2 x 1
            Used as the target vector for Ax = b when solving for Jacobian and GaussSeidel.

            b = [[-100],
                [0],
                [-100]
                [-100],
                [0],
                [-100]
                [-100],
                [0],
                [-100]] when self.size = 3
        """
        # Constructing a matrix of zeros size n x n
        matrix = np.zeros((self.size, self.size))

        # Changing the top and bottom row values to -100
        for i in range(self.size):
            matrix[0,i] = -100
            matrix[-1,i] = -100
        
        # Flattening the matrix into a column vector
        return matrix.T.flatten().reshape(-1,1)
        

    def visualizeModelDesign(self):
        """
        Simple function that converts the sparse matrix back to a dense matrix and 
        returns a heatmap of the design matrix with the seaborn library

        Returns
        ------
        matplotlib axes

        To use
            plot = self.visualize()
            plot.show()

        OR

        In a notebook
            self.visualize()
        """
        # Convert the sparse matrix back into a normal matrix
        matrix = self.designMatrix.todense()

        # Constructing two subplots for the final plot
        fig, (ax, ax_bar) = plt.subplots(figsize = (12,8),
                                         ncols = 2,
                                         constrained_layout = True, 
                                         gridspec_kw = {"width_ratios": [0.8,0.2]})
        
        # Creating a heatmap of the design matrix
        sns.heatmap(matrix, ax = ax)
        ax.set_title("Design Matrix")

        # Creating a visual representation of the vector b
        sns.heatmap(self.b, ax = ax_bar)
        ax_bar.set_title("Vector b")

        # Adding a title to the figure
        fig.suptitle("Visual Representation of Design Matrix and Vector b",
                     fontsize = "xx-large")

        return fig

    def GaussSeidel(self, nIter = 100, tol = 1e-8):
        
        # Starting with an initial vector of ones
        x = np.ones(self.size**2)
        # Creating a copy for future iterations
        x1 = np.copy(x)
        # Storing the design matrix and vector b from the class constructor
        A = self.designMatrix
        b = self.b

        # Storing the Diagonals in A for easier access later
        diags = A.diagonal()
        
        # Initializing a counter and a loss array
        count = 0
        tols = []

        # Beginning outer loop of nIters
        for i in range(nIter):

            # Inner loop to generate x_k+1
            for i in range(x.shape[0]):
                # Slicing through the sparse matrix for non zero values
                rowstart = A.indptr[i]
                rowend = A.indptr[i+1]
                # Taking A[i].T @ x
                Aix = A.data[rowstart:rowend] @ x[A.indices[rowstart:rowend]]

                # Generating each element for x_k+1
                x1[i] = x[i] + ( (b[i] - Aix) / diags[i] )

            # Calculating the difference 
            x_x1 = x - x1
            x = np.copy(x1)
            diff = np.linalg.norm(x_x1)
            tols.append(diff)
            count +=1

            if diff < tol:
                break

        return x1, count, tols