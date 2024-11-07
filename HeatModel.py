#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:28:29 2024

@author: ramsonmunoz
"""

from scipy import sparse
import numpy as np
import sys
import seaborn as sns


class HeatModel:

    def __init__(self, size):
        self.size = size

        self.designMatrix = self.create_Model()
 
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

    def visualize(self):
        """
        Simple function that converts the sparse matrix back to a dense matrix and 
        returns a heatmap of the design matrix with the seaborn library

        To use, just call self.visualize()
        """
        # Convert the sparse matrix back into a normal matrix
        matrix = self.designMatrix.todense()

        # Show the matrix as a heatmap using seaborn
        return sns.heatmap(matrix)