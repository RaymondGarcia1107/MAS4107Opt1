#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:28:29 2024

@authors: ramsonmunoz
"""

from scipy import sparse
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns


class HeatModel:

    def __init__(self, size, tempTop = 0, tempBot = 0, tempRight = 100, tempLeft = 100):
        
        # Initial Variables from Constructor
        self.size = size
        self.tempTop = tempTop
        self.tempBot = tempBot
        self.tempRight = tempRight
        self.tempLeft = tempLeft
        
        # Constructor Function Calls
        self.designMatrix = self.create_Model()
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

        return finalMatrix.tocsr()

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

        # Changing the top and bottom row values to -1*tempLeft and -1*tempRight
        for i in range(self.size):
            matrix[0, i] = -1*self.tempLeft
            matrix[-1, i] = -1*self.tempRight

            # If the first row hasn't been set by top and bottom, change it.
            # This gives priority to the left and right setters for this matrix.
            if matrix[i, 0] == 0:
                matrix[i, 0] = -1*self.tempTop
            
            # Repeated for right
            if matrix[i, -1] == 0:
                matrix[i, -1] = -1*self.tempBot

        # Flattening the matrix into a column vector
        return matrix.T.flatten().reshape(-1, 1)

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
        fig, (ax, ax_bar) = plt.subplots(figsize=(12, 8),
                                         ncols=2,
                                         constrained_layout=True,
                                         gridspec_kw={"width_ratios": [0.8, 0.2]})

        # Creating a heatmap of the design matrix
        sns.heatmap(matrix, ax=ax)
        ax.set_title("Design Matrix")

        # Creating a visual representation of the vector b
        sns.heatmap(self.b, ax=ax_bar)
        ax_bar.set_title("Vector b")

        # Adding a title to the figure
        fig.suptitle("Visual Representation of Design Matrix and Vector b",
                     fontsize="xx-large")

    def GaussSeidel(self, nIter=100, tol=1e-8, stop=True, plot=False, x0=1):
        """
        Solves the linear system Ax = b using the Gauss Seidel iterative method.

        Parameters:
        - nIter (int): Maximum number of iterations allowed.
        - tol (float): Convergence tolerance for the iterative method.
        - stop (boolean): True allows the method to end if tolerance is reached
        - plot (boolean): True will plot the padded solution matrix once solved

        Returns:
        - x1 (np.array): The solution vector after convergence or maximum iterations.
        - count (int): The number of iterations performed.
        - tols (np.array): Recorded tolerance values for each iteration 
            (0: Infinity norm, 1: L2 norm).
        """
        # Starting with an initial vector of ones
        x = np.ones(self.size**2) * x0
        # Creating a copy for future iterations
        x1 = np.copy(x)
        # Storing the design matrix and vector b from the class constructor
        A = self.designMatrix
        b = self.b

        # Storing the Diagonals in A for easier access later
        diags = A.diagonal()

        # Initializing a counter and a loss array
        count = 0
        tols = np.zeros((2, nIter))

        # Beginning outer loop of nIters
        for i in range(nIter):

            # Inner loop to generate x_k+1
            for j in range(x.shape[0]):
                # Slicing through the sparse matrix for non zero values
                rowstart = A.indptr[j]
                rowend = A.indptr[j+1]
                # Taking A[j].T @ x
                Ajx = A.data[rowstart:rowend] @ x[A.indices[rowstart:rowend]]

                # Generating each element for x_k+1
                x1[j] = x[j] + ((b[j] - Ajx) / diags[j])

            # Calculating the difference between the prior and current approximation
            x_x1 = x1 - x

            # Calculating and appending the normalized max norm
            infNorm = np.linalg.norm(x_x1, ord=np.inf) / \
                np.linalg.norm(x1, ord=np.inf)
            tols[0, i] = infNorm

            # Calculating and appending the normalized l2 norm
            l2Norm = np.linalg.norm(x_x1) / np.linalg.norm(x1)
            tols[1, i] = l2Norm

            # Incrementing count
            count += 1

            # Copying the current into the prior for the next iteration
            x = np.copy(x1)

            if stop:
                if infNorm < tol:
                    break

        if plot:
            self.visualizeHeatDistribution(x1)

        return x1, count, tols[:,:count]

 
    def solveJacobi(self, nIter = 100, tol=1e-8, stop=True, plot=False, x0=1):
        """
        Solves the linear system Ax = b using the Jacobi iterative method.

        Parameters:
        - nIter (int): Maximum number of iterations allowed.
        - tol (float): Convergence tolerance for the iterative method.
        - stop (boolean): True allows the method to end if tolerance is reached
        - plot (boolean): True will plot the padded solution matrix once solved

        Returns:
        - x1 (np.array): The solution vector after convergence or maximum iterations.
        - count (int): The number of iterations performed.
        - tols (np.array): Recorded tolerance values for each iteration 
            (0: Ax - b Error, 1: Infinity norm, 2: L2 norm).
        """
        # Retrieve the system matrix A representing coefficients in Ax = b.
        A = self.designMatrix
        # Flatten the right-hand side vector b to ensure it is a 1D array for calculations.
        b = self.b.flatten()
        # Determine the number of variables/equations in the system.
        n = len(b)
        # Initialize the solution vector x with zeros as the initial guess.
        # In the Jacobi method, the initial guess can be any vector; zeros are commonly used.
        x = np.ones(n) * x0

        # Extract the diagonal elements of matrix A.
        diag = A.diagonal()
        # Compute the inverse of the diagonal elements (D^-1), where D is the diagonal matrix of A.
        # This will be used to isolate x in the iteration formula.
        inv_diag = 1.0 / diag

        # Initialize an array to store the convergence tolerances (error, infinity norm and L2 norm) for each iteration.
        tols = np.zeros((3, nIter))
        # Initialize the iteration counter.
        count = 0

        # Begin the iterative process for up to 'maxiter' iterations.
        for i in range(nIter):
            # Compute the matrix-vector product A * x, where x is the current approximation.
            Ax = A @ x
            # Update the solution vector x using the Jacobi iteration formula:
            # x_new = x + D^-1 * (b - A * x)
            x1 = x + inv_diag * (b - Ax)

            # Calculate the residual error as the infinity norm of (A * x1 - b).
            # This represents how close the current solution is to satisfying the system Ax = b.
            error = np.linalg.norm(A.dot(x) - b, ord=np.inf)
            # Record the residual error for this iteration.
            tols[0,i] = error

            # Compute the difference between the new and previous solution vectors.
            x1_x = x1 - x

            # Calculate the normalized infinity norm of the difference between iterations.
            # This measures the maximum relative change in the solution vector.
            # The small epsilon (1e-10) prevents division by zero.
            infNorm = np.linalg.norm(x1_x, ord=np.inf) / (np.linalg.norm(x1, ord=np.inf) + 1e-10)
            # Store the infinity norm in the 'tols' array for convergence analysis.
            tols[1, i] = infNorm

            # Calculate the normalized L2 norm of the difference between iterations.
            # This provides an overall measure of convergence across all variables.
            l2Norm = np.linalg.norm(x1_x) / (np.linalg.norm(x1) + 1e-10)
            # Store the L2 norm in the 'tols' array.
            tols[2, i] = l2Norm

            # Increment the iteration counter.
            count += 1

            if stop:
                # Check for convergence: if the infinity norm is less than the tolerance 'tol', the method has converged.
                if infNorm < tol:
                    break

            # Update x for the next iteration.
            x = x1.copy()

        if plot:
            self.visualizeHeatDistribution(x1)
        
        # If convergence was not achieved within 'maxiter' iterations,
        # return the last computed solution and associated iterCOUNT and tolerances.
        return x1, count, tols[:, :count]

    def visualizeHeatDistribution(self, U, title='Heat Distribution', pad_width = 1):
        """
        Visualizes the heat distribution as a heatmap.

        Parameters:
        - U (np.array): The solution vector representing the heat values at each grid point.
        - title (str): Title of the heatmap plot.
        - pad_width (int): Size of padding for plot.
        """

        # Retrieve the grid size (number of points along one axis).
        n = self.size
        # Reshape the solution vector U into a 2D matrix representing the grid.
        result_matrix = U.reshape((n, n))

        # Apply padding to the grid with the specified boundary temperatures.
        # This adds a border around 'result_matrix' representing the fixed temperatures at the boundaries.
        result_padded = np.pad(
            result_matrix,
            pad_width = pad_width,
            mode='constant',
            constant_values=((self.tempTop, self.tempBot), (self.tempLeft, self.tempRight))
        )
        # Create a new figure for the plot with specified size.
        plt.figure(figsize=(10, 8))
        # Display the heat distribution as an image (heatmap).
        # 'cmap' specifies the color map, and 'interpolation' controls how pixel values are interpolated.
        plt.imshow(result_padded, cmap='coolwarm', interpolation='nearest')
        # Add a colorbar to the plot to indicate temperature values.
        plt.colorbar()
        # Set the title of the plot.
        plt.title(title)
        # Label the x-axis.
        plt.xlabel('X-axis')
        # Label the y-axis.
        plt.ylabel('Y-axis')
        # Display the plot.
        plt.show()
