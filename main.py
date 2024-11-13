
import numpy as np
from scipy.sparse import diags, identity, lil_matrix, csr_matrix
import matplotlib.pyplot as plt
import time


np.set_printoptions(threshold = np.inf) #to be able to print dense matrices without truncating them

""""
##  We have to solve the laplace equation where we are interested in the interior points given the boundary conditions:
    ## U * [(d^2/dx^2) + (d^2/dy^2)] = 0
    ## U(X_i, Y_j) = U_i,j
    ## We can initially simplify to a 3 x 3 grid where we are only interested in the ONE interior point:
        ## Our interior point of interest is U_i,j; using the finite difference approximation when can get the second derivative for both X and Y.
            ## (d^2U / dx^2) = [ (U_i+1, j) - [2(U_i, j)] + (U_i-1, j) ] / (dx)^2
            ## (d^2U / dy^2) = [ (U_i, j+1) - [2(U_i, j)] + (U_i, j-1) ] / (dy)^2
            ## We can simplify by assuming dx = dy and since the Laplace equation = 0,
            ## we can multiply both sides by (dx)^2 or (dy)^2 since they are equal and clear the denominator.
            ## U * [(d^2/dx^2) + (d^2/dy^2)] = 0 RESTATING THE LAPLACE EQUATION
##--------------## [-4(U_i, j)] + (U_i+1, j) + (U_i-1, j) + (U_i, j+1)) + (U_i, j-1) = 0-------------------------------##

## DISCRETE LAPLACE EQUATION : [-4(U_i, j)] + (U_i+1, j) + (U_i-1, j) + (U_i, j+1)) + (U_i, j-1) = 0
    ## We now need to convert this system of linear equations into a Matrix equation Ax = b.
        ## Firstly we would need to have our unknowns as a column vector :  ##   [U_i, j]
                                                                            ##   [U_i, j+1]
                                                                            ##   [U_i, j+2]
                                                                            ##   [   ... ]
                                                                            ##   [U_i+1, j]
                                                                            ##   [U_i+1, j+1]
                                                                            ##   [U_i+1, j+2]
                                                                            ##   [   ...  ]
                                                                            ##   [U_n, n]
        ## Secondly we would need to find the Matrix A:
            ## We need to convert the (i, j) coordinate system to a k coordinate system where:
                ## (i,j) --> k = i + (j - 1)nX
                ## Then the Discrete Laplace Equation becomes [-4(U_k)] + (U_[k + 1]) + (U_[k - 1]) + (U_[k + nX]) + (U_[k - nX])
                ## We also need to take into consideration the boundary conditions:
                    ## Bottom = all values where j = 1
                    ## Top = all values where j = nY
                    ## Left = all values where i = 0
                    ## Right = all values where i = nX
                    """

def create_large_diagonal_matrix(n):
    """Creates a block diagonal matrix where each block is n x n with specified patterns."""
    diagonals = [-4 * np.ones(n), np.ones(n-1), np.ones(n-1)]
    offsets = [0, -1, 1]  # Main diagonal and the two off-diagonals
    block = diags(diagonals, offsets, shape=(n, n), format='csr')

    return block

def create_large_identity_matrix_off_diagonal(n):
    """Creates an identity matrix for the off-diagonal blocks."""
    return identity(n, format='csr')


def construct_matrix_A(n):
    """
    Constructs the large matrix A using diagonal and off-diagonal blocks.
    Additionally, extracts and displays the diagonal matrix D (as a sparse matrix)
    which contains only the diagonal entries of A, and calculates its inverse.
    """
    # Initialize A as a sparse LIL (List of Lists) matrix for efficient construction
    A = lil_matrix((n**2, n**2))

    # Create the diagonal blocks for A
    diagonal_block = create_large_diagonal_matrix(n)
    # Create the off-diagonal blocks for A
    off_diagonal_block = create_large_identity_matrix_off_diagonal(n)

    # Place diagonal blocks in the large matrix A
    for i in range(n):
        # Assign the diagonal block to the appropriate location in A
        A[i * n:(i + 1) * n, i * n:(i + 1) * n] = diagonal_block

    # Place off-diagonal blocks in A
    for i in range(n - 1):
        # Upper off-diagonal block
        A[i * n:(i + 1) * n, (i + 1) * n:(i + 2) * n] = off_diagonal_block
        # Lower off-diagonal block
        A[(i + 1) * n:(i + 2) * n, i * n:(i + 1) * n] = off_diagonal_block

    # Convert A to CSR format for efficient arithmetic operations
    A_csr = A.tocsr()

    # --- Extract the diagonal matrix D (as a sparse matrix) ---

    # Extract the diagonal entries of A
    diag_entries = A_csr.diagonal()
    # Create the diagonal matrix D using the diagonal entries
    D = diags(diag_entries, offsets=0, shape=A_csr.shape, format='csr')

    # Display the diagonal matrix D (non-zero entries)
    print("Diagonal matrix D (non-zero entries):")
    print(D) ## Print CSR
    #print(D.toarray()) #Print Numpy Matrix

    # --- Calculate the inverse of the diagonal matrix D ---

    # Check for zeros on the diagonal to prevent division by zero
    if np.any(diag_entries == 0):
        raise ZeroDivisionError("Zero diagonal element encountered in D; cannot invert.")

    # Compute the inverse of the diagonal entries
    inv_diag_entries = 1 / diag_entries
    #= Create the inverse diagonal matrix D_inv
    D_inv = diags(inv_diag_entries, offsets=0, shape=A_csr.shape, format='csr')

    # Display the inverse of the diagonal matrix D (non-zero entries)
    print("Inverse of the diagonal matrix D (non-zero entries):")
    print(D_inv) ## Print CSR
    #print(D_inv.toarray()) #Print Numpy Matrix

    # Return the constructed matrix A in CSR format
    visualize_matrix_A(A)
    print(f'Matrix A is diagonally dominant: {is_diagonally_dominant(A)}')

    return A_csr


def visualize_matrix_A(A):
    """
    Visualizes the sparse matrix A using matplotlib's plt.spy().

    Parameters:
    - A (scipy.sparse matrix): The sparse matrix to visualize.

    Returns:
    - None
    """
    # Create a new figure with a specified size (optional)
    plt.figure(figsize=(8, 8))

    # Use plt.spy() to visualize the sparsity pattern of A
    plt.spy(A, markersize=1)  # Adjust markersize for better visibility

    # Add title and labels (optional)
    plt.title('Visualization of Matrix A')
    plt.xlabel('Columns')
    plt.ylabel('Rows')

    # Display the plot
    plt.show()

def create_vector_b(n, top_temp, bottom_temp, left_temp, right_temp):
    """
    Creates the vector b for a grid of size n x n with specified boundary temperatures.

    Parameters:
    - n (int): The size of one dimension of the square grid.
    - top_temp (float): Temperature at the top boundary of the grid.
    - bottom_temp (float): Temperature at the bottom boundary of the grid.
    - left_temp (float): Temperature at the left boundary of the grid.
    - right_temp (float): Temperature at the right boundary of the grid.

    Returns:
    - b (np.array): The vector b of size n^2, initialized with boundary temperatures.
    """
    b = np.zeros(n**2)

    # Top and Bottom boundary conditions
    for i in range(n):
                        #We imagine starting from the bottom - up and from left - right
                        # Indices of i go from i to n-1
                        # This means  b[0], b[1], ... , b[n-1]  represent the first  n  elements of the vector  b , which correspond to the bottom boundary of the grid.
                            # The index  (n-1) * n  calculates the position of the first element in the last row of the grid.
                            # This row spans from  b[(n-1) \times n]  to  b[(n-1) \times n + (n-1)] , corresponding to the indices  n^2 - n  to  n^2 - 1 .
                             #### Note indices are shifted 1 as python starts at 0 ####
                        ## The operation sets the last  n  elements of the vector  b, which correspond to the top boundary


        b[i] = (-1) * top_temp  # Top row
        b[(n - 1) * n + i] = (-1) * bottom_temp  # Bottom row

    # Left and Right boundary conditions
    for j in range(n):
        ## Everytime the loop iterates n times it sets every Nth element of b to left temp, starting at 0 and (only if it has not been set my top/bottom loop)
        # Correctly apply the left and right temperatures
        if b[j * n] == 0:  # Only set left if not already set by top/bottom
            b[j * n] = (-1) * left_temp
        ## Everytime the loop iterates it sets the last value of the iteration to right temp (only if it has not been set my top/bottom loop).
        if b[(j + 1) * n - 1] == 0:  # Only set right if not already set by top/bottom
            b[(j + 1) * n - 1] = (-1) * right_temp

    return b
##--------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------
def is_diagonally_dominant(A):
    """
    Checks if the sparse matrix A is diagonally dominant.

    Parameters:
    - A (scipy.sparse matrix): The matrix to check.

    Returns:
    - is_dd (bool): True if A is diagonally dominant, False otherwise.
    """


    # Ensure A is in CSR format for efficient row access
    if not isinstance(A, csr_matrix):
        A = A.tocsr()

    # Get the absolute values of the data
    abs_A = A.copy()
    abs_A.data = np.abs(abs_A.data)

    # Extract diagonal elements
    diag = abs_A.diagonal()

    # Initialize a boolean array to store the result for each row
    is_dd_row = np.zeros(A.shape[0], dtype=bool)

    for i in range(A.shape[0]):
        # Sum of non-diagonal absolute values in row i
        row_start = A.indptr[i]
        row_end = A.indptr[i + 1]
        row_indices = A.indices[row_start:row_end]
        row_data = abs_A.data[row_start:row_end]

        # Exclude the diagonal element from the sum
        off_diag_sum = np.sum(row_data[row_indices != i])

        # Diagonal element
        diag_i = diag[i]

        # Check for diagonal dominance in row i
        if diag_i >= off_diag_sum:
            is_dd_row[i] = True
        else:
            is_dd_row[i] = False

    # The matrix is diagonally dominant if all rows satisfy the condition
    is_dd = np.all(is_dd_row)
    return is_dd

def solveJacobi(A, b, tol=1e-8, maxiter=100, plot=False):
    """
    Solves the linear system Ax = b using the Jacobi method with sparse matrices.

    Parameters:
    - A (scipy.sparse matrix): Coefficient matrix in CSR format.
    - b (np.array): Right-hand side vector.
    - tol (float): Tolerance for convergence.
    - maxiter (int): Maximum number of iterations.
    - plot (bool): If True, plots the error against iteration count.

    Returns:
    - x (np.array): Solution vector.
    """

    n = len(b)  # Number of equations/variables in the system
    x = np.zeros(n)  # Initialize the solution vector x with zeros

    diag = A.diagonal()  # Extract the diagonal elements of A efficiently without converting to dense format

    # Check for zeros on the diagonal to prevent division by zero
    if np.any(diag == 0):
        raise ZeroDivisionError("Zero diagonal element encountered in Jacobi method.")

    if (is_diagonally_dominant(A) == False):
        raise ZeroDivisionError("The function is not diagonally dominant.")


    inv_diag = 1.0 / diag  # Compute the inverse of the diagonal elements for D^{-1}, since D is diagonal then Dinv - 1/ D

    errors = []  # List to store errors at each iteration

    for iter_count in range(maxiter):
        Ax = A.dot(x)  # Compute the matrix-vector product A * x

        # Update the solution vector x using the Jacobi iteration formula
        x_new = x + inv_diag * (b - Ax)

        # Compute error as the residual norm ||A x_new - b||
        error = np.linalg.norm(A.dot(x_new) - b, ord=np.inf)
        errors.append(error)

        # Check for convergence: if the residual norm is less than the tolerance, stop iterating
        if error < tol:
            if plot:
                plt.figure()
                plt.semilogy(range(1, iter_count + 2), errors)
                plt.xlabel('Iteration')
                plt.ylabel('Absolute Error')
                plt.title('Jacobi Method Residual Convergence')
                plt.grid(True)
                plt.show()
            return x_new  # Converged solution

        x = x_new  # Update x for the next iteration

    if plot:
        plt.figure()
        plt.semilogy(range(1, maxiter + 1), errors)
        plt.xlabel('Iteration')
        plt.ylabel('Absolute Error')
        plt.title('Jacobi Method Residual Convergence')
        plt.grid(True)
        plt.show()

    return x  # Return the last computed solution if convergence was not reached within maxiter iterations

def GaussSeidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """
    Solves the linear system Ax = b using the Gauss-Seidel method with sparse matrices.

    Parameters:
    - A (scipy.sparse matrix): Coefficient matrix in CSR format.
    - b (np.array): Right-hand side vector.
    - tol (float): Tolerance for convergence.
    - maxiter (int): Maximum number of iterations.
    - plot (bool): If True, plots the error against iteration count.

    Returns:
    - x (np.array): Solution vector.
    """
    n = len(b)  # Number of equations/variables in the system
    x = np.zeros(n)  # Initialize the solution vector x with zeros

    diag = A.diagonal()  # Extract the diagonal elements of A

    # Check for zeros on the diagonal to prevent division by zero
    if np.any(diag == 0):
        raise ZeroDivisionError("Zero diagonal element encountered in Gauss-Seidel method.")

    A_csr = A.tocsr()  # Ensure A is in CSR format for efficient row-wise operations

    errors = []  # List to store errors at each iteration

    for iter_count in range(maxiter):

        for i in range(n):
            # Get the start and end indices of non-zero elements in row i
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]

            # Extract the column indices and data values of the non-zero elements in row i
            indices = A_csr.indices[row_start:row_end]
            data = A_csr.data[row_start:row_end]

            sum_ = 0.0  # Initialize the sum of off-diagonal elements for the i-th equation

            # Iterate over the non-zero elements in row i
            for idx, j in enumerate(indices):
                if j != i:
                    sum_ += data[idx] * x[j]  # Accumulate the sum using the latest values of x

            # Update the i-th variable using the Gauss-Seidel formula
            x[i] = (b[i] - sum_) / diag[i]

        # Compute error as the residual norm ||A x - b||
        error = np.linalg.norm(A.dot(x) - b, ord=np.inf)
        errors.append(error)

        # Check for convergence: if the residual norm is less than the tolerance, stop iterating
        if error < tol:
            if plot:
                plt.figure()
                plt.semilogy(range(1, iter_count + 2), errors)
                plt.xlabel('Iteration')
                plt.ylabel('Absolute Error')
                plt.title('Gauss-Seidel Method Residual Convergence')
                plt.grid(True)
                plt.show()
            return x  # Converged solution

    if plot:
        plt.figure()
        plt.semilogy(range(1, maxiter + 1), errors)
        plt.xlabel('Iteration')
        plt.ylabel('Absolute Error')
        plt.title('Gauss-Seidel Method Residual Convergence')
        plt.grid(True)
        plt.show()

    return x  # Return the last computed solution if convergence was not reached within maxiter iterations

def visualize_heat_distribution(U, n, title, initempTop, initempBottom, initempLeft, initempRight):
    """Visualizes the heat distribution as a heatmap."""
    result_matrix = U.reshape((n, n))

    # Define padding values and the number of rows/columns to pad
    pad_width = 1  # Padding width constant (representing the border), hence 1

    top_pad = initempTop  # Value for top and bottom padding (init TEMP)
    bottom_pad = initempBottom
    left_pad = initempLeft  # Value for left and right padding (init TEMP)
    right_pad = initempRight

    # Applying padding to the matrix
    result_padded = np.pad(result_matrix, pad_width = pad_width,
                           mode = 'constant',
                           constant_values = ((top_pad, bottom_pad), (left_pad, right_pad)))
    plt.figure(figsize=(10, 8))
    plt.imshow(result_padded, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

def heatDistribution_Results(n = 50, top_temp = 100, bottom_temp = 100, left_temp = 0, right_temp = 0):
    A = construct_matrix_A(n)
    b = create_vector_b(n, top_temp, bottom_temp, left_temp, right_temp)

    start_time_jacobi = time.time()
    x_jacobi = solveJacobi(A, b, plot=True, maxiter=1000)
    end_time_jacobi = time.time()
    time_jacobi = end_time_jacobi - start_time_jacobi

    start_time_gs = time.time()
    x_gs = GaussSeidel(A, b, plot=True, maxiter=1000)
    end_time_gs = time.time()
    time_gs = end_time_gs - start_time_gs

    # Display the run-times
    print(f"Jacobi method took {time_jacobi:.6f} seconds.")
    print(f"Gauss-Seidel method took {time_gs:.6f} seconds.")

    visualize_heat_distribution(x_jacobi, n, "Heat Distribution: Jacobi Method", top_temp, bottom_temp, left_temp,
                                right_temp)
    visualize_heat_distribution(x_gs, n, "Heat Distribution: Gauss-Seidel Method", top_temp, bottom_temp, left_temp,
                                right_temp)


"""MAIN PROGRAM LAPLACE STEADY HEAT FLOW"""

heatDistribution_Results(left_temp=100, right_temp=100, top_temp=100, bottom_temp=0, n = 50)

"""MAIN PROGRAM LAPLACE STEADY HEAT FLOW"""
