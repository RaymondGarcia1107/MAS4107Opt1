# Items needed
* PPT
* Code file

## Code File
1. Functions to construct a n<sup>2</sup> x n<sup>2</sup> sparse matrix
    * Create a block matrix of n x n blocks with -4 in the diagonal and 1 above and below the diagonal. All other values should be 0
    * Create a block matrix of n x n blocks with 1 in the diagonal
    * Combine the two block matrices to an n<sup>2</sup> x n<sup>2</sup> matrix A
    * Visualize and check if the matrix was constructed correctly automatically
    * Construct a vector b of size n<sup>2</sup> x 1
2. Show the diagonal Matrix D from Matrix A and take its inverse. 
3. Solve using the Jacobi Method
    * Generate an iterator function that takes in a matrix, a vector, a tolerance (10^{-8}), and a max iterations.
    * Return the final solution to Ax = b.
4. Store the result vector x as a matrix U[i,j]
    * Rearrange the vector x as a matrix size n x n 
5. Present the final matrix U as a heatmap
    * Pad the matrix with values of 0 and 100 
 