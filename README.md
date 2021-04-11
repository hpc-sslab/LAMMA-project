# LAMMA-project


1. DenseMatrix
  - Implementations of PDGEMM, LU, QR, and Cholesky routines based on ScaLAPACK library are introduced. Instructions how to build and their prerequisites are explained each CPU architectures of KNL and SKL
 


2. SparseMatrix
  - Implementations of mutiphysics, multiscale phenomena, that is, Darcy flow with heterogeneous permeability tensor,
steady state Stokes flow, Efficient 2,3 point quadrature rules on quadrilaterals with nonconforming finite element, and generalized multiscale finite element method for multiscale elliptic problems are listed. In each problem, the resulting discrete linear system is solved by iterative solver(CG, MINRES, AMG etc).


3. Optimization
  - Four implementations with optimization techniques are listed. Instructions to run the applications are explained.