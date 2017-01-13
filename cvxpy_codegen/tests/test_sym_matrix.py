"""
Copyright 2017 Nicholas Moehle

This file is part of CVXPY-CODEGEN.

CVXPY-CODEGEN is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY-CODEGEN is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY-CODEGEN.  If not, see <http://www.gnu.org/licenses/>.
"""

import unittest
import cvxpy_codegen.tests.utils as tu
import cvxpy_codegen as cg
import numpy as np
import cvxpy_codegen.linop_sym.sym_matrix as sym
import scipy.sparse as sp
 
m = 5
n = 6
p = 8
dens = .3
A_sparse  = sp.rand(m,n, dens)
A_sym = sym.as_sym_matrix(A_sparse)
A_dense = A_sparse.toarray()
A_param = sym.as_sym_matrix(cg.Parameter(m, n, value=A_sparse))

B_sparse  = sp.rand(n,p, dens)
B_sym = sym.as_sym_matrix(B_sparse)
B_dense = B_sparse.toarray()
B_param = sym.as_sym_matrix(cg.Parameter(n, p, value=B_sparse))

C_sparse  = sp.rand(m,n, dens)
C_sym = sym.as_sym_matrix(C_sparse)
C_dense = C_sparse.toarray()
C_param = sym.as_sym_matrix(cg.Parameter(m, n, value=C_sparse))

alpha = .5
alpha_sym = sym.SymConst(.5)
alpha_symmat = sym.as_sym_matrix(.5)
alpha_dense = np.eye(1) * alpha
alpha_sparse = sp.csc_matrix(sp.eye(1)) * alpha


class TestSymMatrix(unittest.TestCase):
    EPS = 1e-5


    def assertAlmostEqualMatrices(self, A, B, eps=None):
        if eps == None:
            eps = self.EPS
        D =  abs(A-B)
        if sp.issparse(D):
            D = D.toarray()
        self.assertTrue(np.all(D <= eps))


    def test_symbolic_matrix_addition(self):

        AC = A_sparse+C_sparse

        # Multiply by symbolic.
        AsymCsym = A_sym + C_sym
        self.assertAlmostEqualMatrices(AsymCsym.value, AC)

        # Multiply by sparse.
        AsymCsparse = A_sym + C_sparse
        self.assertAlmostEqualMatrices(AsymCsparse.value, AC)
        AsparseCsym = C_sym.__radd__(A_sparse)
        self.assertAlmostEqualMatrices(AsparseCsym.value, AC)

        # Multiply by dense.
        AsymCdense = A_sym + C_dense
        self.assertAlmostEqualMatrices(AsymCdense.value, AC)
        AdenseCsym = A_dense + C_sym
        self.assertAlmostEqualMatrices(AdenseCsym.value, AC)


    def test_symbolic_matrix_matrix_multiply(self):
        AB = A_sparse*B_sparse

        # Multiply by symbolic.
        AsymBsym = A_sym * B_sym
        self.assertAlmostEqualMatrices(AsymBsym.value, AB)

        # Multiply by sparse.
        AsymBsparse = A_sym * B_sparse
        self.assertAlmostEqualMatrices(AsymBsparse.value, AB)
        AsparseBsym = B_sym.__rmul__(A_sparse)
        self.assertAlmostEqualMatrices(AsparseBsym.value, AB)

        # Multiply by dense.
        AsymBdense = A_sym * B_dense
        self.assertAlmostEqualMatrices(AsymBdense.value, AB)
        AdenseBsym = A_dense * B_sym
        self.assertAlmostEqualMatrices(AdenseBsym.value, AB)


    def test_diag(self):
        const_n1 = sp.rand(n,1, dens)
        true_value = sp.diags(np.squeeze(const_n1.toarray(), 1))
        test_value = sym.diag(sym.as_sym_matrix(const_n1)).value
        self.assertAlmostEqualMatrices(test_value, true_value)
        

    def test_kron(self):
        const_mn = sp.rand(m, n, dens)
        const_np = sp.rand(m, p, dens)
        true_value = sp.csc_matrix(sp.kron(const_mn, const_np))
        test_value = sym.kron(sym.as_sym_matrix(const_mn),
                              sym.as_sym_matrix(const_np)).value
        self.assertAlmostEqualMatrices(test_value, true_value)
        

    def test_kron(self):
        const_mn = sp.rand(m, n, dens)
        true_value = sp.csc_matrix(const_mn.T)
        test_value = sym.transpose(sym.as_sym_matrix(const_mn)).value
        self.assertAlmostEqualMatrices(test_value, true_value)
        

    def test_reciprocals(self):
        const_mn = sp.csc_matrix(sp.rand(n,n, dens))
        new_data = 1.0 / const_mn.data
        true_value = sp.csc_matrix((new_data, const_mn.indices, const_mn.indptr),
                                   shape = const_mn.shape)
        test_value = sym.reciprocals(sym.as_sym_matrix(const_mn)).value
        self.assertAlmostEqualMatrices(test_value, true_value)
        


    def test_symbolic_scalar_matrix_multiply(self):
        alphaAsparse = alpha*A_sparse

        # Multiply by symbolic.
        alphamatAsym = alpha_symmat * A_sym
        self.assertAlmostEqualMatrices(alphamatAsym.value, alphaAsparse)

        # Multiply by sparse.
        Asymalphasparse = A_sym * alpha_sparse
        self.assertAlmostEqualMatrices(Asymalphasparse.value, alphaAsparse)
        alphasparseAsym = A_sym.__rmul__(alpha_sparse)
        self.assertAlmostEqualMatrices(alphasparseAsym.value, alphaAsparse)

        # Multiply by dense.
        alphadenseAsym = alpha_dense * A_sym
        self.assertAlmostEqualMatrices(alphadenseAsym.value, alphaAsparse)
        Asymalphadense = A_sym * alpha_dense
        self.assertAlmostEqualMatrices(Asymalphadense.value, alphaAsparse)

        # Multiply by scalar.
        alphaAsym = alpha * A_sym
        self.assertAlmostEqualMatrices(alphaAsym.value, alphaAsparse)
        Asymalpha = A_sym * alpha
        self.assertAlmostEqualMatrices(Asymalpha.value, alphaAsparse)



    # TODO def test_zero_pad(self):
    # TODO def as_sym_matrix(self):
    # TODO def get_scalar(self):



    def test_value(self):
        self.assertAlmostEqualMatrices(A_sym.value, A_sparse)
        self.assertAlmostEqualMatrices(A_param.value, A_sparse)
        self.assertAlmostEqualMatrices(A_sym.value, A_dense)
        self.assertEqual(A_param.nnz, 
                         A_sparse.shape[0]*A_sparse.shape[1]) # All params are dense.
        self.assertEqual(A_sym.nnz, A_sparse.nnz) # Constant matrix is sparse.
        self.assertEqual((A_param.m, A_param.n), A_sparse.shape)
        




if __name__ == '__main__':
    unittest.main()
