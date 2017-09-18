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

from cvxpy_codegen.object_data.const_expr_data import ConstExprData
from cvxpy_codegen.object_data.coeff_data import CoeffData
from cvxpy_codegen.object_data.aff_atom_data import AffAtomData
import scipy.sparse as sp

class TransposeData(AffAtomData):

    def get_atom_data(self, expr, arg_data, arg_pos):
        sparsity = sp.csr_matrix(arg_data[0].sparsity.T)
        return ConstExprData(expr, arg_data,
                             sparsity = sparsity,
                             macro_name = "transpose")


    def get_coeff_data(self, args, var):
        rows, cols = self.shape
        # Create a sparse matrix representing the transpose.
        val_arr = []
        row_arr = []
        col_arr = []
        for col in range(cols):
            for row in range(rows):
                # Index in transposed coeff.
                row_arr.append(col*rows + row)
                # Index in original coeff.
                col_arr.append(row*cols + col)
                val_arr.append(1.0)
        P = sp.coo_matrix((val_arr, (row_arr, col_arr)),
                          (rows*cols, rows*cols)).tocsc()
        sparsity = sp.csr_matrix(P * args[0].sparsity, dtype='Bool')

        return CoeffData(self, args, var,
                         sparsity = sparsity,
                         macro_name = 'transpose_coeffs')
