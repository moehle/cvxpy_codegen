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

from cvxpy_codegen.object_data.atom_data import AtomData
from cvxpy_codegen.object_data.linop_coeff_data import LinOpCoeffData
import scipy.sparse as sp

def atomdata_transpose(expr, arg_data):
    sparsity = sp.csr_matrix(arg_data[0].sparsity.T)
    return AtomData(expr, arg_data,
                    sparsity = sparsity,
                    macro_name = "transpose")


def coeffdata_transpose(linop, args, var):
    rows, cols = linop.shape
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

    return LinOpCoeffData(linop, args, var,
                          sparsity = sparsity,
                          macro_name = 'transpose_coeffs')
