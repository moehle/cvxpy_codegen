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
import numpy as np


class ReshapeData(AffAtomData):

    def get_atom_data(self, expr, arg_data, arg_pos):

        if len(expr.shape) == 2:
            shape = expr.shape
        elif len(expr.shape) == 1:
            shape = (expr.shape[0], 1)
        elif len(expr.shape) == 0:
            shape = (1,1)
        else:
            raise Exception("Cannot reshape to array"
                            "with more than two dimensions.")

        sparsity = reshape(arg_data[0].sparsity, shape[0], shape[1])

        return ConstExprData(expr, arg_data,
                             macro_name = 'reshape',
                             sparsity = sparsity,
                             work_int = shape[0],
                             work_float = shape[0],
                             data = shape)



    def get_coeff_data(self, args, var):
        return CoeffData(self, args, var,
                         sparsity = args[0].sparsity,
                         inplace = True,
                         macro_name = 'reshape_coeffs')



def reshape(A, m_new,n_new):
    m,n = A.shape

    A = sp.coo_matrix(A)
    I = A.row
    J = A.col
    V = A.data

    I_new = []
    J_new = []
    for (i,j) in zip(I,J):
        idx = i + j*m
        i_new = idx % m_new
        J_new += [(idx - i_new) / m_new]
        I_new += [i_new]

    B = sp.csr_matrix(sp.coo_matrix((V, (I_new, J_new)), (m_new, n_new)))

    return sp.csr_matrix(sp.coo_matrix((V, (I_new, J_new)), (m_new, n_new)))
