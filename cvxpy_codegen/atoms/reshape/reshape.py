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
import numpy as np

# TODO remove for cvxpy 1.0:
from cvxpy.lin_ops.lin_op import LinOp

def atomdata_reshape(expr, arg_data):

    ## TODO remove for cvxpy 1.0:
    #print(expr)
    #if isinstance(expr, LinOpData):
    #    m_new, n_new = expr.size
    #else:
    #    m_new, n_new = expr.get_data()

    m_new, n_new = expr.size

    
    m = arg_data[0].size[0]
    n = arg_data[0].size[1]

    sparsity = reshape(arg_data[0].sparsity, m_new, n_new)

    return AtomData(expr, arg_data,
                    macro_name = 'reshape',
                    sparsity = sparsity,
                    work_int = m_new,
                    work_float = m_new,
                    data = (m_new, n_new))



def coeffdata_reshape(linop, args, var):
    return LinOpCoeffData(linop, args, var,
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
