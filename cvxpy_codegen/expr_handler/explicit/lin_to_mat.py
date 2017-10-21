"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import cvxpy.lin_ops.lin_op as lo
import cvxpy.interface as intf
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sp_la
import cvxpy_codegen.linop_sym.sym_matrix as sym
from cvxpy_codegen.param.param_handler import CBP_TO_SPARSITY # TODO rm

# Utility functions for converting LinOps into matrices.

def flatten(matrix):
    """Converts the matrix into a column vector.

    Parameters
    ----------
    matrix :
        The matrix to flatten.
    """
    if isinstance(matrix, sym.SymMatrix):
        vec = matrix.as_vector()
    else:
        np_mat = intf.DEFAULT_INTF
        matrix = np_mat.const_to_matrix(matrix, convert_scalars=True)
        size = intf.size(matrix)
        return np_mat.reshape(matrix, (size[0]*size[1], 1))
    return vec

def get_coefficients(lin_op):
    """Converts a linear op into coefficients.

    Parameters
    ----------
    lin_op : LinOp
        The linear op to convert.

    Returns
    -------
    list
        A list of (id, coefficient) tuples.
    """
    # VARIABLE converts to a giant identity matrix.
    if lin_op.type == lo.VARIABLE:
        coeffs = var_coeffs(lin_op)
    #elif lin_op.type == lo.PARAM:
     #   coeffs = param_coeffs(lin_op)
    # Constants convert directly to their value.
    elif lin_op.type in CONSTANT_TYPES:
        coeffs = [(lo.CONSTANT_ID, mat.as_vector())]
    elif lin_op.type in TYPE_TO_FUNC:
        # A coefficient matrix for each argument.
        coeff_mats = TYPE_TO_FUNC[lin_op.type](lin_op)
        coeffs = []
        for coeff_mat, arg in zip(coeff_mats, lin_op.args):
            rh_coeffs = get_coefficients(arg)
            coeffs += mul_by_const(coeff_mat, rh_coeffs)
    else:
        raise Exception("Unknown linear operator '%s'" % lin_op.type)
    coeffs = [(c[0], sym.as_sym_matrix(c[1])) for c in coeffs] # All coeffs as SymMatrix
    return coeffs

def get_constant_coeff(lin_op):
    """Converts a linear op into coefficients and returns the constant term.

    Parameters
    ----------
    lin_op : LinOp
        The linear op to convert.

    Returns
    -------
    The constant coefficient or None if none present.
    """
    coeffs = get_coefficients(lin_op)
    for id_, coeff in coeffs:
        if id_ == lo.CONSTANT_ID:
            return coeff
    return None

def var_coeffs(lin_op):
    id_ = lin_op.data
    coeff = sp.eye(lin_op.size[0]*lin_op.size[1]).tocsc()
    return [(id_, coeff)]

def const_mat(lin_op):
    """Returns the matrix for a constant type.

    Parameters
    ----------
    lin_op : LinOp
        The linear op.

    Returns
    -------
    A numerical constant.
    """
    if lin_op.type == lo.PARAM:
        name = lin_op.data.name()
        if name in CBP_TO_SPARSITY.keys():
            sprs = CBP_TO_SPARSITY[name]
            sprs = sp.csc_matrix(sprs)
            coeff = sym.as_sym_matrix(lin_op.data, sparsity=sprs)
        else:
            coeff = sym.as_sym_matrix(lin_op.data)
    elif lin_op.type in [lo.SCALAR_CONST, lo.DENSE_CONST, lo.SPARSE_CONST]:
        coeff = lin_op.data
    return coeff


# A list of all the linear operator types for constants.
CONSTANT_TYPES = [lo.PARAM, lo.SCALAR_CONST, lo.DENSE_CONST, lo.SPARSE_CONST]

# A map of LinOp type to the coefficient matrix function.
TYPE_TO_FUNC = {
    lo.PROMOTE: promote_mat,
    lo.NEG: neg_mat,
    lo.MUL: mul_mat,
    lo.RMUL: rmul_mat,
    lo.MUL_ELEM: mul_elemwise_mat,
    lo.DIV: div_mat,
    lo.SUM_ENTRIES: sum_entries_mat,
    lo.TRACE: trace_mat,
    lo.INDEX: index_mat,
    lo.TRANSPOSE: transpose_mat,
    lo.RESHAPE: lambda lin_op: [1],
    lo.SUM: lambda lin_op: [1]*len(lin_op.args),
    lo.DIAG_VEC: diag_vec_mat,
    lo.DIAG_MAT: diag_mat_mat,
    lo.UPPER_TRI: upper_tri_mat,
    lo.CONV: conv_mat,
    lo.KRON: kron_mat,
    lo.HSTACK: lambda lin_op: stack_mats(lin_op, False),
    lo.VSTACK: lambda lin_op: stack_mats(lin_op, True),
}
