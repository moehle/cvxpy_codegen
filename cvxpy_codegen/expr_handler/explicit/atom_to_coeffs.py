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

import cvxpy_codegen.expr_handler_sym.sym_matrix as sym
import scipy.sparse as sp
import numpy as np


from cvxpy.atoms.affine.binary_operators import DivExpression
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.trace import trace
from cvxpy.atoms.affine.hstack import Hstack
from cvxpy.atoms.affine.diag import diag_vec, diag_mat
from cvxpy.atoms.affine.index import index
from cvxpy.atoms.affine.multiply import multiply
from cvxpy.atoms.affine.promote import Promote
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.affine.transpose import transpose
from cvxpy.atoms.affine.vstack import Vstack
from cvxpy.atoms import *



def get_matrix_shape(expr):
    expr_shape = expr.shape
    if len(expr_shape) == 0:
        expr_shape = (1,1)
    elif len(expr_shape) == 1:
        expr_shape = (expr_shape[0],1)
    return expr_shape



def get_const_arg(expr, arg_coeffs):
    if expr.args[0].is_constant():
        var_idx = 1
        const_idx = 0
    else:
        var_idx = 0
        const_idx = 1
    constant = arg_coeffs[const_idx][0][1]
    return const_idx, var_idx, constant




def const_mat(expr, arg_coeffs):
    """Returns the matrix for a constant type.

    Parameters
    ----------
    lin_op : LinOp
        The linear op.

    Returns
    -------
    A numerical constant.
    """
    if expr.type == lo.PARAM:
        name = expr.data.name()
        if name in CBP_TO_SPARSITY.keys():
            sprs = CBP_TO_SPARSITY[name]
            sprs = sp.csc_matrix(sprs)
            coeff = sym.as_sym_matrix(expr.data, sparsity=sprs)
        else:
            coeff = sym.as_sym_matrix(expr.data)
    elif expr.type in [lo.SCALAR_CONST, lo.DENSE_CONST, lo.SPARSE_CONST]:
        coeff = expr.data
    return coeff


def mul_by_const(constant, rh_coeffs):
    """Multiplies a constant by a list of coefficients."""

    new_coeffs = []
    # Multiply all right-hand terms by the left-hand constant.
    #print(constant.shape)
    for (id_, coeff) in rh_coeffs:
        #print(coeff.size)
        new_coeffs.append((id_, coeff.__rmul__(constant)))
    #for c in new_coeffs:
    #    print
    #    print "Ap:"
    #    print(c[1].Ap)
    #    print
    #    print "Ai:"
    #    print(c[1].Ai)
    return new_coeffs


def mul_coeffs(coeffs, arg_coeff_list):
    new_coeffs = []
    # Multiply all right-hand terms by the left-hand constant.
    #print(constant.shape)
    for coeff, arg_coeffs in zip(coeffs, arg_coeff_list):
        new_coeffs += mul_by_const(coeff, arg_coeffs)
        #for c in new_coeffs:
        #    print
        #    print "Ap:"
        #    print(c[1].Ap)
        #    print
        #    print "Ai:"
        #    print(c[1].Ai)
    return new_coeffs

#def mul_coeffs(coeffs, arg_coeff_list):
#    new_coeffs = []
#    # Multiply all right-hand terms by the left-hand constant.
#    #print(constant.shape)
#    for coeff, arg_coeffs in zip(coeffs, arg_coeff_list):
#        for (id_, constant) in arg_coeffs:
#            #print(coeff.size)
#            new_coeffs += [(id_, coeff.__rmul__(constant))]
#        #for c in new_coeffs:
#        #    print
#        #    print "Ap:"
#        #    print(c[1].Ap)
#        #    print
#        #    print "Ai:"
#        #    print(c[1].Ai)
#    return new_coeffs



def add_mat(expr, arg_coeffs):
    coeffs = [1]*len(expr.args)
    return mul_coeffs(coeffs, arg_coeffs)


def reshape_mat(expr, arg_coeffs):
    coeffs = [1]
    return mul_coeffs(coeffs, arg_coeffs)


def sum_mat(expr, arg_coeffs):
    """Returns the coefficient matrix for SUM_ENTRIES linear op.

    Parameters
    ----------
    expr : LinOp
        The sum entries linear op.

    Returns
    -------
    list of NumPy matrix
        The matrix representing the sum_entries operation.
    """
    size = expr.args[0].size
    coeffs = [np.ones((1, size))]
    return mul_coeffs(coeffs, arg_coeffs)


def trace_mat(expr, arg_coeffs):
    """Returns the coefficient matrix for TRACE linear op.

    Parameters
    ----------
    expr : LinOp
        The trace linear op.

    Returns
    -------
    list of NumPy matrix
        The matrix representing the trace operation.
    """
    rows, _ = expr.args[0].shape
    mat = np.zeros((1, rows**2))
    for i in range(rows):
        mat[0, i*rows + i] = 1
    coeffs = [np.matrix(mat)]
    return mul_coeffs(coeffs, arg_coeffs)

def neg_mat(expr, arg_coeffs):
    """Returns the coefficient matrix for NEG linear op.

    Parameters
    ----------
    expr : LinOp
        The neg linear op.

    Returns
    -------
    list of SciPy CSC matrix
        The matrix representing the neg operation.
    """
    coeffs = [-sp.eye(expr.size).tocsc()]
    return mul_coeffs(coeffs, arg_coeffs)

def div_mat(expr, arg_coeffs):
    """Returns the coefficient matrix for DIV linear op.

    Assumes dividing by scalar constants.

    Parameters
    ----------
    expr : LinOp
        The div linear op.

    Returns
    -------
    list of SciPy CSC matrix
        The matrix representing the div operation.
    """
    const_idx, var_idx, divisor = get_const_arg(expr, arg_coeffs)

    const_idx, var_idx, constant = get_const_arg(expr, arg_coeffs)
    if isinstance(constant, sym.SymMatrix):
        mat = sym.diag(sym.reciprocals(constant.as_vector()))
    else:
        mat = sp.eye(expr.size)/divisor
    return mul_by_const(mat, arg_coeffs[var_idx])



def multiply_mat(expr, arg_coeffs):
    """Returns the coefficient matrix for MUL_ELEM linear op.

    Parameters
    ----------
    expr : LinOp
        The multiply linear op.

    Returns
    -------
    list of SciPy CSC matrix
        The matrix representing the multiply operation.
    """
    const_idx, var_idx, constant = get_const_arg(expr, arg_coeffs)
    if isinstance(constant, sym.SymMatrix):
        mat = sym.diag(constant.as_vector())
    else:
        mat = intf.from_2D_to_1D(flatten(constant))
    return mul_by_const(mat, arg_coeffs[var_idx])


def promote_mat(expr, arg_coeffs):
    """Returns the coefficient matrix for PROMOTE linear op.

    Parameters
    ----------
    expr : LinOp
        The promote linear op.

    Returns
    -------
    list of NumPy matrix
        The matrix for scalar promotion.
    """
    num_entries = expr.size
    coeffs = [np.ones((num_entries, 1))]
    return mul_coeffs(coeffs, arg_coeffs)



def mul_mat(expr, arg_coeffs):
    if expr.args[0].is_constant():
        return lmul_mat(expr, arg_coeffs)
    else:
        return rmul_mat(expr, arg_coeffs)
    


def lmul_mat(expr, arg_coeffs):
    """Returns the coefficient matrix for MUL linear op.

    Parameters
    ----------
    expr : LinOp
        The mul linear op.

    Returns
    -------
    list of SciPy CSC matrix or scalar.
        The matrix for the multiplication on the left operator.
    """
    mat_shape = get_matrix_shape(expr.args[0])
    expr_shape = get_matrix_shape(expr.args[1])
    if len(expr.args[0].shape) == 1:
        if expr.args[0].shape[0] == expr.args[1].shape[0]:
            mat_shape = (mat_shape[1], mat_shape[0])
        else:
            raise ValueError("Incompatible dimensions.")

    # Leftmost arg, first (and only) coeff (corresponding to CONST_ID),
    # we want the coeff, not the id:
    constant = arg_coeffs[0][0][1]

    if isinstance(constant, sym.SymMatrix):
        if constant.shape != (1,1):
            constant = constant.reshape_vec(*mat_shape)
            constant = sym.block_diag(expr_shape[1]*[constant])
    return mul_by_const(constant, arg_coeffs[1])



def rmul_mat(expr, arg_coeffs):
    """Returns the coefficient matrix for RMUL linear op.

    Parameters
    ----------
    expr : LinOp
        The rmul linear op.

    Returns
    -------
    list of SciPy CSC matrix or scalar.
        The matrix for the multiplication on the right operator.
    """
    mat_shape = get_matrix_shape(expr.args[1])
    expr_shape = get_matrix_shape(expr.args[0])
    if len(expr.args[0].shape) == 1:
        if expr.args[0].shape[0] == expr.args[1].shape[0]:
            expr_shape = (expr_shape[1], expr_shape[0])
        else:
            raise ValueError("Incompatible dimensions.")

    # Rightmost arg, first (and only) coeff (corresponding to CONST_ID),
    # and we want the coeff, not the id:
    constant = arg_coeffs[1][0][1]

    if isinstance(constant, sym.SymMatrix):
        sym_eye = sym.as_sym_matrix(sp.csc_matrix(sp.eye(expr_shape[0])))
        constant = constant.reshape_vec(*mat_shape)
        constant = sym.kron(sym.transpose(constant), sym_eye)
    else:
        # Scalars don't need to be replicated.
        if not intf.is_scalar(constant):
            # Matrix is the kronecker product of constant.T and identity.
            # Each column in the product is a linear combination of the
            # columns of the left hand multiple.
            constant = sp.kron(constant.T, sp.eye(expr.size[0])).tocsc()
    return mul_by_const(constant, arg_coeffs[0])

def index_mat(expr, arg_coeffs):
    """Returns the coefficient matrix for indexing.

    Parameters
    ----------
    expr : LinOp
        The index linear op.

    Returns
    -------
    list of SciPy CSC matrix
        The matrix for the index/slice operation.
    """
    # Define behavior for each case.
    slices = expr.get_data()[0]
    if len(expr.args[0].shape) == 1:
        slices = (slices[0], slice(0, 1, 1))
        rows = expr.args[0].shape[0]
        cols = 1
    if len(expr.args[0].shape) == 2:
        if len(slices) == 1:
            slices = (slices[0], slice(0, None, 1))
        rows, cols = expr.args[0].shape

    row_selection = range(rows)[slices[0]]
    col_selection = range(cols)[slices[1]]
    # Construct a coo matrix.
    val_arr = []
    row_arr = []
    col_arr = []
    counter = 0
    for col in col_selection:
        for row in row_selection:
            val_arr.append(1.0)
            row_arr.append(counter)
            col_arr.append(col*rows + row)
            counter += 1
    block_rows = np.prod(expr.shape)
    block_cols = rows*cols
    #return [sp.coo_matrix((val_arr, (row_arr, col_arr)),
    #                      (block_rows, block_cols)).tocsc()], expr.args
    mat = sp.coo_matrix((val_arr, (row_arr, col_arr)), (block_rows, block_cols)).tocsc()
    coeffs = [mat]
    return mul_coeffs(coeffs, arg_coeffs)


def transpose_mat(expr, arg_coeffs):
    """Returns the coefficient matrix for TRANSPOSE linear op.

    Parameters
    ----------
    expr : LinOp
        The transpose linear op.

    Returns
    -------
    list of SciPy CSC matrix
        The matrix for the transpose operation.
    """
    rows, cols = expr.shape
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

    coeffs = [sp.coo_matrix((val_arr, (row_arr, col_arr)),
                          (rows*cols, rows*cols)).tocsc()]
    return mul_coeffs(coeffs, arg_coeffs)

def diag_vec_mat(expr, arg_coeffs):
    """Returns the coefficient matrix for DIAG_VEC linear op.

    Parameters
    ----------
    expr : LinOp
        The diag vec linear op.

    Returns
    -------
    list of SciPy CSC matrix
        The matrix representing placing a vector on a diagonal.
    """
    rows = get_matrix_shape(expr)[0]

    val_arr = []
    row_arr = []
    col_arr = []
    for i in range(rows):
        # Index in the diagonal matrix.
        row_arr.append(i*rows + i)
        # Index in the original vector.
        col_arr.append(i)
        val_arr.append(1.0)

    coeffs = [sp.coo_matrix((val_arr, (row_arr, col_arr)),
                            (rows**2, rows)).tocsc()]
    return mul_coeffs(coeffs, arg_coeffs)


def diag_mat_mat(expr, arg_coeffs):
    """Returns the coefficients matrix for DIAG_MAT linear op.

    Parameters
    ----------
    expr : LinOp
        The diag mat linear op.

    Returns
    -------
    SciPy CSC matrix
        The matrix to extract the diagonal from a matrix.
    """
    rows = get_matrix_shape(expr)[0]

    val_arr = []
    row_arr = []
    col_arr = []
    for i in range(rows):
        # Index in the original matrix.
        col_arr.append(i*rows + i)
        # Index in the extracted vector.
        row_arr.append(i)
        val_arr.append(1.0)

    coeffs = [sp.coo_matrix((val_arr, (row_arr, col_arr)),
                            (rows, rows**2)).tocsc()]
    return mul_coeffs(coeffs, arg_coeffs)


def upper_tri_mat(expr, arg_coeffs):
    """Returns the coefficients matrix for UPPER_TRI linear op.

    Parameters
    ----------
    expr : LinOp
        The upper tri linear op.

    Returns
    -------
    SciPy CSC matrix
        The matrix to vectorize the upper triangle.
    """
    rows, cols = expr.args[0].shape

    val_arr = []
    row_arr = []
    col_arr = []
    count = 0
    for i in range(rows):
        for j in range(cols):
            if j > i:
                # Index in the original matrix.
                col_arr.append(j*rows + i)
                # Index in the extracted vector.
                row_arr.append(count)
                val_arr.append(1.0)
                count += 1

    entries = expr.shape[0]
    return [sp.coo_matrix((val_arr, (row_arr, col_arr)),
                          (entries, rows*cols)).tocsc()], range(len(expr.args))

def conv_mat(expr, arg_coeffs):
    """Returns the coefficient matrix for CONV linear op.

    Parameters
    ----------
    expr : LinOp
        The conv linear op.

    Returns
    -------
    list of NumPy matrices
        The matrix representing the convolution operation.
    """
    constant = const_mat(expr.data)
    # Cast to 1D.
    constant = intf.from_2D_to_1D(constant)
    if isinstance(constant, sym.SymMatrix):
        raise TypeError('Convolution of parameters and variables not currently supported') # TODO

    # Create a Toeplitz matrix with constant as columns.
    rows = expr.size[0]
    nonzeros = expr.data.size[0]
    toeplitz_col = np.zeros(rows)
    toeplitz_col[0:nonzeros] = constant

    cols = expr.args[0].size[0]
    toeplitz_row = np.zeros(cols)
    toeplitz_row[0] = constant[0]
    coeff = sp_la.toeplitz(toeplitz_col, toeplitz_row)

    return [np.matrix(coeff)], range(len(expr.args))

def kron_mat(expr, arg_coeffs):
    """Returns the coefficient matrix for KRON linear op.

    Parameters
    ----------
    expr : LinOp
        The conv linear op.

    Returns
    -------
    list of SciPy CSC matrix
        The matrix representing the Kronecker product.
    """
    constant = const_mat(expr.data)
    if isinstance(constant, sym.SymMatrix):
        raise TypeError('Kronecker product of parameters and variables not currently supported') # TODO
    lh_rows, lh_cols = constant.shape
    rh_rows, rh_cols = expr.args[0].size
    # Stack sections for each column of the output.
    col_blocks = []
    for j in range(lh_cols):
        # Vertically stack A_{ij}Identity.
        blocks = []
        for i in range(lh_rows):
            blocks.append(constant[i, j]*sp.eye(rh_rows))
        column = sp.vstack(blocks)
        # Make block diagonal matrix by repeating column.
        col_blocks.append( sp.block_diag(rh_cols*[column]) )
    coeff = sp.vstack(col_blocks).tocsc()

    return [coeff], range(len(expr.args))


def hstack_mat(expr, arg_coeffs):
    return stack_mat(expr, arg_coeffs, False)

def vstack_mat(expr, arg_coeffs):
    return stack_mat(expr, arg_coeffs, True)


def stack_mat(expr, arg_coeffs, vertical):
    """Returns the coefficient matrices for VSTACK or HSTACK linear op.

    Parameters
    ----------
    expr : LinOp
        The stacked linear op.
    vertical : bool
        Is the stacking vertical?

    Returns
    -------
    list of SciPy CSC matrix
        The matrices representing the stacking operation.
    """
    expr_shape = get_matrix_shape(expr)

    offset = 0
    coeffs = []
    # Make a coefficient for each argument:
    # an identity with an offset.
    for arg in expr.args:
        arg_shape = get_matrix_shape(arg)
        val_arr = []
        row_arr = []
        col_arr = []
        # In hstack, the arguments are laid out in order.
        # In vstack, the arguments' columns are interleaved.
        if vertical:
            col_offset = expr_shape[0]
            offset_incr = arg_shape[0]
        else:
            col_offset = arg_shape[0]
            offset_incr = arg.size

        for i in range(arg_shape[0]):
            for j in range(arg_shape[1]):
                row_arr.append(i + j*col_offset + offset)
                col_arr.append(i + j*arg_shape[0])
                val_arr.append(1)

        shape = (expr.size, arg.size)
        coeff = sp.coo_matrix((val_arr, (row_arr, col_arr)), shape).tocsc()
        coeffs.append(coeff)
        offset += offset_incr
    return mul_coeffs(coeffs, arg_coeffs)




GET_COEFFS = {AddExpression : add_mat,
              DivExpression : div_mat,
              diag_mat      : diag_mat_mat,
              diag_vec      : diag_vec_mat,
              Hstack        : hstack_mat,
              index         : index_mat,
              kron          : kron_mat,
              MulExpression : mul_mat,
              multiply      : multiply_mat,
              NegExpression : neg_mat,
              Promote       : promote_mat,
              reshape       : reshape_mat,
              Sum           : sum_mat,
              trace         : trace_mat,
              transpose     : transpose_mat,
              Vstack        : vstack_mat}


def get_coeffs(expr, arg_coeffs):
    if not type(expr) in GET_COEFFS.keys():
        raise TypeError("Expressions involving "
                        "the %s atom not supported." % str(type(expr)))
    return GET_COEFFS[type(expr)](expr, arg_coeffs)
