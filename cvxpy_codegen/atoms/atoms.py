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

from cvxpy_codegen.atoms.abs import atomdata_abs
from cvxpy_codegen.atoms.add import atomdata_add, coeffdata_add
from cvxpy_codegen.atoms.diag_vec import atomdata_diag_vec
from cvxpy_codegen.atoms.diag_mat import atomdata_diag_mat
from cvxpy_codegen.atoms.hstack import atomdata_hstack
from cvxpy_codegen.atoms.index import atomdata_index, coeffdata_index
from cvxpy_codegen.atoms.max_entries import atomdata_max_entries
from cvxpy_codegen.atoms.mul import atomdata_mul
from cvxpy_codegen.atoms.mul_elemwise import atomdata_mul_elemwise
from cvxpy_codegen.atoms.neg import atomdata_neg, coeffdata_neg
from cvxpy_codegen.atoms.pos import atomdata_pos
from cvxpy_codegen.atoms.reshape import atomdata_reshape
from cvxpy_codegen.atoms.trace import atomdata_trace
from cvxpy_codegen.atoms.vstack import atomdata_vstack


# Each atom module must have a "atomdata_XXX" function.
# Modules for affine atoms must also have a "atomdata_XXX_coeffs" function.
import cvxpy_codegen.atoms.abs
import cvxpy_codegen.atoms.add
import cvxpy_codegen.atoms.diag_vec
import cvxpy_codegen.atoms.diag_mat
import cvxpy_codegen.atoms.hstack
import cvxpy_codegen.atoms.index
import cvxpy_codegen.atoms.max_entries
import cvxpy_codegen.atoms.mul
import cvxpy_codegen.atoms.mul_elemwise
import cvxpy_codegen.atoms.neg
import cvxpy_codegen.atoms.pos
import cvxpy_codegen.atoms.reshape
import cvxpy_codegen.atoms.trace
import cvxpy_codegen.atoms.vstack

from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.trace import trace
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.diag import diag_vec, diag_mat
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.index import index
from cvxpy.atoms.affine.mul_elemwise import mul_elemwise
from cvxpy.atoms import *


GET_ATOM_DATA = {MulExpression : atomdata_mul,
                 AddExpression : atomdata_add,
                 pos           : atomdata_pos,
                 vstack        : atomdata_vstack,
                 hstack        : atomdata_hstack,
                 mul_elemwise  : atomdata_mul_elemwise,
                 index         : atomdata_index,
                 diag_vec      : atomdata_diag_vec,
                 diag_mat      : atomdata_diag_mat,
                 NegExpression : atomdata_neg,
                 reshape       : atomdata_reshape,
                 abs           : atomdata_abs,
                 trace         : atomdata_trace,
                 max_entries   : atomdata_max_entries }


GET_COEFF_DATA = {'sum'      : coeffdata_add,
                  'index'    : coeffdata_index,
                  'neg'      : coeffdata_neg}


GET_ATOM_DATA_FROM_LINOP = { 'sum'      : atomdata_add,
                             'index'    : atomdata_index,
                             'neg'      : atomdata_neg}


def get_atom_data(expr, arg_data):
    if not type(expr) in GET_ATOM_DATA.keys():
        raise TypeError("Constant expressions involving "
                        "the %s atom not supported." % str(type(expr)))
    return GET_ATOM_DATA[type(expr)](expr, arg_data)


def get_coeff_data(linop_data, arg_data, vid):
    if not linop_data.opname in GET_COEFF_DATA.keys():
        raise TypeError("Evaluating linear coefficients for "
                        "atom %s not supported." % str(linop.opname))
    return GET_COEFF_DATA[linop_data.opname](linop_data, arg_data, vid)


def get_atom_data_from_linop(linop_data, arg_data):
    if not linop_data.opname in GET_COEFF_DATA.keys():
        raise TypeError("Constant functions involving"
                        "the %s atom not supported." % str(linop.opname))
    return GET_ATOM_DATA_FROM_LINOP[linop_data.opname](linop_data, arg_data)
